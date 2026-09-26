"""Neutral intermediate representation (IR) built from CAD + spec.

Every exporter (URDF, SRDF, MJCF, Gazebo, Isaac Lab, ManiSkill) reads this IR;
none of them re-reads the CAD. Frames follow the URDF convention: each link
frame sits at its parent joint's origin, axis-aligned with the world in the
zero configuration of the CAD model.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml

from . import cad
from .joints import JointCandidate, infer_joints


@dataclass
class CollisionGeom:
    kind: str  # box | cylinder | sphere | mesh
    transform: np.ndarray  # 4x4 in link frame
    size: tuple = ()  # box: (x,y,z) full extents; cylinder: (radius, length); sphere: (radius,)
    mesh: trimesh.Trimesh | None = None  # for kind == mesh (link frame)
    source: str = ""


@dataclass
class Link:
    name: str
    parts: list[cad.Part] = field(default_factory=list)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world, zero config
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))  # link frame
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about COM
    visuals: dict[str, trimesh.Trimesh] = field(default_factory=dict)  # material -> mesh (link frame)
    collisions: list[CollisionGeom] = field(default_factory=list)
    collision_mode: str = ""
    collision_metrics: dict = field(default_factory=dict)

    def mesh(self) -> trimesh.Trimesh:
        """All parts concatenated, link frame."""
        return trimesh.util.concatenate(list(self.visuals.values()))


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # world, zero config
    axis: np.ndarray  # unit, world == parent frame orientation
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0
    damping: float = 0.0
    friction: float = 0.0
    armature: float = 0.0
    mimic: dict | None = None
    actuator: dict = field(default_factory=dict)
    candidate: JointCandidate | None = None


@dataclass
class Robot:
    name: str
    spec: dict
    links: dict[str, Link]
    joints: dict[str, Joint]
    candidates: list[JointCandidate]
    root: str

    def parent_joint(self, link: str) -> Joint | None:
        return next((j for j in self.joints.values() if j.child == link), None)

    def ordered_links(self) -> list[str]:
        order, stack = [], [self.root]
        while stack:
            name = stack.pop()
            order.append(name)
            stack.extend(reversed([j.child for j in self.joints.values() if j.parent == name]))
        return order


def _match(name: str, table: dict) -> str | None:
    for pattern, value in table.items():
        if fnmatch.fnmatch(name, pattern):
            return value
    return None


def _per(spec_section: dict, key: str) -> dict:
    out = dict(spec_section.get("default", {}))
    out.update(spec_section.get(key, {}))
    return out


def combine_inertia(parts: list[cad.Part], frame_origin: np.ndarray):
    """Total mass, COM (link frame) and inertia about the COM via the parallel-axis theorem."""
    mass = sum(p.mass for p in parts)
    com_world = sum(p.mass * p.com for p in parts) / mass
    inertia = np.zeros((3, 3))
    for p in parts:
        r = p.com - com_world
        inertia += p.inertia + p.mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return mass, com_world - frame_origin, inertia


def check_inertia(name: str, inertia: np.ndarray, mass: float) -> list[str]:
    """Physical validity checks every simulator eventually enforces (MuJoCo at compile time)."""
    issues = []
    if mass <= 0:
        issues.append(f"{name}: non-positive mass {mass}")
    if not np.allclose(inertia, inertia.T, atol=1e-12):
        issues.append(f"{name}: inertia not symmetric")
    ev = np.linalg.eigvalsh(inertia)
    if ev.min() <= 0:
        issues.append(f"{name}: inertia not positive definite {ev}")
    a, b, c = sorted(ev)
    if a + b < c * (1 - 1e-9):
        issues.append(f"{name}: principal moments violate triangle inequality {ev}")
    return issues


def build(spec_path: Path) -> Robot:
    spec = yaml.safe_load(Path(spec_path).read_text())
    base = Path(spec_path).parent
    scale = cad.UNIT_TO_M[spec.get("units", "mm")]
    tess = spec.get("visual", {}).get("tessellation", {})

    # 1. parts, materials, link assignment
    parts = cad.load_parts(base / spec["source"])
    link_of = {}
    for link_name, patterns in spec["links"].items():
        for p in parts:
            if any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
                if p.name in link_of:
                    raise ValueError(f"part {p.name} matches links {link_of[p.name]} and {link_name}")
                link_of[p.name] = link_name
    unassigned = [p.name for p in parts if p.name not in link_of]
    if unassigned:
        raise ValueError(f"parts not assigned to any link: {unassigned}")
    for p in parts:
        p.link = link_of[p.name]
        p.material = _match(p.name, spec["part_materials"]) or "aluminum"
        p.density = spec["materials"][p.material]
        cad.mass_properties(p, scale)
        cad.tessellate(p, scale, tess.get("linear_mm", 0.2), tess.get("angular_deg", 10))

    links = {name: Link(name) for name in spec["links"]}
    for p in parts:
        links[p.link].parts.append(p)

    # 2. geometric joint candidates between links
    candidates = infer_joints(parts, scale)

    # 3. joints: spec wins, geometry fills in axis/origin
    joints = {}
    for jname, js in spec["joints"].items():
        pair = sorted((js["parent"], js["child"]))
        cand = next((c for c in candidates if [c.link_a, c.link_b] == pair), None)
        axis = np.array(js["axis"], float) if isinstance(js.get("axis"), list) else None
        origin = np.array(js["origin"], float) * scale if isinstance(js.get("origin"), list) else None
        if axis is None or origin is None:
            if cand is None:
                raise ValueError(f"joint {jname}: no axis/origin in spec and no geometric candidate")
            axis = cand.direction if axis is None else axis
            origin = cand.origin if origin is None else origin
            if js["type"] == "revolute" and cand.type_hint == "cylindrical":
                print(f"  note: {jname} is a cylindrical interface; spec says revolute")
        axis = axis / np.linalg.norm(axis) * js.get("axis_sign", 1)
        dyn = _per(spec.get("dynamics", {}), jname)
        act = _per(spec.get("actuators", {}), jname)
        lower, upper = js.get("limits", [0.0, 0.0])
        joints[jname] = Joint(
            name=jname, type=js["type"], parent=js["parent"], child=js["child"],
            origin=origin, axis=axis, lower=lower, upper=upper,
            effort=js.get("effort", 0.0), velocity=js.get("velocity", 0.0),
            damping=dyn.get("damping", 0.0), friction=dyn.get("friction", 0.0),
            armature=dyn.get("armature", 0.0), mimic=js.get("mimic"), actuator=act, candidate=cand,
        )

    children = {j.child for j in joints.values()}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise ValueError(f"kinematic tree must have exactly one root, got {roots}")
    for j in joints.values():
        links[j.child].origin = j.origin

    # 4. per-link mass properties and visual meshes in link frames
    for link in links.values():
        link.mass, link.com, link.inertia = combine_inertia(link.parts, link.origin)
        for issue in check_inertia(link.name, link.inertia, link.mass):
            print("  WARNING", issue)
        by_mat: dict[str, list] = {}
        for p in link.parts:
            m = p.mesh.copy()
            m.apply_translation(-link.origin)
            by_mat.setdefault(p.material, []).append(m)
        link.visuals = {mat: trimesh.util.concatenate(ms) for mat, ms in by_mat.items()}

    return Robot(spec["robot"], spec, links, joints, candidates, roots[0])
