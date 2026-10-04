"""Intermediate representation built from CAD + spec; every writer reads it, none re-reads the CAD.
Each link frame sits at its parent joint's origin (URDF convention).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml

from .step import UNIT_TO_M, JointCandidate, Part, infer_joints, load_parts, mass_properties, tessellate


DEFAULT_RGBA = {
    "aluminum": "0.72 0.74 0.78 1",
    "steel": "0.30 0.31 0.34 1",
    "pla": "0.95 0.45 0.15 1",
}


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
    parts: list[Part] = field(default_factory=list)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world, zero config
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # world, zero config
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))  # link frame
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about COM
    visuals: dict[str, trimesh.Trimesh] = field(default_factory=dict)  # key -> mesh (link frame)
    visual_materials: dict[str, str] = field(default_factory=dict)  # key -> material name (default: key)
    collisions: list[CollisionGeom] = field(default_factory=list)
    source_collisions: list[CollisionGeom] = field(default_factory=list)  # from an input URDF
    collision_mode: str = ""
    collision_metrics: dict = field(default_factory=dict)

    def pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.rotation, self.origin
        return T

    def to_link(self, world_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Copy of a zero-configuration world-frame mesh, expressed in this link's frame."""
        m = world_mesh.copy()
        m.apply_transform(np.linalg.inv(self.pose()))
        return m

    def material(self, key: str) -> str:
        return self.visual_materials.get(key, key)

    def mesh(self) -> trimesh.Trimesh:
        """All parts concatenated, link frame."""
        return trimesh.util.concatenate(list(self.visuals.values()))


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # world, zero config (== child link origin)
    axis: np.ndarray  # unit, world frame at zero config
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
    materials: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RGBA))  # name -> "r g b a"
    floating_base: bool = False
    # loop closures: {name, link1, link2, anchor1, anchor2} (anchors in link frames)
    closures: list[dict] = field(default_factory=list)

    def rotate(self, R: np.ndarray) -> None:
        """Re-orient the robot (e.g. a Y-up export for Z-up simulators)."""
        T = np.eye(4)
        T[:3, :3] = R
        for link in self.links.values():
            link.origin = R @ link.origin
            link.rotation = R @ link.rotation
            for p in link.parts:
                if p.mesh is not None:
                    p.mesh = p.mesh.copy()
                    p.mesh.apply_transform(T)
                p.com = R @ p.com
                p.inertia = R @ p.inertia @ R.T
        for j in self.joints.values():
            j.origin = R @ j.origin
            j.axis = R @ j.axis
        # URDF can't store the root link's orientation: bake it into the root's own geometry instead
        root = self.links[self.root]
        Rr = root.rotation
        if not np.allclose(Rr, np.eye(3)):
            Tr = np.eye(4)
            Tr[:3, :3] = Rr
            for k, m in root.visuals.items():
                m = m.copy()
                m.apply_transform(Tr)
                root.visuals[k] = m
            for g in [*root.collisions, *root.source_collisions]:
                if g.kind == "mesh" and g.mesh is not None:  # vertices are already in the link frame
                    g.mesh = g.mesh.copy()
                    g.mesh.apply_transform(Tr)
                else:
                    g.transform = Tr @ g.transform
            for c in self.closures:
                for k in (1, 2):
                    if c[f"link{k}"] == self.root:
                        c[f"anchor{k}"] = Rr @ c[f"anchor{k}"]
            root.com = Rr @ root.com
            root.inertia = Rr @ root.inertia @ Rr.T
            root.rotation = np.eye(3)

    def used_materials(self) -> dict[str, str]:
        """Every material a visual references, with a colour (neutral grey for ones not in ``materials``,
        e.g. the "default" material of a drafted STEP spec)."""
        out = dict(self.materials)
        for link in self.links.values():
            for key in link.visuals:
                out.setdefault(link.material(key), "0.7 0.7 0.72 1")
        return out

    def child_in_parent(self, j: "Joint") -> tuple[np.ndarray, np.ndarray]:
        """(xyz, R) of the child link frame expressed in the parent link frame."""
        p, c = self.links[j.parent], self.links[j.child]
        return p.rotation.T @ (c.origin - p.origin), p.rotation.T @ c.rotation

    def axis_local(self, j: "Joint") -> np.ndarray:
        """Joint axis in the child link (== joint) frame, as URDF and MJCF want it."""
        a = self.links[j.child].rotation.T @ j.axis
        return np.where(np.abs(a) < 1e-9, 0.0, a)

    def moving_joints(self) -> list["Joint"]:
        """Non-fixed joints in depth-first link order (== MuJoCo qpos / SAPIEN active-joint order)."""
        out = []
        for n in self.ordered_links():
            j = self.parent_joint(n)
            if j is not None and j.type != "fixed":
                out.append(j)
        return out

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


def combine_inertia(parts: list[Part], frame_origin: np.ndarray):
    """Total mass, COM (link frame) and inertia about the COM via the parallel-axis theorem."""
    parts = [p for p in parts if p.mass > 0 and np.isfinite(p.mass) and np.isfinite(p.inertia).all()]
    mass = sum(p.mass for p in parts)
    if mass <= 0:  # nothing usable: a massless link (check_inertia reports it)
        return 0.0, np.zeros(3), np.zeros((3, 3))
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
    if not np.isfinite(inertia).all():
        return issues + [f"{name}: inertia is not finite"]
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
    if str(spec["source"]).startswith("http"):
        from .frontends import build_from_onshape

        return build_from_onshape(spec, base)
    if str(spec["source"]).lower().endswith(".urdf"):
        from .frontends import build_from_urdf

        return build_from_urdf(spec, base)
    scale = UNIT_TO_M[spec.get("units", "mm")]
    tess = spec.get("visual", {}).get("tessellation", {})

    # 1. parts, materials, link assignment
    parts = load_parts(base / spec["source"])
    ignore = spec.get("ignore_parts", [])  # placeholder geometry: keep-out zones, reference bodies
    parts = [p for p in parts if not any(fnmatch.fnmatch(p.name, pat) for pat in ignore)]
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
        mass_properties(p, scale)
        tessellate(p, scale, tess.get("linear_mm", 0.2), tess.get("angular_deg", 10))

    links = {name: Link(name) for name in spec["links"]}
    for p in parts:
        links[p.link].parts.append(p)

    # 2. geometric joint candidates between links
    candidates = infer_joints(parts, scale)

    # 3. joints: spec wins, geometry fills in axis/origin
    joints = {}
    for jname, js in spec["joints"].items():
        pair = sorted((js["parent"], js["child"]))
        cand = next((c for c in candidates if sorted((c.link_a, c.link_b)) == pair), None)  # either order
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
            by_mat.setdefault(p.material, []).append(link.to_link(p.mesh))
        link.visuals = {mat: trimesh.util.concatenate(ms) for mat, ms in by_mat.items()}

    robot = Robot(spec["robot"], spec, links, joints, candidates, roots[0],
                  floating_base=spec.get("base", "fixed") == "floating")
    for cname, c in (spec.get("closures") or {}).items():
        world = np.r_[np.array(c["point"], float) * scale, 1.0]
        robot.closures.append({"name": cname, "link1": c["link1"], "link2": c["link2"],
                               "anchor1": (np.linalg.inv(links[c["link1"]].pose()) @ world)[:3],
                               "anchor2": (np.linalg.inv(links[c["link2"]].pose()) @ world)[:3]})
    if spec.get("root_rpy"):  # e.g. [1.5708, 0, 0] for a Y-up export
        from scipy.spatial.transform import Rotation

        robot.rotate(Rotation.from_euler("xyz", spec["root_rpy"]).as_matrix())
    return robot
