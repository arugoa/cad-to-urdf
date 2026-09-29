"""URDF front end: an exporter's URDF (Onshape export, onshape-to-robot, sw2robot, ACDC4Robot, creo2urdf).

Keeps the exporter's frames, joints, limits and inertia; each <visual> becomes one part for the per-part
collision modes. Spec keys: ``source``, ``package_dirs: {pkg: path}``, ``base: fixed|floating``,
``root_rpy``, and the usual joints / dynamics / actuators / collision / srdf overrides.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from . import cad
from .util import slug
from .model import CollisionGeom, Joint, Link, Robot, _per, check_inertia


def _pose(el: ET.Element | None) -> np.ndarray:
    T = np.eye(4)
    if el is None:
        return T
    T[:3, 3] = [float(x) for x in el.get("xyz", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", [float(x) for x in el.get("rpy", "0 0 0").split()]).as_matrix()
    return T


def _floats(s: str | None, n: int, default=0.0) -> list[float]:
    return [float(x) for x in s.split()] if s else [default] * n


class _Resolver:
    """package:// and relative mesh paths, the usual source of broken exports."""

    def __init__(self, urdf: Path, package_dirs: dict[str, str]):
        self.urdf_dir = urdf.parent.resolve()
        self.package_dirs = {k: Path(v).expanduser() for k, v in package_dirs.items()}

    def __call__(self, filename: str) -> Path:
        if filename.startswith("file://"):
            return Path(filename[7:])
        if not filename.startswith("package://"):
            p = Path(filename)
            return p if p.is_absolute() else self.urdf_dir / p
        pkg, _, rest = filename[len("package://"):].partition("/")
        candidates = []
        if pkg in self.package_dirs:
            candidates.append(self.package_dirs[pkg] / rest)
        for d in [self.urdf_dir, *self.urdf_dir.parents][:4]:
            candidates += [d / rest, d / pkg / rest]
            if d.name == pkg:
                candidates.append(d / rest)
        for c in candidates:
            if c.exists():
                return c
        hits = list(self.urdf_dir.rglob(Path(rest).name))
        if len(hits) == 1:
            return hits[0]
        raise FileNotFoundError(f"cannot resolve {filename}; add it to package_dirs in the spec")


def _geometry(geo: ET.Element, resolve: _Resolver) -> tuple[str, tuple, trimesh.Trimesh | None]:
    """(kind, size, mesh-in-element-frame)."""
    child = geo[0]
    if child.tag == "box":
        size = tuple(_floats(child.get("size"), 3))
        return "box", size, trimesh.creation.box(extents=size)
    if child.tag == "cylinder":
        r, l = float(child.get("radius")), float(child.get("length"))
        return "cylinder", (r, l), trimesh.creation.cylinder(radius=r, height=l, sections=32)
    if child.tag == "sphere":
        r = float(child.get("radius"))
        return "sphere", (r,), trimesh.creation.icosphere(subdivisions=3, radius=r)
    if child.tag == "mesh":
        m = trimesh.load(resolve(child.get("filename")), force="mesh", process=True)
        if child.get("scale"):
            m.apply_scale(_floats(child.get("scale"), 3))
        return "mesh", (), m
    raise ValueError(f"unsupported geometry <{child.tag}>")



def build_from_urdf(spec: dict, base: Path) -> Robot:
    urdf_path = (base / spec["source"]).resolve()
    tree = ET.parse(urdf_path).getroot()
    resolve = _Resolver(urdf_path, spec.get("package_dirs", {}))

    named_colors = {}
    for m in tree.findall("material"):
        c = m.find("color")
        if c is not None:
            named_colors[m.get("name")] = c.get("rgba")

    # --- joints / tree, world poses at q = 0
    joint_els = tree.findall("joint")
    children = {j.find("child").get("link") for j in joint_els}
    link_names = [l.get("name") for l in tree.findall("link")]
    roots = [n for n in link_names if n not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, got {roots}")
    root = roots[0]
    T_root = np.eye(4)
    T_root[:3, :3] = Rotation.from_euler("xyz", spec.get("root_rpy", [0, 0, 0])).as_matrix()
    T_root[:3, 3] = spec.get("root_xyz", [0, 0, 0])
    world = {root: T_root}
    pending = list(joint_els)
    while pending:
        progressed = False
        for j in list(pending):
            p, c = j.find("parent").get("link"), j.find("child").get("link")
            if p in world:
                world[c] = world[p] @ _pose(j.find("origin"))
                pending.remove(j)
                progressed = True
        if not progressed:
            raise ValueError("URDF joints do not form a tree")

    # --- links
    robot_materials = {}
    links: dict[str, Link] = {}
    for lel in tree.findall("link"):
        name = lel.get("name")
        T = world[name]
        link = Link(name, origin=T[:3, 3].copy(), rotation=T[:3, :3].copy())
        inert = lel.find("inertial")
        if inert is not None and inert.find("mass") is not None:
            Ti = _pose(inert.find("origin"))
            ie = inert.find("inertia")
            g = lambda k: float(ie.get(k, 0.0)) if ie is not None else 0.0
            I = np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]])
            link.mass = float(inert.find("mass").get("value"))
            link.com = Ti[:3, 3]
            link.inertia = Ti[:3, :3] @ I @ Ti[:3, :3].T
        for i, v in enumerate(lel.findall("visual")):
            kind, size, m = _geometry(v.find("geometry"), resolve)
            m.apply_transform(_pose(v.find("origin")))
            key = f"{i}_{slug(v.get('name', ''))}" if v.get("name") else f"{i}"
            link.visuals[key] = m
            mat = v.find("material")
            rgba = None
            mat_name = f"{name}_{key}"
            if mat is not None:
                c = mat.find("color")
                rgba = c.get("rgba") if c is not None else named_colors.get(mat.get("name"))
                mat_name = slug(mat.get("name") or mat_name)
            robot_materials.setdefault(mat_name, rgba or "0.7 0.7 0.7 1")
            link.visual_materials[key] = mat_name
            world_mesh = m.copy()
            world_mesh.apply_transform(T)
            part = cad.Part(name=f"{name}/{key}", shape=None, link=name, mesh=world_mesh)
            part.volume = float(world_mesh.volume if world_mesh.is_watertight else world_mesh.convex_hull.volume)
            link.parts.append(part)
        for c in lel.findall("collision"):
            kind, size, m = _geometry(c.find("geometry"), resolve)
            To = _pose(c.find("origin"))
            if kind == "mesh":
                m.apply_transform(To)
                link.source_collisions.append(CollisionGeom("mesh", np.eye(4), mesh=m, source="input"))
            else:
                link.source_collisions.append(CollisionGeom(kind, To, size, source="input"))
        for issue in check_inertia(name, link.inertia, link.mass):
            print("  note (input URDF):", issue)
        links[name] = link

    # --- joints
    overrides = spec.get("joints", {}) or {}
    joints: dict[str, Joint] = {}
    for jel in joint_els:
        jname, jtype = jel.get("name"), jel.get("type")
        child = jel.find("child").get("link")
        ax = np.array(_floats(jel.find("axis").get("xyz") if jel.find("axis") is not None else None, 3, 0.0))
        if not ax.any():
            ax = np.array([1.0, 0, 0])
        axis_world = world[child][:3, :3] @ (ax / np.linalg.norm(ax))
        lim = jel.find("limit")
        la = (lambda k, d=0.0: float(lim.get(k, d)) if lim is not None else d)
        dyn_el = jel.find("dynamics")
        mim = jel.find("mimic")
        ov = overrides.get(jname, {})
        dyn = _per(spec.get("dynamics", {}), jname)
        lower, upper = ov.get("limits", [la("lower"), la("upper")])
        joints[jname] = Joint(
            name=jname, type=ov.get("type", jtype), parent=jel.find("parent").get("link"), child=child,
            origin=world[child][:3, 3].copy(), axis=axis_world * ov.get("axis_sign", 1),
            lower=lower, upper=upper,
            effort=ov.get("effort", la("effort")), velocity=ov.get("velocity", la("velocity")),
            damping=dyn.get("damping", float(dyn_el.get("damping", 0)) if dyn_el is not None else 0.0),
            friction=dyn.get("friction", float(dyn_el.get("friction", 0)) if dyn_el is not None else 0.0),
            armature=dyn.get("armature", 0.0),
            mimic=ov.get("mimic") or ({"joint": mim.get("joint"), "multiplier": float(mim.get("multiplier", 1)),
                                       "offset": float(mim.get("offset", 0))} if mim is not None else None),
            actuator=_per(spec.get("actuators", {}), jname),
        )

    r = Robot(spec.get("robot", tree.get("name")), spec, links, joints, [], root,
              floating_base=spec.get("base", "fixed") == "floating")
    r.materials.update(robot_materials)
    return r
