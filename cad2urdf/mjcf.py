"""Native MJCF writer (MuJoCo / MJX / MuJoCo Warp / Newton's MJCF importer).

Written directly from the IR instead of compiling the URDF, because MJCF can
carry what URDF cannot: armature, actuators with gains, equality constraints
for mimic joints, contact excludes from the SRDF, keyframes and per-geom
contact parameters.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .model import Robot
from .urdf import MATERIAL_RGBA, fmt


def _quat(T: np.ndarray) -> str:
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return fmt((w, x, y, z))


def build_mjcf(
    robot: Robot,
    meshdir: str = "../meshes",
    excludes: list[tuple[str, str]] = (),
    keyframes: dict[str, dict[str, float]] | None = None,
    floor: bool = True,
    filterparent: bool = True,
    collision_only: bool = False,
) -> ET.ElementTree:
    spec = robot.spec
    fr = spec.get("contact", {}).get("friction", [1.0, 0.005, 0.0001])
    root = ET.Element("mujoco", model=robot.name)
    ET.SubElement(root, "compiler", angle="radian", meshdir=meshdir, autolimits="true")
    opt = ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    if not filterparent:
        ET.SubElement(opt, "flag", filterparent="disable")

    default = ET.SubElement(root, "default")
    vis = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(vis, "geom", contype="0", conaffinity="0", group="2", density="0")
    col = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(col, "geom", group="3", friction=fmt(fr), condim="4", density="0", rgba="0.2 0.6 1 0.4")

    asset = ET.SubElement(root, "asset")
    for mat, rgba in MATERIAL_RGBA.items():
        ET.SubElement(asset, "material", name=mat, rgba=rgba)
    for link in robot.links.values():
        if not collision_only:
            for mat in link.visuals:
                ET.SubElement(asset, "mesh", name=f"{link.name}_{mat}", file=f"visual/{link.name}_{mat}.stl")
        for i, g in enumerate(link.collisions):
            if g.kind == "mesh":
                ET.SubElement(asset, "mesh", name=f"{link.name}_col{i}", file=f"collision/{link.name}_{i}.stl")

    world = ET.SubElement(root, "worldbody")
    if floor:
        ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
        ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1", rgba="0.9 0.9 0.9 1")

    def add_body(parent_el: ET.Element, name: str):
        link = robot.links[name]
        j = robot.parent_joint(name)
        pos = link.origin - (robot.links[j.parent].origin if j else np.zeros(3))
        body = ET.SubElement(parent_el, "body", name=name, pos=fmt(pos))
        I = link.inertia
        ET.SubElement(body, "inertial", pos=fmt(link.com), mass=f"{link.mass:.6g}",
                      fullinertia=fmt((I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])))
        if j is not None:
            attrs = dict(name=j.name, type="hinge" if j.type == "revolute" else "slide",
                         axis=fmt(np.where(np.abs(j.axis) < 1e-9, 0, j.axis)), range=fmt((j.lower, j.upper)),
                         damping=f"{j.damping:.6g}", frictionloss=f"{j.friction:.6g}", armature=f"{j.armature:.6g}")
            if j.effort:
                attrs["actuatorfrcrange"] = fmt((-j.effort, j.effort))
            ET.SubElement(body, "joint", attrs)
        if not collision_only:
            for mat in link.visuals:
                ET.SubElement(body, "geom", {"class": "visual", "type": "mesh", "mesh": f"{name}_{mat}", "material": mat})
        for i, g in enumerate(link.collisions):
            a = {"class": "collision", "name": f"{name}_col{i}"}
            if g.kind == "mesh":
                a.update(type="mesh", mesh=f"{name}_col{i}")
            else:
                a.update(pos=fmt(g.transform[:3, 3]), quat=_quat(g.transform))
                if g.kind == "box":
                    a.update(type="box", size=fmt(np.array(g.size) / 2))
                elif g.kind == "cylinder":
                    a.update(type="cylinder", size=fmt((g.size[0], g.size[1] / 2)))
                else:
                    a.update(type="sphere", size=fmt(g.size[:1]))
            ET.SubElement(body, "geom", a)
        for child in [jj.child for jj in robot.joints.values() if jj.parent == name]:
            add_body(body, child)

    add_body(world, robot.root)
    # a site at the tool centre point, handy for IK / task code
    tcp = world.find(f".//body[@name='{robot.spec['srdf']['end_effector']['parent_link']}']") if "srdf" in spec else None
    if tcp is not None:
        ET.SubElement(tcp, "site", name="tcp", pos="0 0 0.12", size="0.005")

    if excludes:
        contact = ET.SubElement(root, "contact")
        for a, b in excludes:
            ET.SubElement(contact, "exclude", body1=a, body2=b)

    mimics = [j for j in robot.joints.values() if j.mimic]
    if mimics:
        eq = ET.SubElement(root, "equality")
        for j in mimics:
            m = j.mimic
            ET.SubElement(eq, "joint", joint1=j.name, joint2=m["joint"],
                          polycoef=fmt((m.get("offset", 0.0), m.get("multiplier", 1.0), 0, 0, 0)))

    act = ET.SubElement(root, "actuator")
    ctrl_joints = []
    for j in robot.joints.values():
        a = j.actuator
        if a.get("kind", "none") == "none":
            continue
        ctrl_joints.append(j.name)
        ET.SubElement(act, "position", name=j.name, joint=j.name, kp=f"{a['kp']:.6g}", kv=f"{a.get('kv', 0):.6g}",
                      ctrlrange=fmt((j.lower, j.upper)), forcerange=fmt((-j.effort, j.effort)))

    if keyframes:
        kf = ET.SubElement(root, "keyframe")
        order = [j.name for j in _joint_order(robot)]
        for kname, q in keyframes.items():
            qpos = [q.get(n, 0.0) for n in order]
            ctrl = [q.get(n, 0.0) for n in ctrl_joints]
            ET.SubElement(kf, "key", name=kname, qpos=fmt(qpos), ctrl=fmt(ctrl))

    ET.indent(root)
    return ET.ElementTree(root)


def _joint_order(robot: Robot):
    """Joints in MJCF depth-first body order (== qpos order)."""
    return [robot.parent_joint(n) for n in robot.ordered_links() if robot.parent_joint(n)]


def write_mjcf(robot: Robot, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    build_mjcf(robot, **kw).write(path, encoding="unicode")
