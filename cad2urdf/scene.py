"""Static scenery (competition fields, arenas) from STEP: surfaces welcome, no joints.

    python -m cad2urdf.scene field.step -o build/field [--units mm] [--no-validate]

Robot exports are solids; field exports are often surface soups (hundreds of
zero-thickness faces, e.g. meshes converted in Onshape). A robot route cannot
use them, so this path is separate. Every rule is deterministic:

* one "leaf" per STEP entity (solid or shell), tessellated in world metres;
* geometric duplicates (same bounds and area to 1 mm / 1 cm^2) are dropped;
* closed leaf: convex hull if it fills >= 90 % of its hull, else CoACD;
* flat leaf (< 1 mm thick): a box ``thickness`` thick placed *behind* the face
  (against its normal), so the walking surface stays exactly where the CAD has
  it; flat leaves smaller than ``min_area`` (decals, stencils, bevels) get no
  collision;
* open non-flat leaf: its convex hull, or an oriented box if the hull is thin.

Outputs: ``<name>.urdf`` (one fixed link), ``mjcf/<name>.xml`` (static world
geoms, includable), ``maniskill/<name>_scene.py`` (``build_<name>(scene)`` for
ManiSkill/SAPIEN actor builders), per-colour visual meshes (STL + GLB), convex
collision STLs, ``report.json`` and a drop-test ``validation.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from . import cad
from .collision import _cap_vertices, _coacd, proper_frame
from .model import CollisionGeom
from .urdf import fmt, rpy


@dataclass
class Leaf:
    label: str
    mesh: trimesh.Trimesh
    rgba: tuple
    closed: bool


def load_leaves(step: Path, units: str, lin_mm: float = 2.0, ang_rad: float = 0.3) -> list[Leaf]:
    from build123d import import_step

    scale = cad.UNIT_TO_M[units]
    root = import_step(str(step))
    out = []

    def walk(s):
        if s.children:
            for c in s.children:
                walk(c)
            return
        v, f = s.tessellate(lin_mm * 1e-3 / scale, ang_rad)
        if not f:
            return
        m = trimesh.Trimesh(np.array([[p.X, p.Y, p.Z] for p in v]) * scale, np.array(f), process=True)
        if m.is_watertight and m.volume < 0:
            m.invert()  # surface-model exports often come with inward normals
        c = getattr(s, "color", None)
        rgba = tuple(round(float(x), 3) for x in tuple(c)) if c is not None else (0.7, 0.7, 0.7, 1.0)
        out.append(Leaf(s.label or f"leaf_{len(out)}", m, rgba, bool(m.is_watertight)))

    walk(root)
    return out


def dedupe(leaves: list[Leaf]) -> tuple[list[Leaf], int]:
    seen, out = set(), []
    for l in leaves:
        key = (tuple(np.round(l.mesh.bounds.ravel(), 3)), round(l.mesh.area, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(l)
    return out, len(leaves) - len(out)


def _box(center, R, extents, source) -> CollisionGeom:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, center
    return CollisionGeom("box", T, tuple(float(e) for e in extents), source=source)


def leaf_collision(l: Leaf, cfg: dict, max_v: int) -> tuple[list[CollisionGeom], str]:
    m = l.mesh
    obb = m.bounding_box_oriented.primitive
    ext = np.array(obb.extents)
    T_obb = proper_frame(obb.transform)
    R = T_obb[:3, :3]
    thin_axis = int(np.argmin(ext))
    if l.closed and m.volume > 0:
        hull = m.convex_hull
        if m.volume / hull.volume >= cfg["convex_fill"]:
            return [CollisionGeom("mesh", np.eye(4), mesh=_cap_vertices(hull, max_v), source=f"{l.label} hull")], "hull"
        return _coacd(m, {"threshold": cfg["coacd_threshold"], "max_hulls": cfg["coacd_max_hulls"]}, max_v,
                      f"{l.label} coacd"), "coacd"
    if ext[thin_axis] < cfg["flat_tol"]:
        if m.area < cfg["min_area"]:
            return [], "skipped-small"
        n = R[:, thin_axis]
        # orient the box behind the face: the area-weighted normal points to open air
        mean_n = (m.face_normals * m.area_faces[:, None]).sum(0)
        if np.dot(mean_n, n) < 0:
            n = -n
        t = cfg["thickness"]
        e = ext.copy()
        e[thin_axis] = t
        center = T_obb[:3, 3] - n * (t / 2)
        return [_box(center, R, e, f"{l.label} slab")], "slab"
    hull = m.convex_hull
    if ext[thin_axis] < cfg["thickness"]:
        e = ext.copy()
        e[thin_axis] = cfg["thickness"]
        return [_box(T_obb[:3, 3], R, e, f"{l.label} obb")], "obb"
    return [CollisionGeom("mesh", np.eye(4), mesh=_cap_vertices(hull, max_v), source=f"{l.label} hull")], "open-hull"


DEFAULTS = {"thickness": 0.01, "flat_tol": 1e-3, "min_area": 0.002, "convex_fill": 0.9,
            "coacd_threshold": 0.05, "coacd_max_hulls": 16, "max_hull_vertices": 64}


def build(step: Path, out: Path, name: str, units: str = "mm", cfg: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    leaves = load_leaves(step, units)
    leaves, n_dupes = dedupe(leaves)
    geoms, kinds = [], {}
    for l in leaves:
        g, kind = leaf_collision(l, cfg, cfg["max_hull_vertices"])
        geoms += g
        kinds[kind] = kinds.get(kind, 0) + 1

    (out / "meshes" / "visual").mkdir(parents=True, exist_ok=True)
    (out / "meshes" / "collision").mkdir(parents=True, exist_ok=True)
    by_color: dict[tuple, list] = {}
    for l in leaves:
        by_color.setdefault(l.rgba, []).append(l.mesh)
    visuals = []
    for i, (rgba, ms) in enumerate(sorted(by_color.items())):
        m = trimesh.util.concatenate(ms)
        stem = f"{name}_c{i}"
        m.export(out / "meshes" / "visual" / f"{stem}.stl")
        g = m.copy()
        g.visual = trimesh.visual.ColorVisuals(g, face_colors=np.array([int(255 * x) for x in rgba]))
        g.export(out / "meshes" / "visual" / f"{stem}.glb")
        visuals.append((stem, rgba))
    for i, g in enumerate(geoms):
        if g.kind == "mesh":
            g.mesh.export(out / "meshes" / "collision" / f"{name}_{i}.stl")

    _write_urdf(out / f"{name}.urdf", name, visuals, geoms)
    _write_mjcf(out / "mjcf" / f"{name}.xml", name, visuals, geoms)
    _write_maniskill(out / "maniskill" / f"{name}_scene.py", name, visuals, geoms)
    lo, hi = np.array([l.mesh.bounds for l in leaves]).min((0, 1)), np.array([l.mesh.bounds for l in leaves]).max((0, 1))
    report = {"leaves": len(leaves) + n_dupes, "duplicates_dropped": n_dupes, "collision_rules": kinds,
              "collision_geoms": len(geoms), "boxes": sum(g.kind == "box" for g in geoms),
              "convex_meshes": sum(g.kind == "mesh" for g in geoms), "visual_meshes": len(visuals),
              "visual_triangles": int(sum(len(l.mesh.faces) for l in leaves)),
              "bounds_m": [np.round(lo, 3).tolist(), np.round(hi, 3).tolist()], "config": cfg}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _write_urdf(path: Path, name: str, visuals, geoms) -> None:
    root = ET.Element("robot", name=name)
    root.append(ET.Comment(" static scene generated by cad2urdf.scene: load with a fixed root "))
    link = ET.SubElement(root, "link", name=name)
    for stem, rgba in visuals:
        v = ET.SubElement(link, "visual", name=stem)
        ET.SubElement(ET.SubElement(v, "geometry"), "mesh", filename=f"meshes/visual/{stem}.stl")
        ET.SubElement(ET.SubElement(v, "material", name=stem), "color", rgba=fmt(rgba))
    for i, g in enumerate(geoms):
        c = ET.SubElement(link, "collision", name=f"{name}_col{i}")
        ET.SubElement(c, "origin", xyz=fmt(g.transform[:3, 3]), rpy=fmt(rpy(g.transform)))
        geo = ET.SubElement(c, "geometry")
        if g.kind == "box":
            ET.SubElement(geo, "box", size=fmt(g.size))
        else:
            ET.SubElement(geo, "mesh", filename=f"meshes/collision/{name}_{i}.stl")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def _quat(T):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return fmt((w, x, y, z))


def _write_mjcf(path: Path, name: str, visuals, geoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", meshdir="../meshes")
    asset = ET.SubElement(root, "asset")
    for stem, rgba in visuals:
        ET.SubElement(asset, "mesh", name=stem, file=f"visual/{stem}.stl")
    for i, g in enumerate(geoms):
        if g.kind == "mesh":
            ET.SubElement(asset, "mesh", name=f"{name}_col{i}", file=f"collision/{name}_{i}.stl")
    wb = ET.SubElement(root, "worldbody")
    for stem, rgba in visuals:
        ET.SubElement(wb, "geom", type="mesh", mesh=stem, rgba=fmt(rgba), contype="0", conaffinity="0", group="2")
    for i, g in enumerate(geoms):
        a = {"name": f"{name}_col{i}", "group": "3", "rgba": "0.2 0.6 1 0.3"}
        if g.kind == "box":
            a.update(type="box", size=fmt(np.array(g.size) / 2), pos=fmt(g.transform[:3, 3]), quat=_quat(g.transform))
        else:
            a.update(type="mesh", mesh=f"{name}_col{i}")
        ET.SubElement(wb, "geom", a)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def _write_maniskill(path: Path, name: str, visuals, geoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes = [(np.round(g.transform[:3, 3], 6).tolist(), [float(x) for x in _quat(g.transform).split()],
              np.round(np.array(g.size) / 2, 6).tolist()) for g in geoms if g.kind == "box"]
    meshes = [i for i, g in enumerate(geoms) if g.kind == "mesh"]
    fn = re.sub(r"\W", "_", name)
    path.write_text(f'''"""Static scene `{name}` for ManiSkill 3 / SAPIEN (generated by cad2urdf.scene).

    from {fn}_scene import build_{fn}
    build_{fn}(self.scene)          # in your env's _load_scene
"""

import os

import sapien

_MESH = os.path.join(os.path.dirname(__file__), "..", "meshes")
VISUALS = {[stem for stem, _ in visuals]!r}
BOXES = {boxes!r}  # (center, quat wxyz, half_size), metres
CONVEX = {meshes!r}  # meshes/collision/{name}_<i>.stl, each one convex piece


def build_{fn}(scene, pose=sapien.Pose(), collision=True, visual=True, static=True):
    b = scene.create_actor_builder()
    if visual:
        for stem in VISUALS:
            b.add_visual_from_file(os.path.join(_MESH, "visual", stem + ".glb"))
    if collision:
        for c, q, h in BOXES:
            b.add_box_collision(sapien.Pose(c, q), h)
        for i in CONVEX:
            b.add_convex_collision_from_file(os.path.join(_MESH, "collision", f"{name}_{{i}}.stl"))
    b.set_initial_pose(pose)
    return b.build_static(name="{name}") if static else b.build_kinematic(name="{name}")
''')


# ------------------------------------------------------------------ validation
def drop_test(out: Path, name: str, n: int = 200, seed: int = 0) -> dict:
    """Drop spheres onto random points; they must come to rest on the *visual* surface."""
    import mujoco

    base = ET.parse(out / "mjcf" / f"{name}.xml").getroot()
    visual = trimesh.util.concatenate([trimesh.load(p) for p in sorted((out / "meshes" / "visual").glob("*.stl"))])
    lo, hi = visual.bounds
    rng = np.random.default_rng(seed)
    r = 0.04
    pts, tops = [], []
    while len(pts) < n:
        xy = rng.uniform(lo[:2] + 0.3, hi[:2] - 0.3)
        hits, _, _ = visual.ray.intersects_location([[xy[0], xy[1], hi[2] + 1]], [[0, 0, -1]])
        if len(hits):
            pts.append(xy)
            tops.append(hits[:, 2].max())
    wb = base.find("worldbody")
    for i, (xy, z) in enumerate(zip(pts, tops)):
        b = ET.SubElement(wb, "body", name=f"ball{i}", pos=fmt((xy[0], xy[1], z + 0.15)))
        ET.SubElement(b, "freejoint")
        ET.SubElement(b, "geom", type="sphere", size=str(r), mass="0.1", contype="2", conaffinity="1")
    ET.SubElement(base, "option", timestep="0.002")
    tmp = out / "mjcf" / "_drop.xml"
    ET.ElementTree(base).write(tmp)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    d = mujoco.MjData(m)
    for _ in range(1000):
        mujoco.mj_step(m, d)
    pos = np.array([d.body(f"ball{i}").xpos for i in range(n)])
    z = pos[:, 2] - r
    err = z - np.array(tops)
    moved = np.linalg.norm(pos[:, :2] - np.array(pts), axis=1)
    miss = np.abs(err) >= 0.015
    rolled = miss & (moved > 0.05)  # slid/rolled off a narrow top (wall edge, pipe): physics, not a hole
    through = miss & ~rolled  # came down in place below the visual surface: a collision gap
    return {"balls": n, "landed_within_15mm": int((~miss).sum()), "rolled_off_narrow_tops": int(rolled.sum()),
            "fell_through_in_place": int(through.sum()),
            "gaps": [{"xy": np.round(pts[i], 3).tolist(), "visual_top_z": round(float(tops[i]), 3),
                      "rest_z": round(float(z[i]), 3)} for i in np.where(through)[0]],
            "median_err_mm": round(float(np.median(err[~miss])) * 1e3, 2)}


def sapien_load(out: Path, name: str) -> dict:
    import importlib.util

    import sapien

    spec = importlib.util.spec_from_file_location("s", out / "maniskill" / f"{name}_scene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    actor = getattr(mod, f"build_{re.sub(chr(92) + 'W', '_', name)}")(scene, visual=False)
    shapes = actor.find_component_by_type(sapien.physx.PhysxRigidStaticComponent).collision_shapes
    return {"ok": True, "collision_shapes": len(shapes)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cad2urdf.scene")
    ap.add_argument("step", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--units", default="mm")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)
    name = args.name or re.sub(r"\W", "_", args.step.stem.lower()).strip("_")
    if name[0].isdigit():
        name = "scene_" + name
    args.out.mkdir(parents=True, exist_ok=True)
    rep = build(args.step, args.out, name, args.units)
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, indent=2))
    if not args.no_validate:
        val = {"mujoco_drop_test": drop_test(args.out, name)}
        # separate process: SAPIEN's native libs can crash when loaded after CoACD/OpenCascade
        import subprocess

        code = f"import json; from cad2urdf.scene import sapien_load; from pathlib import Path; " \
               f"print(json.dumps(sapien_load(Path({str(args.out)!r}), {name!r})))"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        val["sapien"] = json.loads(lines[-1]) if lines else {"ok": False, "error": r.stderr.strip()[-300:]}
        (args.out / "validation.json").write_text(json.dumps(val, indent=2))
        print(json.dumps(val, indent=2))


if __name__ == "__main__":
    sys.exit(main())
