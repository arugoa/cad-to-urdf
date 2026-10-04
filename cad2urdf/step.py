"""STEP front end: loading, joint inference, the spec draft and the inspector.

* loading: solids, B-rep mass properties, tessellation
* joints: a shaft inside a bore on two links (coaxial cylindrical faces of nearly equal radius) gives a
  candidate with an axis, origin and a ``revolute`` / ``cylindrical`` hint; flat slides, ball joints and
  belt/gear drives have no such signature
* draft: touching parts are rigidly joined unless the contact is a joint (running fit, loose pin, bearing
  inner race, servo output horn, meshing gears); a min-cut removes incidental contacts from posed CAD;
  running fits become joints; every guess is a REVIEW line
* inspector: ``python -m cad2urdf.step robot.step [--spec spec.yaml]`` lists part groups, their contacts
  and shared axes as spec-ready ``axis`` / ``origin``, for writing a spec by hand
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepExtrema import BRepExtrema_ShapeProximity
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from build123d import GeomType, Shape, import_step

from .util import UnionFind, is_fastener


UNIT_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}


@dataclass
class Part:
    name: str
    shape: Shape  # exact B-rep, CAD units
    material: str = ""
    density: float = 0.0  # kg/m^3
    link: str = ""
    # SI mass properties, world frame (zero configuration)
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about COM
    volume: float = 0.0  # m^3
    mesh: trimesh.Trimesh | None = None  # world frame, metres
    skipped_faces: int = 0  # faces that failed to triangulate


def load_parts(step_path: Path) -> list[Part]:
    """Flatten a STEP assembly into uniquely named solids (multi-solid leaves get ``_0, _1, ...``)."""
    root = import_step(str(step_path))
    parts: list[Part] = []

    def walk(shape: Shape, prefix: str):
        if shape.children:
            for child in shape.children:
                walk(child, prefix)
            return
        solids = shape.solids()
        base = shape.label or f"part_{len(parts)}"
        for i, solid in enumerate(solids):
            name = base if len(solids) == 1 else f"{base}_{i}"
            parts.append(Part(name=name, shape=solid))

    walk(root, "")
    # exports repeat names (every screw): suffix duplicates
    seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    for p in parts:
        counts[p.name] = counts.get(p.name, 0) + 1
    for p in parts:
        if counts[p.name] > 1:
            seen[p.name] = seen.get(p.name, 0) + 1
            p.name = f"{p.name}#{seen[p.name]}"
    return parts


def mass_properties(part: Part, unit_scale: float) -> None:
    """Exact volume/COM/inertia from the B-rep, scaled to SI with the part density."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(part.shape.wrapped, props)
    vol_cad = props.Mass()  # with density 1 this is the volume, in CAD units^3
    c = props.CentreOfMass()
    m = props.MatrixOfInertia()  # about the centre of mass, CAD units^5
    inertia_cad = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    part.volume = vol_cad * unit_scale**3
    part.mass = part.density * part.volume
    part.com = np.array([c.X(), c.Y(), c.Z()]) * unit_scale
    part.inertia = inertia_cad * part.density * unit_scale**5


def tessellate(part: Part, unit_scale: float, linear_mm: float, angular_deg: float) -> None:
    """Mesh the part face by face; faces that fail to triangulate are counted in ``skipped_faces``."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    lin = linear_mm * 1e-3 / unit_scale  # tolerance expressed in CAD units
    BRepMesh_IncrementalMesh(part.shape.wrapped, lin, False, np.radians(angular_deg), True)
    verts, tris, skipped, offset = [], [], 0, 0
    exp = TopExp_Explorer(part.shape.wrapped, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        exp.Next()
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        if poly is None or poly.NbTriangles() == 0:
            skipped += 1
            continue
        trsf = loc.Transformation()
        pts = [poly.Node(i).Transformed(trsf) for i in range(1, poly.NbNodes() + 1)]
        verts += [(p.X(), p.Y(), p.Z()) for p in pts]
        rev = face.Orientation() == TopAbs_REVERSED
        for i in range(1, poly.NbTriangles() + 1):
            a, b, c = poly.Triangle(i).Get()
            tris.append((a - 1 + offset, c - 1 + offset, b - 1 + offset) if rev else (a - 1 + offset, b - 1 + offset, c - 1 + offset))
        offset += poly.NbNodes()
    mesh = trimesh.Trimesh(np.array(verts) * unit_scale, np.array(tris), process=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    part.mesh = mesh
    part.skipped_faces = skipped


@dataclass
class CylFace:
    part: str
    link: str
    radius: float  # m
    point: np.ndarray  # a point on the axis (m)
    direction: np.ndarray  # unit axis
    t0: float  # axial extent along `direction`, measured from the origin projection
    t1: float
    convex: bool  # True: shaft (outer surface); False: bore (hole)


@dataclass
class JointCandidate:
    link_a: str
    link_b: str
    direction: np.ndarray
    origin: np.ndarray
    radius: float
    engagement: float  # axial overlap of shaft and bore (m)
    shaft_length: float  # free shaft length: not held by its own link's bores (m)
    type_hint: str
    evidence: list[str] = field(default_factory=list)
    clearance: float = 0.0  # radial, smallest over the matched shaft/bore pairs (m); 0 => press fit

    def summary(self) -> str:
        d = np.round(self.direction, 3).tolist()
        o = np.round(self.origin, 4).tolist()
        return (
            f"{self.link_a} <-> {self.link_b}: {self.type_hint:<11} axis={d} origin={o} "
            f"r={self.radius * 1e3:.1f}mm engagement={self.engagement * 1e3:.1f}mm "
            f"free_shaft={self.shaft_length * 1e3:.1f}mm  [{', '.join(self.evidence)}]"
        )


def _v(vec) -> np.ndarray:
    return np.array([vec.X, vec.Y, vec.Z], dtype=float)


def _canonical(direction: np.ndarray) -> np.ndarray:
    d = direction / np.linalg.norm(direction)
    # flip so the largest component is positive -> parallel axes compare equal
    return -d if d[np.argmax(np.abs(d))] < 0 else d


def cylindrical_faces(part: Part, unit_scale: float) -> list[CylFace]:
    faces = []
    for f in part.shape.faces():
        if f.geom_type != GeomType.CYLINDER:
            continue
        # read the cylinder from OpenCascade: build123d's Face.radius is None for many exported faces
        try:
            cyl = BRepAdaptor_Surface(f.wrapped).Cylinder()
        except Exception:  # noqa: BLE001  (not an analytic cylinder after all)
            continue
        ax = cyl.Axis()
        radius = cyl.Radius()
        p0 = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()]) * unit_scale
        d = _canonical(np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()]))
        # axis point closest to the origin, so coaxial faces share it
        p0 = p0 - np.dot(p0, d) * d
        verts = np.array([_v(v) for v in f.vertices()]).reshape(-1, 3) * unit_scale
        if len(verts) == 0:  # full seamless cylinder: fall back to bounding box
            bb = f.bounding_box()
            verts = np.array([_v(bb.min), _v(bb.max)]) * unit_scale
        t = verts @ d
        c = _v(f.center()) * unit_scale
        n = _v(f.normal_at(f.center()))
        radial = c - (p0 + np.dot(c, d) * d)
        convex = float(np.dot(n, radial)) > 0
        faces.append(CylFace(part.name, part.link, radius * unit_scale, p0, d, t.min(), t.max(), convex))
    return faces


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -np.inf
    for lo, hi in sorted(intervals):
        if hi <= end:
            continue
        total += hi - max(lo, end)
        end = hi
    return total


def _coaxial(a: CylFace, b: CylFace, cos_tol: float, offset_tol: float) -> bool:
    return abs(np.dot(a.direction, b.direction)) >= cos_tol and np.linalg.norm(a.point - b.point) <= offset_tol


class _coaxial_index:
    """k-d tree over (direction, axis point, radius): near-coaxial face pairs without an O(F^2) scan.
    Its radius is looser than the exact tolerances, which the caller re-checks."""

    def __init__(self, faces, angle_tol_deg, offset_tol, radial_tol):
        from scipy.spatial import cKDTree

        self.faces = faces
        w_dir = offset_tol / max(np.sin(np.radians(angle_tol_deg)), 1e-9)
        w_rad = offset_tol / radial_tol
        self.r = offset_tol * 2.0
        self.id = {id(f): k for k, f in enumerate(faces)}
        pts = np.array([np.r_[f.direction * w_dir, f.point, f.radius * w_rad] for f in faces]).reshape(-1, 7)
        self.tree = cKDTree(pts) if len(faces) else None
        self.pts = pts

    def query_pairs_all(self):
        return set() if self.tree is None else self.tree.query_pairs(self.r)

    def neighbours(self, f):
        if self.tree is None:
            return []
        return self.tree.query_ball_point(self.pts[self.id[id(f)]], self.r)


def infer_joints(
    parts: list[Part],
    unit_scale: float,
    radial_tol: float = 0.6e-3,
    angle_tol_deg: float = 0.5,
    offset_tol: float = 0.1e-3,
    min_engagement: float = 0.5e-3,
    slide_ratio: float = 2.0,
) -> list[JointCandidate]:
    faces = [cf for p in parts for cf in cylindrical_faces(p, unit_scale)]
    cos_tol = np.cos(np.radians(angle_tol_deg))
    near = _coaxial_index(faces, angle_tol_deg, offset_tol, radial_tol)
    matches: dict[tuple, list] = {}
    for i, j in sorted(near.query_pairs_all()):
        a, b = faces[i], faces[j]
        if a.link == b.link or a.convex == b.convex:
            continue  # need one shaft and one bore on different links
        if not _coaxial(a, b, cos_tol, offset_tol):
            continue
        shaft, bore = (a, b) if a.convex else (b, a)
        if not (0 <= bore.radius - shaft.radius <= radial_tol):
            continue
        lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
        if hi - lo < min_engagement:
            continue
        key = (*sorted((a.link, b.link)), *np.round(a.direction, 3), *np.round(a.point, 4))
        matches.setdefault(key, []).append((shaft, bore, lo, hi))

    out = []
    for key, group in matches.items():
        shaft0 = group[0][0]
        d, p = shaft0.direction, shaft0.point
        lo = min(g[2] for g in group)
        hi = max(g[3] for g in group)
        engagement = _union_length([(g[2], g[3]) for g in group])
        # free shaft length = shaft span minus the part held in its own link's bores
        shafts = {id(g[0]): g[0] for g in group}.values()
        span = _union_length([(s.t0, s.t1) for s in shafts])
        held = []
        for s in shafts:
            for f in (faces[k] for k in near.neighbours(s)):
                if f.link == s.link and not f.convex and _coaxial(f, s, cos_tol, offset_tol) \
                        and 0 <= f.radius - s.radius <= radial_tol:
                    a, b = max(f.t0, s.t0), min(f.t1, s.t1)
                    if b > a:
                        held.append((a, b))
        free = span - _union_length(held)
        hint = "cylindrical" if free > slide_ratio * engagement else "revolute"
        out.append(
            JointCandidate(
                link_a=key[0],
                link_b=key[1],
                direction=d,
                origin=p + d * (lo + hi) / 2,
                radius=shaft0.radius,
                engagement=engagement,
                shaft_length=free,
                type_hint=hint,
                evidence=sorted({f"{g[0].part}->{g[1].part}" for g in group}),
                clearance=min(g[1].radius - g[0].radius for g in group),
            )
        )
    return out


ROOT_HINT = re.compile(r"chassis|base|frame|body|hull", re.I)
# a bearing needs the name AND annular geometry (a "bearing plate" is a plate)
BEARING_NAME = re.compile(r"bearing|(?<![a-z0-9])\d+x\d+x\d+(?![a-z0-9])|^mr\d+|^\d{4}(zz|rs|2rs)?$", re.I)
# touching gears mesh rather than join
GEAR_NAME = re.compile(r"(?<![a-z])\d+t(?![a-z])|gear|pinion|pulley|sprocket", re.I)
# servo parts include their horn, so they touch both the mount and the driven part
SERVO_NAME = re.compile(r"sts\d{4}|scs\d{2,4}|sm\d{2}bl|xl-?\d{3}|xm-?\d{3}|xh-?\d{3}|xc-?\d{3}|xw-?\d{3}|"
                        r"ax-?1[28]|mx-?\d{2}|dynamixel|feetech|servo|lx-?\d{3}|mg9\d{2}|ds3\d{3}", re.I)
# placeholder geometry, not physical parts
IGNORE_HINT = re.compile(r"no.?blockage|keep.?out|zone|envelope|reference|clearance.?vol|dummy|placeholder", re.I)

HORN_DOMINANCE = 0.6  # horn contacts weaker than this fraction of the strongest are grazes, not the output
MAX_INCIDENTAL = 40  # a loop is broken only if its weakest contact has fewer overlapping faces than this
LOOSE_PIVOT_MAX = 1.0e-3  # loosest hole a pin can still pivot in (printed linkages)


def _touching(parts: list[Part], scale: float, tol: float) -> list[tuple[str, str]]:
    """Pairs of parts whose surfaces come within ``tol``: bounding-box prefilter, then OpenCascade's
    mesh-based proximity test (exact B-rep distance is ~1000x slower). The mesh deflection is added to
    the tolerance so tessellation can't hide a contact."""
    deflection = 0.02e-3 / scale  # CAD units
    tol_cad = tol / scale + deflection
    # before meshing: build123d's bounding_box() discards triangulations
    bbs = [p.shape.bounding_box() for p in parts]
    lo = np.array([[b.min.X, b.min.Y, b.min.Z] for b in bbs]) - tol_cad
    hi = np.array([[b.max.X, b.max.Y, b.max.Z] for b in bbs]) + tol_cad
    for p in parts:
        BRepMesh_IncrementalMesh(p.shape.wrapped, deflection, False, 0.2, True)
    out = []
    for i in range(len(parts)):
        cand = np.where((lo[i + 1:] <= hi[i]).all(1) & (lo[i] <= hi[i + 1:]).all(1))[0] + i + 1
        for j in cand:
            pr = BRepExtrema_ShapeProximity(parts[i].shape.wrapped, parts[j].shape.wrapped, tol_cad)
            pr.Perform()
            if pr.IsDone() and pr.OverlapSubShapes1().Size() > 0:
                out.append((parts[i].name, parts[j].name))
    return out


def _cached_touching(step_path: Path, parts, scale, tol):
    """``_touching``, cached per STEP file content (it is the slow step)."""
    h = hashlib.sha1(Path(step_path).read_bytes()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "cad2urdf" / f"touch_{h}_{tol:.0e}.json"
    names = [p.name for p in parts]
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["names"] == names:
            return [tuple(x) for x in data["pairs"]]
    pairs = _touching(parts, scale, tol)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"names": names, "pairs": pairs}))
    return pairs


def _annulus(p: Part, scale: float):
    """(axis_dir, axis_point, r_in, r_out, center) if the part is a ring (bearing-like), else None."""

    fs = cylindrical_faces(p, scale)
    cos_tol = np.cos(np.radians(0.5))
    best = None
    for i in (f for f in fs if not f.convex):
        for o in (f for f in fs if f.convex and f.radius > i.radius):
            if not _coaxial(i, o, cos_tol, 0.1e-3):
                continue
            w = max(i.t1, o.t1) - min(i.t0, o.t0)
            ring = np.pi * (o.radius ** 2 - i.radius ** 2) * w
            if 0.5 * ring <= p.volume <= 1.3 * ring:
                mid_t = (min(i.t0, o.t0) + max(i.t1, o.t1)) / 2
                cand = (o.direction, o.point, i.radius, o.radius, o.point + o.direction * mid_t, w)
                if best is None or o.radius - i.radius > best[3] - best[2]:
                    best = cand
    return best


def _servo_horn(p: Part, scale: float):
    """Horn discs of a servo (the largest coaxial group of thin cylinders, r >= 4 mm, length <= r):
    (axis_dir, axis_point, [(t0, t1, r), ...], center, width), or None."""

    discs = [f for f in cylindrical_faces(p, scale)
             if f.convex and f.radius >= 4e-3 and (f.t1 - f.t0) <= f.radius]
    best = None
    for f in discs:
        group = [g for g in discs if _coaxial(f, g, np.cos(np.radians(0.5)), 0.1e-3)
                 and abs(g.radius - f.radius) < 0.5e-3]
        if best is None or (f.radius, len(group)) > (best[0].radius, len(best[1])):
            best = (f, group)
    if best is None:
        return None
    f, group = best
    t0, t1 = min(g.t0 for g in group), max(g.t1 for g in group)
    return f.direction, f.point, [(g.t0, g.t1, g.radius) for g in group], f.point + f.direction * (t0 + t1) / 2, t1 - t0


def _horn_faces(p: Part, horn, scale: float, tol: float = 0.3e-3):
    """Faces of the servo that belong to its horn discs (their centres lie inside a disc's cylinder)."""
    d, pt, discs = horn[0], horn[1], horn[2]
    out = []
    for f in p.shape.faces():
        c = np.array([f.center().X, f.center().Y, f.center().Z]) * scale
        t = float(np.dot(c, d))
        radial = np.linalg.norm(c - (pt + d * t))
        if any(t0 - tol <= t <= t1 + tol and radial <= r + tol for t0, t1, r in discs):
            out.append(f)
    return out


def _horn_contact(servo_faces, q: Part, tol_cad: float) -> int:
    """How many of the servo's horn faces touch part q."""
    defl = tol_cad / 3
    BRepMesh_IncrementalMesh(q.shape.wrapped, defl, False, 0.2, True)
    n = 0
    for f in servo_faces:
        BRepMesh_IncrementalMesh(f.wrapped, defl, False, 0.2, True)
        pr = BRepExtrema_ShapeProximity(f.wrapped, q.shape.wrapped, tol_cad)
        pr.Perform()
        n += bool(pr.IsDone() and pr.OverlapSubShapes1().Size() > 0)
    return n


def _loose_pivots(cands, press_fit_tol, max_running_clearance):
    """Pins snug in one part and >= 10x looser in another turn in the loose hole. Returns the (shaft, bore)
    pairs that pivot and the ones that hold the pin. Screws don't qualify (their thread has no clearance)."""
    by_shaft = defaultdict(list)
    for c in cands:
        for e in c.evidence:
            if "->" in e:
                a, b = e.split("->")
                by_shaft[a].append((c.clearance, b.split(" ")[0]))
    pivots, held = set(), set()
    for shaft, fits in by_shaft.items():
        if is_fastener(shaft):
            continue
        snug = [(cl, b) for cl, b in fits if press_fit_tol < cl <= max_running_clearance]
        loose = [(cl, b) for cl, b in fits if max_running_clearance < cl <= LOOSE_PIVOT_MAX]
        if snug and loose and min(cl for cl, _ in loose) >= 10 * max(cl for cl, _ in snug):
            pivots |= {(shaft, b) for _, b in loose}
            held |= {(shaft, b) for _, b in snug}
    return pivots, held


def _classify_fits(cands, rings, press_fit_tol, max_running_clearance, part_link=None, pivots=(), held=()):
    """Split fits into running fits (joints) and rigid ones: a bearing's outer race, or a fastener in a
    clearance hole that isn't a bearing bore. ``pivots``/``held`` override the clearance window for pins."""
    running, fixed = [], []
    for c in cands:
        pairs = [e.split("->") for e in c.evidence if "->" in e]
        ev = {(a, b.split(" ")[0]) for a, b in pairs}
        if ev & set(pivots):
            running.append(c)
            continue
        if ev & set(held):
            if part_link is None:
                fixed.append(c)
            continue
        if not (press_fit_tol < c.clearance <= max_running_clearance):
            continue
        shafts = {a for a, _ in pairs}
        bores = {b.split(" ")[0] for _, b in pairs}
        if shafts & set(rings) or any(is_fastener(a) for a in shafts) and not bores & set(rings):
            if part_link is None:  # part-level pass: link names are part names
                fixed.append(c)
            continue
        running.append(c)
    return running, fixed


def _contact_strength(a: Part, b: Part, tol_cad: float) -> float:
    """Number of overlapping faces: firm mounts share many, a folded arm resting on its base a few."""
    pr = BRepExtrema_ShapeProximity(a.shape.wrapped, b.shape.wrapped, tol_cad)
    pr.Perform()
    return float(pr.OverlapSubShapes1().Size() + pr.OverlapSubShapes2().Size()) if pr.IsDone() else 1.0


def _inner_side(q: Part, ring, scale: float) -> bool:
    """Part q sits inside the ring's mid radius (the shaft side)."""

    d, pt, r_in, r_out = ring[0], ring[1], ring[2], ring[3]
    probe = type("F", (), {"direction": d, "point": pt})
    co = [f for f in cylindrical_faces(q, scale) if _coaxial(f, probe, np.cos(np.radians(0.5)), 0.1e-3)]
    return bool(co) and max(f.radius for f in co) <= (r_in + r_out) / 2


def _on_axis(part: Part, c: JointCandidate, scale: float) -> bool:
    """The part hugs the joint axis (spacer, bushing): every vertex within 3 pin radii + 1 mm."""
    v = np.array([tuple(q) for q in part.shape.vertices()]) * scale - c.origin
    radial = np.linalg.norm(np.cross(v, c.direction), axis=1)
    return bool(radial.max() <= 3 * c.radius + 1e-3)


def _link_name(parts: list, taken: set[str]) -> str:
    """Named after its largest part that isn't a fastener or bearing."""
    real = [p for p in parts if not (is_fastener(p.name) or BEARING_NAME.search(p.name))]
    main = max(real or parts, key=lambda p: p.volume)
    base = re.sub(r"[^a-z0-9]+", "_", main.name.split("#")[0].lower()).strip("_")[:24] or "link"
    if base[0].isdigit():
        base = "link_" + base
    name, k = base, 2
    while name in taken:
        name, k = f"{base}_{k}", k + 1
    taken.add(name)
    return name


def draft_spec(step_path: Path, units: str = "mm", touch_tol: float = 0.05e-3,
               press_fit_tol: float = 0.005e-3, max_running_clearance: float = 0.15e-3) -> tuple[dict, list[str]]:
    scale = UNIT_TO_M[units]
    parts = load_parts(step_path)
    ignored = [p.name for p in parts if IGNORE_HINT.search(p.name)]
    parts = [p for p in parts if p.name not in set(ignored)]
    for p in parts:
        p.link = p.name  # every part its own "link" for candidate search
        p.density = 1000.0
        mass_properties(p, scale)
    by_name = {p.name: p for p in parts}
    rings = {p.name: r for p in parts if BEARING_NAME.search(p.name) and (r := _annulus(p, scale))}
    cands = infer_joints(parts, scale, radial_tol=LOOSE_PIVOT_MAX)
    pivots, held = _loose_pivots(cands, press_fit_tol, max_running_clearance)
    running, forced_fixed = _classify_fits(cands, rings, press_fit_tol, max_running_clearance,
                                           pivots=pivots, held=held)
    running_pairs = {tuple(sorted((c.link_a, c.link_b))) for c in running}
    gears = {p.name for p in parts if GEAR_NAME.search(p.name)}
    inner_of: dict[str, set[str]] = defaultdict(set)  # bearing/servo -> parts on its rotating side
    servos = {}
    for p in parts:
        if SERVO_NAME.search(p.name) and (h := _servo_horn(p, scale)):
            servos[p.name] = (h, _horn_faces(p, h, scale))
            d, pt, discs, center, w = h
            rings[p.name] = (d, pt, discs[0][2], discs[0][2], center, w)  # same layout as a bearing
    tol_cad = (touch_tol + 0.02e-3) / scale

    touching = _cached_touching(step_path, parts, scale, touch_tol)
    horn_graze: set[tuple[str, str]] = set()
    for srv, (_, faces) in servos.items():  # each neighbour is the output, a graze, or the mount
        nbrs = sorted({b if a == srv else a for a, b in touching if srv in (a, b)} - set(servos))
        strength = {q: _horn_contact(faces, by_name[q], tol_cad) for q in nbrs}
        top = max(strength.values(), default=0)
        for q, n in strength.items():
            if n and n >= HORN_DOMINANCE * top:
                inner_of[srv].add(q)
            elif n:
                horn_graze.add(tuple(sorted((srv, q))))
        if not [q for q in nbrs if not strength[q]]:  # no body contact: the strongest graze is the mount
            grazes = [q for q in nbrs if strength[q] and q not in inner_of[srv]]
            if grazes:
                horn_graze.discard(tuple(sorted((srv, max(grazes, key=lambda q: strength[q])))))

    def fixed(a: str, b: str) -> bool:
        if a in gears and b in gears:
            return False  # meshing teeth
        if tuple(sorted((a, b))) in horn_graze:
            return False  # incidental graze of a servo horn
        for srv, other in ((a, b), (b, a)):
            if srv in servos and other in inner_of[srv]:
                return False  # the servo's output: this contact is the joint
        for brg, other in ((a, b), (b, a)):
            if brg in rings and brg not in servos and other not in rings and _inner_side(by_name[other], rings[brg], scale):
                inner_of[brg].add(other)
                return False  # a bearing's shaft side: this contact is the joint
        return True

    review: list[str] = []
    import networkx as nx

    for p in parts:  # re-mesh for the strength measurements
        BRepMesh_IncrementalMesh(p.shape.wrapped, 0.02e-3 / scale, False, 0.2, True)
    G = nx.Graph()
    G.add_nodes_from(p.name for p in parts)
    for a, b in touching:
        if tuple(sorted((a, b))) not in running_pairs and fixed(a, b):
            # servo bodies and bearing outer races are mounted: never cut those
            cap = 1e9 if (a in servos or b in servos or a in rings or b in rings) \
                else _contact_strength(by_name[a], by_name[b], tol_cad)
            G.add_edge(a, b, capacity=cap)
    for c in cands:  # press fits: never cut
        if c.clearance <= press_fit_tol and fixed(c.link_a, c.link_b):
            G.add_edge(c.link_a, c.link_b, capacity=1e9)
    for c in forced_fixed:  # fasteners in clearance holes, bearing outer races in housings
        G.add_edge(c.link_a, c.link_b, capacity=1e9)

    # Both sides of every joint (servo/bearing output, running fit) must end up in different links:
    # contacts joining them in posed CAD are removed with a minimum cut.
    must_split = [(s_, o) for s_ in sorted(inner_of) for o in sorted(inner_of[s_])] + sorted(running_pairs)
    cut_edges, welded_shut = [], []
    changed = True
    while changed:
        changed = False
        for srv, o in must_split:
            if (srv, o) in welded_shut or not nx.has_path(G, srv, o):
                continue
            val, (side_a, side_b) = nx.minimum_cut(G, srv, o)
            if val >= 1e8:  # only press fits / mounts connect them: report instead
                welded_shut.append((srv, o))
                continue
            cut = sorted((u, v) for u in side_a for v in G[u] if v in side_b)
            G.remove_edges_from(cut)
            cut_edges += cut
            changed = True
    if welded_shut:
        review.append(f"{len(welded_shut)} joint(s) welded shut by press fits / servo mounts (check the "
                      f"link grouping): {welded_shut[:6]}")
    # Joints must form a tree. A loop is broken at its weakest incidental contact; strong loops (real
    # four-bars) are kept and reported.
    joint_pairs = [(s_, o) for s_, os_ in inner_of.items() for o in os_] + sorted(running_pairs)
    kept_loops: list[frozenset] = []
    for _ in range(50):
        comp = {n: i for i, cc in enumerate(nx.connected_components(G)) for n in cc}
        J = nx.MultiGraph()
        for a, b in joint_pairs:
            if comp[a] != comp[b]:
                J.add_edge(comp[a], comp[b], ends={comp[a]: a, comp[b]: b})
        cycle = next((c for c in nx.cycle_basis(nx.Graph(J))
                      if len(c) >= 3 and frozenset(frozenset(comp_parts) for comp_parts in
                                                   ([n for n, cc in comp.items() if cc == ci] for ci in c))
                      not in kept_loops), None)
        if cycle is None:
            break
        best = None
        for k, ci in enumerate(cycle):
            prev_c, next_c = cycle[k - 1], cycle[(k + 1) % len(cycle)]
            ea = next(iter(J.get_edge_data(ci, prev_c).values()))["ends"][ci]
            eb = next(iter(J.get_edge_data(ci, next_c).values()))["ends"][ci]
            if ea == eb:
                continue
            H = G.subgraph([n for n, c in comp.items() if c == ci])
            val, (sa, sb) = nx.minimum_cut(H, ea, eb)
            if best is None or val < best[0]:
                best = (val, sorted((u, v) for u in sa for v in H[u] if v in sb))
        if best is None or best[0] > MAX_INCIDENTAL:
            kept_loops.append(frozenset(frozenset(n for n, cc in comp.items() if cc == ci) for ci in cycle))
            if len(kept_loops) > 1 and kept_loops[-1] in kept_loops[:-1]:
                break
            review.append("closed kinematic loop kept (strong contacts; a real linkage or a coaxial support "
                          "such as an idler horn): " + " / ".join(
                              sorted(n for n, cc in comp.items() if cc == ci)[0] for ci in cycle))
            continue
        G.remove_edges_from(best[1])
        cut_edges += best[1]
    if cut_edges:
        review.append(f"cut {len(cut_edges)} incidental contact(s) (posed/folded CAD): "
                      + ", ".join(f"{u}|{v}" for u, v in cut_edges[:8]) + (" ..." if len(cut_edges) > 8 else ""))

    uf = UnionFind([p.name for p in parts])
    for a, b in G.edges:
        uf.union(a, b)

    if ignored:
        review.append(f"ignored {len(ignored)} placeholder part(s) (keep-out/zone/reference): {ignored[:8]}"
                      + (" ..." if len(ignored) > 8 else ""))

    # re-infer fits with the final grouping, so a pin's held length isn't counted as free shaft
    for p in parts:
        p.link = uf.find(p.name)
    running, _ = _classify_fits(infer_joints(parts, scale, radial_tol=LOOSE_PIVOT_MAX), rings, press_fit_tol,
                                max_running_clearance, pivots=pivots, held=held,
                                part_link={p.name: p.link for p in parts})

    welded = []
    for brg, inners in inner_of.items():
        d, pt, r_in, r_out, center, w = rings[brg]
        for q in sorted(inners):
            a, b = uf.find(q), uf.find(brg)
            if a == b:
                welded.append(brg)
                continue
            running.append(JointCandidate(link_a=a, link_b=b, direction=d, origin=center, radius=r_in,
                                          engagement=w, shaft_length=w, type_hint="revolute",
                                          evidence=[f"{q}->{brg} (bearing)"], clearance=1e-5))
    review.append(f"{len(rings) - len(servos)} bearing(s) and {len(servos)} servo(s) recognised, "
                  f"{len(gears)} gear-named part(s); "
                  f"{len(set(welded))} bearing(s) had both sides welded together by other contacts: "
                  f"{sorted(set(welded))[:6]}")
    # Merge links that add no motion: a small part on a coaxial pivot between two already-jointed links
    # (idler horn), or a tiny part spinning on a pin (spacer).
    def collinear(c1, c2) -> bool:
        if abs(float(np.dot(c1.direction, c2.direction))) < np.cos(np.radians(1.0)):
            return False
        return float(np.linalg.norm(np.cross(c2.origin - c1.origin, c1.direction))) < 1e-3

    merged_pivots = []
    changed = True
    while changed:
        changed = False
        by_comp: dict[str, list] = defaultdict(list)
        for c in running:
            a, b = uf.find(c.link_a), uf.find(c.link_b)
            if a != b:
                by_comp[a].append((b, c))
                by_comp[b].append((a, c))
        for comp_c, es in sorted(by_comp.items()):
            nbrs = sorted({o for o, _ in es})
            members = {p.name for p in parts if uf.find(p.name) == comp_c}
            if len(nbrs) == 1 and not any(n in inner_of for n in members) and sum(
                    by_name[n].volume for n in members) < 0.01 * sum(p.volume for p in parts) \
                    and all(_on_axis(by_name[n], es[0][1], scale) for n in members):
                uf.union(next(iter(members)), nbrs[0])
                merged_pivots.append(sorted(members))
                changed = True
                break
            if len(nbrs) != 2 or not all(collinear(es[0][1], c) for _, c in es):
                continue
            na, nb = nbrs
            if not any(o == nb and collinear(es[0][1], c) for o, c in by_comp.get(na, [])):
                continue
            if len(members) * 2 > len(parts):
                continue

            def firmness(n):
                side = {p.name for p in parts if uf.find(p.name) == n}
                return sum(_contact_strength(by_name[a], by_name[b], tol_cad) for a, b in touching
                           if (a in members and b in side) or (b in members and a in side))

            outputs = {uf.find(q) for qs in inner_of.values() for q in qs}
            target = max((na, nb), key=lambda n: (firmness(n), n in outputs, n))
            uf.union(next(iter(members)), target)
            merged_pivots.append(sorted(members))
            changed = True
            break
    if merged_pivots:
        review.append(f"merged {len(merged_pivots)} redundant coaxial pivot(s) (idler horns etc.) into their "
                      f"neighbour: {merged_pivots[:6]}")

    groups: dict[str, list[Part]] = defaultdict(list)
    for p in parts:
        groups[uf.find(p.name)].append(p)
    taken: set[str] = set()
    comp_name = {}
    for rep, ps in sorted(groups.items(), key=lambda kv: -sum(p.volume for p in kv[1])):
        comp_name[rep] = _link_name(ps, taken)

    def root_score(rep):
        ps = groups[rep]
        return (any(ROOT_HINT.search(p.name) for p in ps), sum(p.volume for p in ps))

    root = max(groups, key=root_score)
    edges = defaultdict(list)
    for c in running:
        a, b = uf.find(c.link_a), uf.find(c.link_b)
        if a != b:
            edges[a].append((b, c))
            edges[b].append((a, c))
    joints, seen, queue = {}, {root}, [root]
    while queue:
        cur = queue.pop(0)
        for nxt, c in sorted(edges[cur], key=lambda e: -e[1].engagement):
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
            parent, child = comp_name[cur], comp_name[nxt]
            jname = f"{parent}_to_{child}"
            # a pin with some free length is a pivot; only a long guide rod (free > 12 radii) slides
            slide = c.type_hint != "revolute" and c.shaft_length > 12 * c.radius
            if not slide:
                if c.type_hint != "revolute":
                    review.append(f"joints.{jname}: cylindrical fit (slide OR spin) on a short pin drafted as revolute")
                j = {"type": "revolute", "parent": parent, "child": child, "limits": [-3.1416, 3.1416],
                     "effort": 10.0, "velocity": 5.0}
                review.append(f"joints.{jname}.limits: placeholder +/-pi")
            else:
                half = round(float(c.shaft_length) / 2, 4)
                j = {"type": "prismatic", "parent": parent, "child": child, "limits": [-half, half],
                     "effort": 50.0, "velocity": 0.5}
                review.append(f"joints.{jname}: cylindrical fit (slide OR spin) drafted as prismatic; "
                              f"limits = +/- half the free shaft ({half} m)")
            scale = UNIT_TO_M[units]
            j["axis"] = [round(float(x), 6) for x in c.direction]  # explicit, so the compile never re-infers it
            j["origin"] = [round(float(x) / scale, 4) for x in c.origin]  # CAD units
            joints[jname] = j
    orphans = [comp_name[r] for r in groups if r not in seen]
    links = {comp_name[r]: sorted(p.name for p in ps) for r, ps in groups.items() if r in seen}
    if orphans:
        for r in groups:
            if r not in seen:
                links[comp_name[root]] += sorted(p.name for p in groups[r])
        review.append(f"{len(orphans)} group(s) with no running fit to the rest were merged into "
                      f"{comp_name[root]}: {orphans[:8]}" + (" ..." if len(orphans) > 8 else ""))

    spec = {
        "robot": re.sub(r"[^a-z0-9_]", "_", step_path.stem.lower()),
        "source": os.path.abspath(step_path),
        "units": units,
        "materials": {"default": 1200},
        "part_materials": {"*": "default"},
        **({"ignore_parts": sorted(ignored)} if ignored else {}),
        "links": links,
        "joints": joints,
        "dynamics": {"default": {"damping": 0.1, "friction": 0.01, "armature": 0.001}},
        "actuators": {"default": {"kind": "position", "kp": 100.0, "kv": 5.0}},
    }
    review.append("materials: one uniform density (1200 kg/m^3); set real materials / mass overrides")
    return spec, list(dict.fromkeys(review))


def write_draft(spec: dict, review: list[str], path: Path) -> None:
    header = "# DRAFT generated by cad2urdf.step (deterministic). Review before trusting:\n"
    header += "".join(f"#   REVIEW {r}\n" for r in review)
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False, width=120))


def name_group(name: str) -> str:
    """'0102_rotated_base_默认_7' -> '0102_rotated_base'; 'Screw#3' -> 'Screw' (solid / copy suffixes and
    SolidWorks' default-configuration tag removed)."""
    n = re.sub(r"([_#]\d+)+$", "", name)
    return re.sub(r"[_ ]?(默认|Default|default)$", "", n)


def shared_axes(parts, scale, angle_tol_deg=0.5, offset_tol=0.2e-3):
    faces = [f for p in parts for f in cylindrical_faces(p, scale)]
    idx = _coaxial_index(faces, angle_tol_deg, offset_tol, radial_tol=10.0)  # radius ignored
    cos_tol = np.cos(np.radians(angle_tol_deg))
    axes: dict[tuple, list] = defaultdict(list)
    for i, j in idx.query_pairs_all():
        a, b = faces[i], faces[j]
        if a.link == b.link or not _coaxial(a, b, cos_tol, offset_tol):
            continue
        if a.link > b.link:
            a, b = b, a
        key = (a.link, b.link, *np.round(a.direction, 2), *np.round(a.point * 1e3, 0))
        axes[key].append((a, b))
    out = []
    for key, pairs in axes.items():
        a0 = pairs[0][0]
        lo = min(min(a.t0, b.t0) for a, b in pairs)
        hi = max(max(a.t1, b.t1) for a, b in pairs)
        radii = sorted({round(f.radius * 1e3, 2) for ab in pairs for f in ab})
        out.append({"groups": key[:2], "direction": a0.direction, "origin": a0.point + a0.direction * (lo + hi) / 2,
                    "radii_mm": radii, "faces": len(pairs),
                    "clearance_mm": min(abs(a.radius - b.radius) for a, b in pairs) * 1e3})
    return sorted(out, key=lambda x: (x["groups"], -max(x["radii_mm"])))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cad2urdf.step", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", type=Path)
    ap.add_argument("--spec", type=Path, help="group solids by this spec's links instead of by part name")
    ap.add_argument("--units", default="mm")
    args = ap.parse_args(argv)
    scale = UNIT_TO_M[args.units]
    parts = load_parts(args.step)
    if args.spec:
        links = yaml.safe_load(args.spec.read_text())["links"]
        for p in parts:
            p.link = next((k for k, pats in links.items() if any(fnmatch.fnmatch(p.name, q) for q in pats)), "?")
    else:
        for p in parts:
            p.link = name_group(p.name)
    groups = defaultdict(list)
    for p in parts:
        mass_properties(p, scale)
        groups[p.link].append(p)
    print(f"{len(parts)} solids in {len(groups)} groups")
    for g, ps in sorted(groups.items()):
        lo = np.min([np.array(tuple(p.shape.bounding_box().min)) for p in ps], axis=0)
        hi = np.max([np.array(tuple(p.shape.bounding_box().max)) for p in ps], axis=0)
        print(f"  {g}: {len(ps)} solid(s), {sum(p.volume for p in ps) * 1e6:.1f} cm3, "
              f"bbox {np.round(lo).tolist()} .. {np.round(hi).tolist()}")


    touching = _cached_touching(args.step, parts, scale, 0.05e-3)
    link_of = {p.name: p.link for p in parts}
    contacts = defaultdict(int)
    for a, b in touching:
        ga, gb = sorted((link_of[a], link_of[b]))
        if ga != gb:
            contacts[(ga, gb)] += 1
    axes = defaultdict(list)
    for ax in shared_axes(parts, scale):
        axes[ax["groups"]].append(ax)

    print("\ngroup pairs (touching solids / shared axes):")
    for pair in sorted(set(contacts) | set(axes)):
        print(f"  {pair[0]} <-> {pair[1]}: {contacts.get(pair, 0)} touching solid pair(s)")
        for ax in axes.get(pair, []):
            d = np.round(ax["direction"], 4).tolist()
            o = np.round(ax["origin"] / scale, 2).tolist()
            print(f"      axis: {d}  origin: {o}  radii {ax['radii_mm']} mm, clearance {ax['clearance_mm']:.3f} mm, "
                  f"{ax['faces']} face pair(s)")


if __name__ == "__main__":
    from .util import sandbox

    sandbox("cad2urdf.step")
    main()
