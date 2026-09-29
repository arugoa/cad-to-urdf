"""Joint candidates from geometry: a shaft inside a bore on two different links.

Coaxial cylindrical faces of (nearly) equal radius that overlap along the axis survive STEP export even
though mates don't. Collinear matches merge into one candidate with an axis, origin and hint:
``revolute`` (free shaft about the engagement length) or ``cylindrical`` (long free shaft: slide or spin).
Flat slides, ball joints and belt/gear drives have no such signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from build123d import GeomType
from OCP.BRepAdaptor import BRepAdaptor_Surface

from .cad import Part


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
