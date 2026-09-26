"""Geometric joint inference from a mate-less B-rep assembly.

Idea: a joint in a real mechanism almost always shows up as a *shaft inside a
bore* — two cylindrical faces on different rigid bodies that are coaxial, have
(nearly) the same radius and overlap along the axis. That signature survives
STEP export even though the CAD mates do not.

For each pair of links we collect such shaft/bore matches, merge the collinear
ones (a pin through two clevis plates gives two matches on one axis), and emit a
candidate with an axis, an origin on that axis and a type hint:

* ``revolute``     the shaft's free length (not buried in its own link's
                   press-fit bores) is about the engagement length -> it spins
* ``cylindrical``  the free shaft is much longer than the bore -> it could slide
                   *or* spin (e.g. a linear rail); the spec must disambiguate

Limits: planar/prismatic joints on flat ways, ball joints, belt/gear
transmissions and flexures have no shaft/bore signature and need the spec (or
native CAD mates). Parts in the same link are ignored, which is why link
grouping happens first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from build123d import GeomType

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
        axis = f.axis_of_rotation
        p0 = _v(axis.position) * unit_scale
        d = _canonical(_v(axis.direction))
        # project the axis point to the foot of the perpendicular from the world origin
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
        faces.append(CylFace(part.name, part.link, f.radius * unit_scale, p0, d, t.min(), t.max(), convex))
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
    matches: dict[tuple, list] = {}
    for i, a in enumerate(faces):
        for b in faces[i + 1 :]:
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
            for f in faces:
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
            )
        )
    return out
