"""Lighter visuals: drop fasteners and decimate each part within a surface-error bound.

    simplify: {drop_fasteners: true, visual_faces_per_link: 20000}   # `simplify: false` turns it off

Fastener mass stays in the link. The face budget is shared by surface area and is a target: a part is
never decimated past 0.5% of its size.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .model import Robot
from .urdf import budget_mesh
from .util import is_fastener


def _deviation(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 1000) -> float:
    """Symmetric surface deviation (99th percentile of sampled point-to-surface distances), metres."""
    d1 = trimesh.proximity.closest_point(b, trimesh.sample.sample_surface(a, n, seed=0)[0])[1]
    d2 = trimesh.proximity.closest_point(a, trimesh.sample.sample_surface(b, n, seed=1)[0])[1]
    return float(np.percentile(np.r_[d1, d2], 99))


def safe_decimate(mesh: trimesh.Trimesh, faces: int, rel_tol: float = 0.005, abs_tol: float = 0.2e-3):
    """Decimate towards ``faces`` without moving the surface more than max(rel_tol x size, abs_tol).
    Open or multi-shell parts tear under aggressive decimation, so gentler settings and then larger
    targets are tried; the part is kept as is if nothing fits."""
    tol = max(rel_tol * float(mesh.extents.max()), abs_tol)
    target = faces
    while target < 0.8 * len(mesh.faces):
        for aggression in (7, 2, 0):
            try:
                out = mesh.simplify_quadric_decimation(face_count=target, aggression=aggression)
            except Exception:  # noqa: BLE001  (no decimator installed)
                return budget_mesh(mesh, target)
            if 0 < len(out.faces) < 0.9 * len(mesh.faces) and _deviation(mesh, out) <= tol:
                return out
        target *= 2
    return mesh


def _part_key(link, part) -> str:
    """Which visual group (material) a part belongs to."""
    if part.material in link.visuals:  # STEP / Onshape: grouped by material
        return part.material
    tail = part.name.split("/", 1)[-1]
    if tail in link.visuals:
        return tail
    return next(iter(link.visuals), "mesh")


def simplify_robot(robot: Robot, drop_fasteners: bool = True, visual_faces_per_link: int = 20000) -> dict:
    """Returns {link: {"dropped", "faces_before", "faces_after"}}."""
    report = {}
    for link in robot.links.values():
        before = int(sum(len(m.faces) for m in link.visuals.values()))
        parts = list(link.parts)
        dropped = [p for p in parts if drop_fasteners and is_fastener(p.name)]
        keep = [p for p in parts if p not in dropped] or parts  # never empty a link completely
        if not keep or all(p.mesh is None for p in keep):
            report[link.name] = {"dropped": 0, "faces_before": before, "faces_after": before}
            continue
        mats = {k: link.material(k) for k in link.visuals}
        area = sum(p.mesh.area for p in keep if p.mesh is not None) or 1.0
        groups: dict[str, list] = {}
        for p in keep:
            if p.mesh is None:
                continue
            budget = max(40, int(visual_faces_per_link * p.mesh.area / area))
            groups.setdefault(_part_key(link, p), []).append(safe_decimate(link.to_link(p.mesh), budget))
        link.visuals = {k: trimesh.util.concatenate(ms) for k, ms in groups.items()}
        link.visual_materials = {k: mats.get(k, k) for k in link.visuals}
        link.parts = keep  # collision generation now ignores the fasteners too
        report[link.name] = {"dropped": len(dropped), "faces_before": before,
                             "faces_after": int(sum(len(m.faces) for m in link.visuals.values()))}
    return report


def fit_spheres(mesh: trimesh.Trimesh, max_spheres: int = 24, resolution: int = 24, overshoot: float = 0.4,
                target: float = 0.97, samples: int = 1500):
    """Spheres covering ``target`` of a part's surface: candidates on interior voxels (radius = depth +
    ``overshoot`` x half-thickness, so corners are reachable), picked greedily. [(center, radius), ...]"""
    if not mesh.is_watertight or mesh.volume <= 0:
        mesh = mesh.convex_hull
    pitch = float(max(mesh.extents)) / resolution
    try:
        pts = mesh.voxelized(pitch).fill().points
        pts = pts[mesh.contains(pts)] if len(pts) > 64 else pts
    except Exception:  # noqa: BLE001
        pts = np.zeros((0, 3))
    if len(pts) == 0:
        c = mesh.bounding_sphere.primitive
        return [(np.asarray(c.center), float(c.radius))]
    if len(pts) > 3000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    depth = np.maximum(trimesh.proximity.signed_distance(mesh, pts), pitch * 0.25)  # > 0 inside
    radius = depth + overshoot * depth.max()
    surf = trimesh.sample.sample_surface(mesh, samples, seed=0)[0]
    covers = np.linalg.norm(surf[None] - pts[:, None], axis=2) <= radius[:, None]  # candidates x samples
    uncovered = np.ones(len(surf), bool)
    spheres = []
    while uncovered.mean() > 1 - target and len(spheres) < max_spheres:
        gain = (covers & uncovered).sum(1)
        i = int(np.argmax(gain))
        if gain[i] == 0:
            break
        spheres.append((pts[i], float(radius[i])))
        uncovered &= ~covers[i]
    return spheres
