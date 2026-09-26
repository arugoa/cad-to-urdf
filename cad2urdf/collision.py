"""Collision geometry for multi-part links, at a per-link granularity.

Modes (cheapest -> most faithful), selected per link in the spec:

none        nothing
box         one oriented bounding box around the whole link
auto        per part: primitive if it fills >= 80 %, hull if small or nearly
            convex, else CoACD on that part (the router's default)
primitives  one box/cylinder/sphere per *part* (tightest bounding primitive),
            tiny parts culled; falls back to that part's convex hull when no
            primitive fills it well
hull        one convex hull of the whole link, vertex-capped
decompose   CoACD approximate convex decomposition of the whole link
mesh        the raw visual mesh (most engines silently convexify it -> avoid)
keep        the collision elements already present in an input URDF

Every convex piece is capped at ``max_hull_vertices`` (PhysX GPU cooking
limit is 64; MuJoCo and Bullet accept more but gain nothing from it).
"""

from __future__ import annotations

import numpy as np
import trimesh

from .model import CollisionGeom, Link, Robot, _per


def _cap_vertices(hull: trimesh.Trimesh, max_vertices: int) -> trimesh.Trimesh:
    """Reduce a convex hull to <= max_vertices by farthest-point sampling its vertices.

    The result is inscribed in the original hull; we then scale it about its
    centroid so it recovers the original volume (a cheap, slightly conservative fix).
    """
    v = hull.vertices
    if len(v) <= max_vertices:
        return hull
    chosen = [int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))]
    d = np.linalg.norm(v - v[chosen[0]], axis=1)
    while len(chosen) < max_vertices:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(v - v[i], axis=1))
    reduced = trimesh.convex.convex_hull(v[chosen])
    if reduced.volume > 0:
        s = (hull.volume / reduced.volume) ** (1 / 3)
        c = reduced.centroid
        reduced.vertices = (reduced.vertices - c) * s + c
    return reduced


def _hull_geom(mesh: trimesh.Trimesh, max_v: int, source: str) -> CollisionGeom:
    hull = _cap_vertices(mesh.convex_hull, max_v)
    return CollisionGeom("mesh", np.eye(4), mesh=hull, source=source)


def proper_frame(T: np.ndarray) -> np.ndarray:
    """trimesh can return a mirrored (det -1) OBB frame; flip one axis so it is a rotation."""
    T = np.array(T, dtype=float)
    if np.linalg.det(T[:3, :3]) < 0:
        T[:3, 2] *= -1
    return T


def _best_primitive(mesh: trimesh.Trimesh) -> tuple[CollisionGeom, float]:
    """Tightest of OBB / bounding cylinder / bounding sphere, and its fill ratio."""
    options = []
    obb = mesh.bounding_box_oriented
    options.append((obb.volume, CollisionGeom("box", proper_frame(obb.primitive.transform), tuple(obb.primitive.extents))))
    try:
        cyl = mesh.bounding_cylinder
        p = cyl.primitive
        options.append((cyl.volume, CollisionGeom("cylinder", proper_frame(p.transform), (p.radius, p.height))))
    except Exception:  # degenerate meshes
        pass
    sph = mesh.bounding_sphere
    options.append((sph.volume, CollisionGeom("sphere", np.array(sph.primitive.transform), (sph.primitive.radius,))))
    vol, geom = min(options, key=lambda o: o[0])
    if geom.kind == "box":
        geom = _axis_align_box(geom)
    return geom, float(mesh.volume / vol)


def _axis_align_box(g: CollisionGeom) -> CollisionGeom:
    """If an OBB is just an axis permutation, write it axis-aligned (rpy 0) with permuted extents."""
    R = g.transform[:3, :3]
    if np.allclose(np.abs(R).round(), np.abs(R), atol=1e-6):
        T = np.eye(4)
        T[:3, 3] = g.transform[:3, 3]
        return CollisionGeom("box", T, tuple(np.abs(R) @ np.array(g.size)), source=g.source)
    return g


def link_collisions(link: Link, cfg: dict, max_v: int) -> list[CollisionGeom]:
    mode = cfg.get("mode", "hull")
    link.collision_mode = mode
    whole = link.mesh()
    if mode == "none" or (mode != "keep" and not link.visuals):
        return []
    if mode == "keep":  # the input URDF's own collision elements, unchanged
        return list(link.source_collisions)
    if mode == "mesh":
        return [CollisionGeom("mesh", np.eye(4), mesh=whole, source="raw")]
    if mode == "box":
        obb = whole.bounding_box_oriented
        g = CollisionGeom("box", proper_frame(obb.primitive.transform), tuple(obb.primitive.extents), source="link-obb")
        return [_axis_align_box(g)]
    if mode == "hull":
        return [_hull_geom(whole, max_v, "link-hull")]
    if mode == "primitives":
        total = sum(p.volume for p in link.parts)
        min_frac = cfg.get("min_part_fraction", 0.02)
        min_fill = cfg.get("min_fill", 0.55)
        out = []
        for p in link.parts:
            if p.volume < min_frac * total:
                continue  # culled: bolts, pins, small brackets
            m = link.to_link(p.mesh)
            g, fill = _best_primitive(m)
            if fill >= min_fill:
                g.source = f"{p.name} ({g.kind}, fill {fill:.2f})"
                out.append(g)
            else:
                out.append(_hull_geom(m, max_v, f"{p.name} (hull, primitive fill {fill:.2f})"))
        return out
    if mode == "auto":
        from_urdf = all(p.shape is None for p in link.parts)
        if from_urdf and link.mass <= 0:
            return []  # massless link from an input URDF: a frame or decoration (stickers, lightbars)
        return _auto(link, cfg, max_v)
    if mode == "decompose":
        return _coacd(whole, cfg, max_v, "coacd")
    raise ValueError(f"unknown collision mode {mode!r}")


def _auto(link: Link, cfg: dict, max_v: int) -> list[CollisionGeom]:
    """Per part: primitive if it fills well, else a hull if the part is small, else CoACD on that part.

    Deterministic (CoACD seed 0). Works for STEP parts and for an exporter URDF's per-visual parts.
    """
    total = sum(p.volume for p in link.parts) or 1.0
    min_frac = cfg.get("min_part_fraction", 0.02)
    min_fill = cfg.get("min_fill", 0.8)
    out = []
    for p in link.parts:
        if p.volume < min_frac * total:
            continue
        m = link.to_link(p.mesh)
        g, fill = _best_primitive(m)
        if fill >= min_fill:
            g.source = f"{p.name} ({g.kind}, fill {fill:.2f})"
            out.append(g)
            continue
        hull = m.convex_hull
        if p.volume < 0.1 * total or (m.is_watertight and m.volume / max(hull.volume, 1e-12) > 0.85):
            out.append(_hull_geom(m, max_v, f"{p.name} (hull)"))
            continue
        out += _coacd(m, {"threshold": cfg.get("threshold", 0.05), "max_hulls": cfg.get("max_hulls", 8)},
                      max_v, f"{p.name} coacd")
    return out


def _coacd(mesh: trimesh.Trimesh, cfg: dict, max_v: int, tag: str) -> list[CollisionGeom]:
    # Imported lazily: loading coacd's native library before OpenCascade (build123d) makes
    # import_step segfault in the same process.
    import coacd

    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=cfg.get("threshold", 0.05),
        max_convex_hull=cfg.get("max_hulls", -1),
        max_ch_vertex=max_v,
        decimate=True,
        preprocess_mode="auto",
        seed=0,
    )
    return [CollisionGeom("mesh", np.eye(4), mesh=_cap_vertices(trimesh.Trimesh(v, f).convex_hull, max_v),
                          source=f"{tag}-{i}") for i, (v, f) in enumerate(parts)]


def geom_to_mesh(g: CollisionGeom) -> trimesh.Trimesh:
    if g.kind == "box":
        return trimesh.creation.box(extents=g.size, transform=g.transform)
    if g.kind == "cylinder":
        return trimesh.creation.cylinder(radius=g.size[0], height=g.size[1], transform=g.transform, sections=32)
    if g.kind == "sphere":
        s = trimesh.creation.icosphere(subdivisions=3, radius=g.size[0])
        s.apply_transform(g.transform)
        return s
    return g.mesh


def metrics(link: Link, n: int = 60000, seed: int = 0) -> dict:
    """Volumetric agreement between the exact parts and the collision set (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    part_meshes = [link.to_link(p.mesh) for p in link.parts]
    coll = [geom_to_mesh(g) for g in link.collisions]
    all_bounds = np.array([m.bounds for m in part_meshes + coll])
    lo, hi = all_bounds[:, 0].min(0), all_bounds[:, 1].max(0)
    pts = rng.uniform(lo, hi, size=(n, 3))
    in_cad = np.zeros(n, bool)
    for m in part_meshes:
        in_cad |= m.contains(pts)
    in_col = np.zeros(n, bool)
    for m in coll:
        in_col |= m.contains(pts)
    inter, union = (in_cad & in_col).sum(), (in_cad | in_col).sum()
    verts = [len(m.vertices) for m in coll if m is not None]
    return {
        "geoms": len(link.collisions),
        "kinds": sorted({g.kind for g in link.collisions}),
        "iou": float(inter / union) if union else 0.0,
        "coverage": float(inter / in_cad.sum()) if in_cad.sum() else 0.0,  # CAD volume inside collision
        "excess": float((in_col & ~in_cad).sum() / max(in_col.sum(), 1)),  # collision volume that is air
        "max_hull_vertices": max([len(g.mesh.vertices) for g in link.collisions if g.kind == "mesh"], default=0),
        "total_vertices": int(sum(verts)),
    }


def build_collisions(robot: Robot, with_metrics: bool = True) -> None:
    spec = robot.spec.get("collision", {})
    max_v = spec.get("max_hull_vertices", 64)
    for link in robot.links.values():
        link.collisions = link_collisions(link, _per(spec, link.name), max_v)
        if with_metrics and link.collisions and link.parts:
            link.collision_metrics = metrics(link)
