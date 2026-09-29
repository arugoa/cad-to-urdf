"""Collision geometry per link. Modes, set per link in the spec (``collision: {default: {mode: ...}}``):

none, box (one OBB), spheres (a few per part), primitives (box/cylinder/sphere per part, else hull),
hull (one per link), auto (default: primitive, small-part hull, else CoACD, per part),
decompose (CoACD on the whole link), mesh (raw visual mesh), keep (the input URDF's own collisions).

Every piece is convex and capped at ``max_hull_vertices`` (64: PhysX GPU limit). Each link then gets a
budget: ``max_geoms`` pieces (default 8), one for links under ``small_link_fraction`` of the robot's volume.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .model import CollisionGeom, Link, Robot, _per


def _cap_vertices(hull: trimesh.Trimesh, max_vertices: int) -> trimesh.Trimesh:
    """Farthest-point-sample a hull down to ``max_vertices``, rescaled to keep the original volume."""
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
            return []  # massless link from an input URDF: a frame or decoration
        return _auto(link, cfg, max_v)
    if mode == "decompose":
        return _coacd(whole, cfg, max_v, "coacd")
    if mode == "spheres":
        return _spheres(link, cfg)
    raise ValueError(f"unknown collision mode {mode!r}")


def _auto(link: Link, cfg: dict, max_v: int) -> list[CollisionGeom]:
    """Per part: a primitive if it fills >= min_fill, a hull if small or nearly convex, else CoACD."""
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


def _spheres(link: Link, cfg: dict) -> list[CollisionGeom]:
    """A few spheres per significant part, ``max_spheres`` per link shared by surface area."""
    from .simplify import fit_spheres

    total = sum(p.volume for p in link.parts) or 1.0
    min_frac = cfg.get("min_part_fraction", 0.02)
    budget = cfg.get("max_spheres", 32)
    kept = [p for p in link.parts if p.mesh is not None and p.volume >= min_frac * total]
    area = sum(p.mesh.area for p in kept) or 1.0
    out = []
    for p in kept:
        n = max(1, round(budget * p.mesh.area / area))
        for i, (c, r) in enumerate(fit_spheres(link.to_link(p.mesh), n, cfg.get("resolution", 24))):
            T = np.eye(4)
            T[:3, 3] = c
            out.append(CollisionGeom("sphere", T, (r,), source=f"{p.name} sphere {i}"))
    return out


def _coacd(mesh: trimesh.Trimesh, cfg: dict, max_v: int, tag: str) -> list[CollisionGeom]:
    # lazy: importing coacd before OpenCascade makes import_step segfault
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


def _merge_to_budget(geoms: list[CollisionGeom], budget: int, max_v: int) -> list[CollisionGeom]:
    """Merge pieces pairwise (least added hull volume, among 6 nearest neighbours) down to ``budget``."""
    if len(geoms) <= budget:
        return geoms
    pieces = [(geom_to_mesh(g).convex_hull, g) for g in geoms]
    while len(pieces) > budget:
        cents = np.array([m.centroid for m, _ in pieces])
        best = None
        for i in range(len(pieces)):
            order = np.argsort(np.linalg.norm(cents - cents[i], axis=1))[1:7]
            for j in order:
                if j < i and i in np.argsort(np.linalg.norm(cents - cents[j], axis=1))[1:7]:
                    continue  # pair already scored from j's side
                hull = trimesh.Trimesh(np.vstack([pieces[i][0].vertices, pieces[j][0].vertices])).convex_hull
                waste = hull.volume - pieces[i][0].volume - pieces[j][0].volume
                if best is None or waste < best[0]:
                    best = (waste, i, int(j), hull)
        _, i, j, hull = best
        for k in sorted((i, j), reverse=True):
            pieces.pop(k)
        pieces.append((hull, None))
    return [g if g is not None else _hull_geom(m, max_v, "merged hull") for m, g in pieces]


def build_collisions(robot: Robot, with_metrics: bool = True) -> None:
    """Collision geometry for every link, then the per-link budget (see the module docstring)."""
    spec = robot.spec.get("collision", {})
    max_v = spec.get("max_hull_vertices", 64)
    total = sum(p.volume for link in robot.links.values() for p in link.parts) or 1.0
    for link in robot.links.values():
        cfg = _per(spec, link.name)
        link.collisions = link_collisions(link, cfg, max_v)
        if link.collision_mode in ("auto", "decompose", "primitives") and link.collisions:
            small = sum(p.volume for p in link.parts) < cfg.get("small_link_fraction", 0.005) * total
            budget = 1 if small else cfg.get("max_geoms", 8)
            if len(link.collisions) > budget and budget == 1:
                g, fill = _best_primitive(link.mesh())
                link.collisions = [g] if fill >= cfg.get("min_fill", 0.8) else [_hull_geom(link.mesh(), max_v, "small link hull")]
            else:
                link.collisions = _merge_to_budget(link.collisions, budget, max_v)
        if with_metrics and link.collisions and link.parts:
            link.collision_metrics = metrics(link)
