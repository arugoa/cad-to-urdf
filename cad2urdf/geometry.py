"""Mesh and collision geometry.

* visuals: ``budget_mesh`` / ``split_heavy_visuals`` keep every STL under 100k triangles (MuJoCo rejects
  200k); ``simplify_robot`` drops fasteners (their mass stays) and decimates each part within a surface
  error bound. Spec: ``simplify: {drop_fasteners: true, visual_faces_per_link: 20000}`` or ``false``.
* collision, per link (``collision: {default: {mode: ...}}``): none, box (one OBB), spheres (a few per
  part), primitives (box/cylinder/sphere per part, else hull), hull (one per link), auto (default:
  primitive, small-part hull, else CoACD per part), decompose (CoACD on the whole link), mesh (raw visual
  mesh), keep (the input URDF's own). Every piece is convex, at most ``max_hull_vertices`` (64: PhysX GPU),
  and each link gets at most ``max_geoms`` pieces (8; one for links under ``small_link_fraction`` of the
  robot's volume).
* ``limit_sweep``: moves each child to its limits and measures the solid volume it shares with its parent;
  a jump means the limit drives the part through material (simulators can't catch this).
"""

from __future__ import annotations

import numpy as np
import trimesh

from .model import CollisionGeom, Link, Robot, _per
from .util import is_fastener


MAX_VISUAL_FACES = 100_000  # per STL file; MuJoCo rejects files over 200k triangles


def budget_mesh(mesh, max_faces: int = MAX_VISUAL_FACES):
    """Quadric-decimate a mesh to at most ``max_faces`` triangles (unchanged if already within budget)."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception:  # noqa: BLE001  (no decimator installed: keep the MuJoCo limit at least)
        out = mesh.submesh([mesh.area_faces.argsort()[::-1][:max_faces]], append=True)
    return out if len(out.faces) else mesh


def split_heavy_visuals(robot: Robot, max_faces: int = MAX_VISUAL_FACES) -> None:
    """Split visual meshes over ``max_faces`` into several files by connected parts (decimation stalls on
    links fused from many parts); a single part over budget is decimated."""
    import trimesh

    for link in robot.links.values():
        new, mats = {}, {}
        for key, mesh in link.visuals.items():
            if len(mesh.faces) <= max_faces:
                new[key], mats[key] = mesh, link.material(key)
                continue
            chunks, cur, n = [], [], 0
            for comp in sorted(mesh.split(only_watertight=False), key=lambda c: -len(c.faces)):
                comp = budget_mesh(comp, max_faces)
                if n + len(comp.faces) > max_faces and cur:
                    chunks.append(cur)
                    cur, n = [], 0
                cur.append(comp)
                n += len(comp.faces)
            if cur:
                chunks.append(cur)
            for i, ch in enumerate(chunks):
                new[f"{key}_{i}"] = trimesh.util.concatenate(ch)
                mats[f"{key}_{i}"] = link.material(key)
        link.visuals, link.visual_materials = new, mats


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


def _manifold(mesh: trimesh.Trimesh):
    import manifold3d as mf

    if mesh is None or not mesh.is_watertight:
        return None
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    return None if m.is_empty() else m


def _union(parts):
    import manifold3d as mf

    solids = [s for s in (_manifold(p.mesh) for p in parts) if s is not None]
    return mf.Manifold.batch_boolean(solids, mf.OpType.Add) if solids else None, len(solids)


def _motion(joint, q: float) -> np.ndarray:
    """World transform applied to the child's zero-pose geometry when the joint is at q."""
    T = np.eye(4)
    a = joint.axis / np.linalg.norm(joint.axis)
    if joint.type == "prismatic":
        T[:3, 3] = a * q
        return T
    R = trimesh.transformations.rotation_matrix(q, a, joint.origin)
    return R


def limit_sweep(robot: Robot, tol_cm3: float = 1.0) -> dict:
    out, flagged = {}, []
    solids = {name: _union(link.parts) for name, link in robot.links.items()}
    for j in robot.joints.values():
        if j.type not in ("revolute", "prismatic") or j.mimic:
            continue
        (child, _), (parent, _) = solids[j.child], solids[j.parent]
        if child is None or parent is None:
            out[j.name] = {"skipped": "no closed part geometry"}
            continue
        vols = {}
        for tag, q in (("zero", 0.0), ("lower", j.lower), ("upper", j.upper)):
            moved = child.transform(_motion(j, q)[:3, :4])
            vols[tag] = float((moved ^ parent).volume()) * 1e6  # cm^3
        out[j.name] = {f"overlap_cm3_at_{k}": round(v, 2) for k, v in vols.items()}
        if max(vols["lower"], vols["upper"]) > vols["zero"] + tol_cm3:
            flagged.append(j.name)
    return {"joints": out, "limits_driving_into_parent": flagged}
