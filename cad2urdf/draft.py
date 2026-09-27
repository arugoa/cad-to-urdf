"""Deterministic draft spec for a mate-less STEP assembly.

Rules (no LLM, same answer every run):

1. Every solid is a part. Two parts are *fixed together* if their B-reps touch
   (distance <= ``touch_tol``) and the contact is not a running fit.
2. A *running fit* is a coaxial shaft/bore pair with radial clearance in
   (``press_fit_tol``, ``max_running_clearance``] (see ``joints.infer_joints``).
   Zero clearance = press fit = fixed; a larger gap is a fastener clearance
   hole (bolt in plate) = fixed if the parts touch elsewhere (bolt head).
   This is the modelling convention the rule relies on: bearings/bushings drawn
   with a small clearance, press fits without, bolt holes with a normal
   clearance-hole gap.
3. Links = connected components of "fixed together" (union-find).
4. Root link = the component whose parts are named like ``base``/``chassis``/
   ``frame``, else the heaviest one (by volume).
5. Joints = running fits between components, tree-ified by BFS from the root,
   preferring the longest engagement. ``revolute`` hints stay revolute;
   ``cylindrical`` (long free shaft: slide or spin) is written as prismatic
   and flagged ``REVIEW``.
6. Limits are placeholders (revolute +/-pi, prismatic +/- half the free shaft)
   and flagged ``REVIEW``: geometry cannot tell where the hard stops are.

The result is written as ``robot_spec.draft.yaml`` next to the outputs; review
the REVIEW lines, rename links/joints if you like, and re-run with it.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from OCP.BRepExtrema import BRepExtrema_ShapeProximity
from OCP.BRepMesh import BRepMesh_IncrementalMesh

from . import cad
from .joints import JointCandidate, infer_joints

ROOT_HINT = re.compile(r"chassis|base|frame|body|hull", re.I)
# Bearings: a name that says so AND annular geometry. Both are required (a "bearing plate" is a plate).
BEARING_NAME = re.compile(r"bearing|(?<![a-z0-9])\d+x\d+x\d+(?![a-z0-9])|^mr\d+|^\d{4}(zz|rs|2rs)?$", re.I)
# Gears/pulleys: tooth-count names ("117t") or explicit words. Two touching gears are meshing, not fixed.
GEAR_NAME = re.compile(r"(?<![a-z])\d+t(?![a-z])|gear|pinion|pulley|sprocket", re.I)
# Hobby/robot servos: the part includes its output horn, so it touches both the mount and the driven part.
SERVO_NAME = re.compile(r"sts\d{4}|scs\d{2,4}|sm\d{2}bl|xl-?\d{3}|xm-?\d{3}|xh-?\d{3}|xc-?\d{3}|xw-?\d{3}|"
                        r"ax-?1[28]|mx-?\d{2}|dynamixel|feetech|servo|lx-?\d{3}|mg9\d{2}|ds3\d{3}", re.I)
# Placeholder geometry that is not part of the physical robot (keep-out volumes, reference bodies).
IGNORE_HINT = re.compile(r"no.?blockage|keep.?out|zone|envelope|reference|clearance.?vol|dummy|placeholder", re.I)


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _touching(parts: list[cad.Part], scale: float, tol: float) -> list[tuple[str, str]]:
    """Pairs of parts whose surfaces come within ``tol`` of each other.

    Bounding-box prefilter, then OpenCascade's mesh-based proximity test
    (BRepExtrema_ShapeProximity). The exact B-rep distance took ~1.5 s per pair
    on a real 1,142-part robot (hours in total); the mesh test takes < 1 ms.
    The mesh deflection is added to the tolerance so tessellation cannot hide contact.
    """
    deflection = 0.02e-3 / scale  # CAD units
    tol_cad = tol / scale + deflection
    # bounding boxes FIRST: build123d's bounding_box() discards the triangulation the proximity test needs
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
    """The contact search is the slow step (minutes on 1,000+ parts); cache it per file content."""
    import hashlib
    import json

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


def _annulus(p: cad.Part, scale: float):
    """(axis_dir, axis_point, r_in, r_out, center) if the part is a ring (bearing-like), else None."""
    from .joints import _coaxial, cylindrical_faces

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


def _servo_horn(p: cad.Part, scale: float):
    """Horn discs of a servo: the largest coaxial group of thin convex cylinders (r >= 4 mm, length <= r).

    Returns (axis_dir, axis_point, [(t0, t1, r), ...], center, width) or None.
    """
    from .joints import _coaxial, cylindrical_faces

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


def _horn_faces(p: cad.Part, horn, scale: float, tol: float = 0.3e-3):
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


def _horn_contact(servo_faces, q: cad.Part, tol_cad: float) -> int:
    """How many of the servo's horn faces touch part q."""
    # (re)mesh right before testing: bounding_box() calls elsewhere drop triangulations
    defl = tol_cad / 3
    BRepMesh_IncrementalMesh(q.shape.wrapped, defl, False, 0.2, True)
    n = 0
    for f in servo_faces:
        BRepMesh_IncrementalMesh(f.wrapped, defl, False, 0.2, True)
        pr = BRepExtrema_ShapeProximity(f.wrapped, q.shape.wrapped, tol_cad)
        pr.Perform()
        n += bool(pr.IsDone() and pr.OverlapSubShapes1().Size() > 0)
    return n


# a horn contact counts as the servo's output only if it is at least this fraction of the strongest one;
# weaker ones are grazes from a posed/folded assembly (neither joint nor rigid)
HORN_DOMINANCE = 0.6
# a contact cut to break a joint loop must be weaker than this (overlapping faces on both sides);
# stronger loops are kept as real closed linkages
MAX_INCIDENTAL = 40


def _contact_strength(a: cad.Part, b: cad.Part, tol_cad: float) -> float:
    """Contact strength ~ number of overlapping faces on both sides (firm mounts share many faces;
    a folded arm resting on its base touches along a few)."""
    pr = BRepExtrema_ShapeProximity(a.shape.wrapped, b.shape.wrapped, tol_cad)
    pr.Perform()
    return float(pr.OverlapSubShapes1().Size() + pr.OverlapSubShapes2().Size()) if pr.IsDone() else 1.0


def _inner_side(q: cad.Part, ring, scale: float) -> bool:
    """True if part q sits inside the ring's mid radius around its axis (shaft, spacer on the shaft)."""
    from .joints import _coaxial, cylindrical_faces

    d, pt, r_in, r_out = ring[0], ring[1], ring[2], ring[3]
    probe = type("F", (), {"direction": d, "point": pt})
    co = [f for f in cylindrical_faces(q, scale) if _coaxial(f, probe, np.cos(np.radians(0.5)), 0.1e-3)]
    return bool(co) and max(f.radius for f in co) <= (r_in + r_out) / 2


def _link_name(names: list[str], taken: set[str]) -> str:
    stems = Counter(re.split(r"[_\-\s]", n)[0].lower() for n in names)
    base = re.sub(r"[^a-z0-9_]", "_", stems.most_common(1)[0][0]) or "link"
    name, k = base, 2
    while name in taken:
        name, k = f"{base}_{k}", k + 1
    taken.add(name)
    return name


def draft_spec(step_path: Path, units: str = "mm", touch_tol: float = 0.05e-3,
               press_fit_tol: float = 0.005e-3, max_running_clearance: float = 0.15e-3) -> tuple[dict, list[str]]:
    scale = cad.UNIT_TO_M[units]
    parts = cad.load_parts(step_path)
    ignored = [p.name for p in parts if IGNORE_HINT.search(p.name)]
    parts = [p for p in parts if p.name not in set(ignored)]
    for p in parts:
        p.link = p.name  # every part its own "link" for candidate search
        p.density = 1000.0
        cad.mass_properties(p, scale)
    cands = infer_joints(parts, scale)
    running = [c for c in cands if press_fit_tol < c.clearance <= max_running_clearance]
    running_pairs = {tuple(sorted((c.link_a, c.link_b))) for c in running}

    by_name = {p.name: p for p in parts}
    rings = {p.name: r for p in parts if BEARING_NAME.search(p.name) and (r := _annulus(p, scale))}
    gears = {p.name for p in parts if GEAR_NAME.search(p.name)}
    inner_of: dict[str, set[str]] = defaultdict(set)  # bearing/servo -> parts on its rotating side
    servos = {}
    for p in parts:
        if SERVO_NAME.search(p.name) and (h := _servo_horn(p, scale)):
            servos[p.name] = (h, _horn_faces(p, h, scale))
            d, pt, discs, center, w = h
            rings[p.name] = (d, pt, discs[0][2], discs[0][2], center, w)  # same shape as a bearing entry
    tol_cad = (touch_tol + 0.02e-3) / scale

    touching = _cached_touching(step_path, parts, scale, touch_tol)
    horn_graze: set[tuple[str, str]] = set()
    for srv, (_, faces) in servos.items():  # classify each servo's neighbours: output / graze / mount
        nbrs = sorted({b if a == srv else a for a, b in touching if srv in (a, b)} - set(servos))
        strength = {q: _horn_contact(faces, by_name[q], tol_cad) for q in nbrs}
        top = max(strength.values(), default=0)
        for q, n in strength.items():
            if n and n >= HORN_DOMINANCE * top:
                inner_of[srv].add(q)
            elif n:
                horn_graze.add(tuple(sorted((srv, q))))
        if not [q for q in nbrs if not strength[q]]:  # no body contact at all: every servo is mounted somewhere
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
                return False  # contact on the servo's output horn: this contact IS the joint
        for brg, other in ((a, b), (b, a)):
            if brg in rings and brg not in servos and other not in rings and _inner_side(by_name[other], rings[brg], scale):
                inner_of[brg].add(other)
                return False  # shaft side of a bearing: this contact IS the joint
        return True

    review: list[str] = []
    # Contact graph: an edge = a fixed relation, weighted by contact strength (overlapping faces).
    import networkx as nx

    for p in parts:  # (re)mesh once for the strength measurements below
        BRepMesh_IncrementalMesh(p.shape.wrapped, 0.02e-3 / scale, False, 0.2, True)
    G = nx.Graph()
    G.add_nodes_from(p.name for p in parts)
    for a, b in touching:
        if tuple(sorted((a, b))) not in running_pairs and fixed(a, b):
            # a servo housing is always bolted to what its body touches: never cut those contacts
            cap = 1e9 if (a in servos or b in servos) else _contact_strength(by_name[a], by_name[b], tol_cad)
            G.add_edge(a, b, capacity=cap)
    for c in cands:  # press fits: never cut
        if c.clearance <= press_fit_tol and fixed(c.link_a, c.link_b):
            G.add_edge(c.link_a, c.link_b, capacity=1e9)

    # Constraint: a servo/bearing and the parts on its rotating side must end up in different links.
    # CAD is often saved in a folded/posed configuration where links rest against each other; those
    # incidental contacts are cut with a minimum cut (weakest total contact first). Deterministic.
    cut_edges = []
    changed = True
    while changed:
        changed = False
        for srv in sorted(inner_of):
            for o in sorted(inner_of[srv]):
                if nx.has_path(G, srv, o):
                    _, (side_a, side_b) = nx.minimum_cut(G, srv, o)
                    cut = sorted((u, v) for u in side_a for v in G[u] if v in side_b)
                    G.remove_edges_from(cut)
                    cut_edges += cut
                    changed = True
    # A mechanism's joints must form a tree over its links. If the rigid contacts plus the joints close
    # a loop, one of the rigid contacts on it is incidental (posed CAD): cut the weakest contact inside a
    # link on the loop. Loops whose cheapest cut is strong (a real four-bar) are kept and reported.
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
                break  # same loop again: nothing left to cut
            review.append("closed kinematic loop kept (strong contacts; a real linkage or a coaxial support "
                          "such as an idler horn): " + " / ".join(
                              sorted(n for n, cc in comp.items() if cc == ci)[0] for ci in cycle))
            continue
        G.remove_edges_from(best[1])
        cut_edges += best[1]
    if cut_edges:
        review.append(f"cut {len(cut_edges)} incidental contact(s) (posed/folded CAD): "
                      + ", ".join(f"{u}|{v}" for u, v in cut_edges[:8]) + (" ..." if len(cut_edges) > 8 else ""))

    uf = _UF([p.name for p in parts])
    for a, b in G.edges:
        uf.union(a, b)

    groups: dict[str, list[cad.Part]] = defaultdict(list)
    for p in parts:
        groups[uf.find(p.name)].append(p)
    taken: set[str] = set()
    comp_name = {}
    for rep, ps in sorted(groups.items(), key=lambda kv: -sum(p.volume for p in kv[1])):
        comp_name[rep] = _link_name([p.name for p in ps], taken)

    def root_score(rep):
        ps = groups[rep]
        return (any(ROOT_HINT.search(p.name) for p in ps), sum(p.volume for p in ps))

    root = max(groups, key=root_score)
    if ignored:
        review.append(f"ignored {len(ignored)} placeholder part(s) (keep-out/zone/reference): {ignored[:8]}"
                      + (" ..." if len(ignored) > 8 else ""))

    # component graph from running fits, re-inferred with the final grouping so a
    # pin's press-fit length counts as held, not free (revolute vs cylindrical)
    for p in parts:
        p.link = uf.find(p.name)
    running = [c for c in infer_joints(parts, scale) if press_fit_tol < c.clearance <= max_running_clearance]

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
            if c.type_hint == "revolute":
                j = {"type": "revolute", "parent": parent, "child": child, "limits": [-3.1416, 3.1416],
                     "effort": 10.0, "velocity": 5.0}
                review.append(f"joints.{jname}.limits: placeholder +/-pi")
            else:
                half = round(float(c.shaft_length) / 2, 4)
                j = {"type": "prismatic", "parent": parent, "child": child, "limits": [-half, half],
                     "effort": 50.0, "velocity": 0.5}
                review.append(f"joints.{jname}: cylindrical fit (slide OR spin) drafted as prismatic; "
                              f"limits = +/- half the free shaft ({half} m)")
            joints[jname] = j
    orphans = [comp_name[r] for r in groups if r not in seen]
    links = {comp_name[r]: sorted(p.name for p in ps) for r, ps in groups.items() if r in seen}
    if orphans:
        # attach to root by a fixed relation: merge their parts into the root link
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
    return spec, review


def write_draft(spec: dict, review: list[str], path: Path) -> None:
    header = "# DRAFT generated by cad2urdf.draft (deterministic). Review before trusting:\n"
    header += "".join(f"#   REVIEW {r}\n" for r in review)
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False, width=120))
