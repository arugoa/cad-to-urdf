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
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from . import cad
from .joints import infer_joints

ROOT_HINT = re.compile(r"base|chassis|frame|body|hull", re.I)


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
    bbs = {}
    for p in parts:
        bb = p.shape.bounding_box()
        bbs[p.name] = (np.array([bb.min.X, bb.min.Y, bb.min.Z]) * scale - tol,
                       np.array([bb.max.X, bb.max.Y, bb.max.Z]) * scale + tol)
    out = []
    for i, a in enumerate(parts):
        lo_a, hi_a = bbs[a.name]
        for b in parts[i + 1:]:
            lo_b, hi_b = bbs[b.name]
            if np.any(hi_a < lo_b) or np.any(hi_b < lo_a):
                continue
            d = BRepExtrema_DistShapeShape(a.shape.wrapped, b.shape.wrapped)
            if d.IsDone() and d.Value() * scale <= tol:
                out.append((a.name, b.name))
    return out


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
    for p in parts:
        p.link = p.name  # every part its own "link" for candidate search
        p.density = 1000.0
        cad.mass_properties(p, scale)
    cands = infer_joints(parts, scale)
    running = [c for c in cands if press_fit_tol < c.clearance <= max_running_clearance]
    running_pairs = {tuple(sorted((c.link_a, c.link_b))) for c in running}

    uf = _UF([p.name for p in parts])
    for a, b in _touching(parts, scale, touch_tol):
        if tuple(sorted((a, b))) not in running_pairs:
            uf.union(a, b)
    for c in cands:  # press fits
        if c.clearance <= press_fit_tol:
            uf.union(c.link_a, c.link_b)

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
    review = []

    # component graph from running fits, re-inferred with the final grouping so a
    # pin's press-fit length counts as held, not free (revolute vs cylindrical)
    for p in parts:
        p.link = uf.find(p.name)
    running = [c for c in infer_joints(parts, scale) if press_fit_tol < c.clearance <= max_running_clearance]
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
        review.append(f"links {orphans}: no running fit to the rest, merged into {comp_name[root]}")

    spec = {
        "robot": re.sub(r"[^a-z0-9_]", "_", step_path.stem.lower()),
        "source": os.path.abspath(step_path),
        "units": units,
        "materials": {"default": 1200},
        "part_materials": {"*": "default"},
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
