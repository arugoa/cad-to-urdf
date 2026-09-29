"""List part groups, their contacts and shared axes in a STEP file, for writing a spec by hand.

    python -m cad2urdf.inspect_step robot.step                    # solids grouped by part name
    python -m cad2urdf.inspect_step robot.step --spec spec.yaml   # grouped by the spec's links

A shared axis is a set of coaxial cylindrical faces on both groups, at any radius or clearance. Each is
printed as spec-ready ``axis`` / ``origin`` (CAD units). Motors and joint modules show up as dense axes
(dozens of face pairs); bolt holes as 1-3.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from . import cad
from .joints import _coaxial, _coaxial_index, cylindrical_faces


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
    ap = argparse.ArgumentParser(prog="cad2urdf.inspect_step", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", type=Path)
    ap.add_argument("--spec", type=Path, help="group solids by this spec's links instead of by part name")
    ap.add_argument("--units", default="mm")
    args = ap.parse_args(argv)
    scale = cad.UNIT_TO_M[args.units]
    parts = cad.load_parts(args.step)
    if args.spec:
        links = yaml.safe_load(args.spec.read_text())["links"]
        for p in parts:
            p.link = next((k for k, pats in links.items() if any(fnmatch.fnmatch(p.name, q) for q in pats)), "?")
    else:
        for p in parts:
            p.link = name_group(p.name)
    groups = defaultdict(list)
    for p in parts:
        cad.mass_properties(p, scale)
        groups[p.link].append(p)
    print(f"{len(parts)} solids in {len(groups)} groups")
    for g, ps in sorted(groups.items()):
        lo = np.min([np.array(tuple(p.shape.bounding_box().min)) for p in ps], axis=0)
        hi = np.max([np.array(tuple(p.shape.bounding_box().max)) for p in ps], axis=0)
        print(f"  {g}: {len(ps)} solid(s), {sum(p.volume for p in ps) * 1e6:.1f} cm3, "
              f"bbox {np.round(lo).tolist()} .. {np.round(hi).tolist()}")

    from .draft import _cached_touching

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
    from .safety import sandbox

    sandbox("cad2urdf.inspect_step")
    main()
