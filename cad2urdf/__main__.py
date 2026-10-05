"""cad2urdf compiler: python -m cad2urdf SPEC.yaml -o OUTDIR [--study]

spec -> IR -> simplified visuals -> collision -> URDF, SRDF, MJCF, Gazebo, Isaac Lab, ManiSkill -> report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import geometry, model, writers


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cad2urdf")
    ap.add_argument("spec", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--study", action="store_true", help="also evaluate every collision mode on every link")
    ap.add_argument("--samples", type=int, default=None, help="self-collision samples (overrides spec)")
    args = ap.parse_args(argv)
    t0 = time.time()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("1/6 reading CAD + spec, inferring joints, computing inertia")
    robot = model.build(args.spec)
    for c in robot.candidates:
        print("   candidate", c.summary())

    cfg = robot.spec.get("simplify", {})
    if cfg is not False:
        rep = geometry.simplify_robot(robot, cfg.get("drop_fasteners", True), cfg.get("visual_faces_per_link", 20000))
        before, after = sum(r["faces_before"] for r in rep.values()), sum(r["faces_after"] for r in rep.values())
        print(f"   simplified visuals: {before:,} -> {after:,} triangles, "
              f"{sum(r['dropped'] for r in rep.values())} fasteners dropped")

    print("2/6 collision geometry")
    geometry.build_collisions(robot)

    print("3/6 meshes + URDF")
    writers.export_meshes(robot, out / "meshes")
    writers.write_urdf(robot, out / f"{robot.name}.urdf", mesh_prefix="meshes")
    writers.write_urdf(robot, out / "gazebo" / f"{robot.name}.gazebo.urdf",
                    mesh_prefix=f"package://{robot.name}_description/meshes", flavour="gazebo")

    print("4/6 SRDF (sampled self-collision matrix)")
    s = robot.spec.get("srdf", {})
    samples = args.samples or s.get("collision_samples", 5000)
    disabled, stats = writers.collision_matrix(robot, out / "meshes", samples=samples)
    writers.build_srdf(robot, disabled).write(out / f"{robot.name}.srdf", encoding="unicode", xml_declaration=True)

    print("5/6 MJCF, USD + Isaac Lab / ManiSkill / Gazebo side files")
    # MuJoCo doesn't filter parent/child contacts when the parent is welded to the world
    excludes = [p for p, r in disabled.items() if r in ("Adjacent", "Default", "Always")]
    keyframes = {}
    for name, st in s.get("group_states", {}).items():
        keyframes[name] = writers.expand_mimic(robot, {**{j: 0.0 for j in robot.joints}, **st["joints"]})
    writers.write_mjcf(robot, out / "mjcf" / f"{robot.name}.xml", meshdir="../meshes", excludes=excludes,
                    keyframes=keyframes)
    from . import usd_asset  # lazy: pxr loads heavy native libraries

    usd_asset.write_usd(robot, out / "usd", excludes=excludes)
    writers.write_targets(robot, out)

    print("6/6 report + joint-limit sweep")
    from .geometry import limit_sweep

    sweep = limit_sweep(robot)
    for jn in sweep["limits_driving_into_parent"]:
        print(f"  REVIEW joint {jn}: its limits drive the child into the parent's material "
              f"({sweep['joints'][jn]}); check the limit sign/offset")
    report = {
        "robot": robot.name,
        "joint_candidates": [c.summary() for c in robot.candidates],
        "links": {
            n: {
                "parts": [p.name for p in l.parts],
                "mass_kg": round(l.mass, 5),
                "com_m": np.round(l.com, 5).tolist(),
                "inertia_diag": np.round(np.diag(l.inertia), 8).tolist(),
                "inertia_issues": model.check_inertia(n, l.inertia, l.mass),
                "collision_mode": l.collision_mode,
                "collision": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in l.collision_metrics.items()},
            }
            for n, l in robot.links.items()
        },
        "limit_sweep": sweep,
        "srdf_disabled": {f"{a}|{b}": r for (a, b), r in sorted(disabled.items())},
        "collision_sampling": stats,
    }
    if args.study:
        report["collision_study"] = collision_study(robot)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    return robot, report


STUDY_MODES = [
    {"mode": "box"},
    {"mode": "primitives", "min_part_fraction": 0.02},
    {"mode": "hull"},
    {"mode": "decompose", "threshold": 0.05, "max_hulls": 8},
    {"mode": "decompose", "threshold": 0.02, "max_hulls": 32},
]


def collision_study(robot: model.Robot) -> dict:
    max_v = robot.spec.get("collision", {}).get("max_hull_vertices", 64)
    table = {}
    for link in robot.links.values():
        keep = link.collisions, link.collision_mode, link.collision_metrics
        rows = {}
        for cfg in STUDY_MODES:
            label = cfg["mode"] + (f"@{cfg['threshold']}" if "threshold" in cfg else "")
            t = time.time()
            link.collisions = geometry.link_collisions(link, cfg, max_v)
            m = geometry.metrics(link)
            m["seconds"] = round(time.time() - t, 2)
            rows[label] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
        link.collisions, link.collision_mode, link.collision_metrics = keep
        table[link.name] = rows
    return table


if __name__ == "__main__":
    from cad2urdf.util import sandbox

    sandbox("cad2urdf")
    main()
