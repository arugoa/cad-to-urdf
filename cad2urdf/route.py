"""cad2sim router CLI: pick the route from (CAD, format, simulator) and run it.

    python -m cad2urdf.route --list
    python -m cad2urdf.route --cad onshape --format native --sim maniskill                 # plan only
    python -m cad2urdf.route --cad onshape --format step --sim maniskill --run \\
        --input robot.step --out build/robot [--spec overrides.yaml]
    python -m cad2urdf.route --cad solidworks --format native --sim mujoco --run --input robot.urdf --out build/r
    python -m cad2urdf.route --cad onshape --format native --sim isaaclab --run \\
        --input "https://cad.onshape.com/documents/..." --out build/r      # needs ONSHAPE_* keys (our own API client)

Everything here is deterministic. The only judgment calls (joint limits,
materials, mimic couplings for STEP input) are written as REVIEW lines in the
draft spec; pass corrections with --spec.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

from . import routes


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_spec(args, fe: routes.FrontEnd, fin: routes.Finish) -> tuple[dict, list[str]]:
    review: list[str] = []
    user = yaml.safe_load(Path(args.spec).read_text()) if args.spec else {}
    if fe.produces == "step":
        if user.get("links") and user.get("joints"):
            spec = dict(user)
            spec["source"] = os.path.abspath(args.input) if args.input else spec["source"]
        else:
            from .draft import draft_spec, write_draft

            spec, review = draft_spec(Path(args.input), units=user.get("units", "mm"))
            write_draft(spec, review, args.out / "robot_spec.draft.yaml")
            spec = deep_merge(spec, user)
    else:
        src = str(args.input)
        src = src if src.startswith("http") else os.path.abspath(src)  # Onshape URL or exporter URDF
        spec = deep_merge({"source": src}, user)
        spec["source"] = src
    spec.setdefault("collision", {})
    spec["collision"].setdefault("default", fin.collision_default)
    spec.setdefault("actuators", {"default": {"kind": "position", "kp": 100.0, "kv": 5.0}})
    return spec, review


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cad2urdf.route", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cad", choices=routes.CADS)
    ap.add_argument("--format", default="native", help="native | step | urdf-export (Onshape)")
    ap.add_argument("--sim", choices=routes.SIMS)
    ap.add_argument("--list", action="store_true", help="print the full routing matrix")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--input", help="STEP file, exporter URDF, or Onshape document URL")
    ap.add_argument("--out", type=Path, default=Path("build/robot"))
    ap.add_argument("--spec", help="YAML overrides merged on top of the generated spec")
    ap.add_argument("--samples", type=int, default=2000, help="self-collision samples for the SRDF")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print(routes.matrix_markdown())
        return
    if not (args.cad and args.sim):
        ap.error("--cad and --sim are required (or --list)")
    fe, fin = routes.front_end(args.cad, args.format), routes.FINISHES[args.sim]
    print(routes.plan(args.cad, args.format, args.sim))
    if not args.run:
        return
    if not args.input:
        ap.error("--run needs --input")
    if not fe.automatable_here and not str(args.input).lower().endswith(".urdf"):
        ap.error(f"{fe.tool} runs inside the CAD tool; export there and pass the resulting URDF as --input")

    args.out.mkdir(parents=True, exist_ok=True)
    spec, review = build_spec(args, fe, fin)
    spec_path = args.out / "robot_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False, width=120))
    print(f"\nspec -> {spec_path}")
    for r in review:
        print(f"  REVIEW {r}")

    from .__main__ import main as compile_main

    compile_main([str(spec_path), "-o", str(args.out), "--samples", str(args.samples)])
    if not args.no_validate:
        # separate process: simulator native libs (SAPIEN/PhysX) can crash when loaded after CoACD/OCC
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        subprocess.run([sys.executable, "-m", "cad2urdf.validate", str(args.out), "--sims",
                        ",".join(dict.fromkeys(["yourdfpy", *fin.checks]))], env=env)
    if review:
        print(f"\n{len(review)} REVIEW item(s) in {args.out / 'robot_spec.draft.yaml'}; "
              "fix them in an overrides file and re-run with --spec.")


if __name__ == "__main__":
    main()
