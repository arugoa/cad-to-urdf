"""Router: (CAD, export format, simulator) -> a plan, and optionally run it.

The CAD and format pick the front end; the simulator picks collision defaults and checks. Every route
ends in cad2urdf's own compiler.

    python -m cad2urdf.route --list
    python -m cad2urdf.route --cad onshape --format native --sim maniskill        # plan only
    python -m cad2urdf.route --cad solidworks --format step --sim mujoco --run --input robot.step --out build/r
    python -m cad2urdf.route --cad onshape --format native --sim isaaclab --run --input <assembly URL> --out build/r

For STEP input the guesses (limits, materials, couplings) are REVIEW lines in the draft spec; pass
corrections with --spec.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_spec(args, fe: FrontEnd, fin: Finish) -> tuple[dict, list[str]]:
    review: list[str] = []
    user = yaml.safe_load(Path(args.spec).read_text()) if args.spec else {}
    if fe.produces == "step":
        if user.get("links") and user.get("joints"):
            spec = dict(user)
            spec["source"] = os.path.abspath(args.input) if args.input else spec["source"]
        else:
            from .step import draft_spec, write_draft

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
    ap.add_argument("--cad", choices=CADS)
    ap.add_argument("--format", default="native", help="native | step | urdf-export (Onshape)")
    ap.add_argument("--sim", choices=SIMS)
    ap.add_argument("--list", action="store_true", help="print the full routing matrix")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--input", help="STEP file, exporter URDF, or Onshape document URL")
    ap.add_argument("--out", type=Path, default=Path("build/robot"))
    ap.add_argument("--spec", help="YAML overrides merged on top of the generated spec")
    ap.add_argument("--samples", type=int, default=2000, help="self-collision samples for the SRDF")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print(matrix_markdown())
        return
    if not (args.cad and args.sim):
        ap.error("--cad and --sim are required (or --list)")
    fe, fin = front_end(args.cad, args.format), FINISHES[args.sim]
    print(plan(args.cad, args.format, args.sim))
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


CADS = ("onshape", "solidworks", "fusion", "creo", "urdf")
FORMATS = ("native", "step")
SIMS = ("maniskill", "mujoco", "isaaclab", "gazebo", "pybullet")


@dataclass
class FrontEnd:
    tool: str
    produces: str  # what the finish stage ingests: "urdf" | "step"
    automatable_here: bool  # can run on Linux without the CAD GUI
    how: list[str]
    needs: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    direct: dict[str, str] = field(default_factory=dict)  # sim -> output the tool can write directly


STEP_FRONT = FrontEnd(
    tool="cad2urdf STEP front end (geometric joint inference)",
    produces="step",
    automatable_here=True,
    how=[
        "Export the WHOLE assembly as one STEP file (AP242 or AP214), not one file per part.",
        "`python -m cad2urdf.route ... --run --input model.step` writes a draft spec "
        "(links from the top-level assembly tree, joints from shaft/bore candidates); review it, then re-run.",
    ],
    needs=["robot_spec.yaml: link grouping + joint types/limits (drafted automatically, confirmed by you)"],
    caveats=[
        "STEP carries no mates: joints are inferred from coaxial shaft/bore pairs only.",
        "Flat slides, ball joints, belts/gears and joint limits are not recoverable from geometry.",
        "Surface-only exports (no solids) can be converted as static scenery, not as robots.",
    ],
)

FRONT_ENDS: dict[tuple[str, str], FrontEnd] = {
    ("onshape", "native"): FrontEnd(
        tool="cad2urdf Onshape front end (cad2urdf/frontends.py) via the Onshape REST API",
        produces="urdf",
        automatable_here=True,
        how=[
            "Create an API key pair (docs/ONSHAPE_API_KEYS.md) and export ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY.",
            "`--run --input <assembly URL>`: reads every mate (fastened -> same link, revolute/slider -> joints, "
            "gear relations -> mimic), mate limits, Onshape mass properties and per-part meshes.",
        ],
        needs=["assembly URL (the /e/ element must be the assembly tab)", "API keys in the environment"],
        caveats=["Revolute mates without limits become continuous joints: set limits in Onshape or the spec.",
                 "Ball/planar mates are not joints here (reported as REVIEW)."],
    ),
    ("onshape", "urdf-export"): FrontEnd(
        tool="Onshape native URDF export (v1.212+, Mar 2026)",
        produces="urdf",
        automatable_here=False,
        how=["Right-click the assembly tab -> Export -> URDF (mesh format GLTF or STL, Medium resolution).",
             "Unzip, then `--run --input robot.urdf`."],
        needs=["no mate naming convention needed: every mate becomes a joint"],
        caveats=["All mates become joints; fastened mates become fixed joints (merged by the finish stage)."],
        direct={"isaaclab": "Onshape -> USD export for Isaac Sim (PTC/NVIDIA workflow) keeps joints"},
    ),
    ("solidworks", "native"): FrontEnd(
        tool="sw2robot / solidworks_urdf_exporter2 (JSK, 2026)",
        produces="urdf",
        automatable_here=False,
        how=["On Windows with SolidWorks open: run the sw2robot extract step on the assembly "
             "(infers the link tree and axes from mates), fix axes in its browser editor, export URDF.",
             "Copy the URDF + meshes here, then `--run --input robot.urdf`.",
             "Fallback: ros/solidworks_urdf_exporter (manual wizard, SW coordinate frames, Y-up by default)."],
        needs=["Windows + SolidWorks for the extract step"],
        caveats=["Classic sw_urdf_exporter output is often Y-up and has package:// paths: "
                 "set root_rpy and package_dirs in the spec."],
        direct={"mujoco": "sw2robot can write MJCF itself"},
    ),
    ("fusion", "native"): FrontEnd(
        tool="ACDC4Robot add-in (URDF / SDFormat / MJCF)",
        produces="urdf",
        automatable_here=False,
        how=["In Fusion: use only Rigid / Revolute / Slider joints between top-level components; "
             "avoid rigid groups (use rigid joints); name the base component `base_link`.",
             "Run ACDC4Robot -> URDF. (An LLM with the Fusion MCP can drive this step.)",
             "Copy output here, then `--run --input robot.urdf`."],
        needs=["Fusion running; joints defined"],
        caveats=["Fusion's API works in cm; ACDC4Robot converts, hand-written scripts must too."],
        direct={"mujoco": "ACDC4Robot writes MJCF", "gazebo": "ACDC4Robot writes SDFormat"},
    ),
    ("creo", "native"): FrontEnd(
        tool="creo2urdf (IIT) from a Creo Mechanism",
        produces="urdf",
        automatable_here=False,
        how=["Model joints as Mechanism connections (Pin/Slider/Weld); put a PARENT_CHILD_INTERFACE_CSYS "
             "at each joint; write the creo2urdf YAML/CSV; run the plug-in in Creo.",
             "Copy output here, then `--run --input model.urdf`."],
        needs=["Creo Toolkit licence (creo2urdf DLL is not unlocked)"],
        caveats=["Only revolute/prismatic/fixed; ball joints become 3 revolutes. Without Toolkit: use STEP."],
    ),
    ("urdf", "native"): FrontEnd(
        tool="existing URDF (any exporter or hand-written)",
        produces="urdf",
        automatable_here=True,
        how=["`--run --input robot.urdf` (set package_dirs / root_rpy in the spec if needed)."],
    ),
}

for cad in ("onshape", "solidworks", "fusion", "creo"):
    FRONT_ENDS[(cad, "step")] = STEP_FRONT


@dataclass
class Finish:
    outputs: list[str]
    collision_default: dict
    notes: list[str]
    checks: list[str]


FINISHES: dict[str, Finish] = {
    "maniskill": Finish(
        outputs=["urdf", "srdf", "maniskill"],
        collision_default={"mode": "auto", "min_part_fraction": 0.02},
        notes=["Each collision STL is one convex piece <= 64 verts (PhysX GPU), so load_multiple_collisions=False.",
               "SAPIEN reads <robot>.srdf automatically but only applies reason=\"Default\" pairs.",
               "Mimic joints run through PDJointPosMimicController (normalize_action=False)."],
        checks=["sapien", "maniskill"],
    ),
    "mujoco": Finish(
        outputs=["mjcf", "urdf"],
        collision_default={"mode": "auto", "min_part_fraction": 0.02},
        notes=["Native MJCF: armature, position actuators, mimic -> <equality>, SRDF pairs -> <contact><exclude> "
               "(incl. Adjacent pairs: MuJoCo does not filter parent/child when the parent is welded to world).",
               "Mesh geoms are convexified by MuJoCo, so every collision mesh is already one convex piece."],
        checks=["mujoco"],
    ),
    "isaaclab": Finish(
        outputs=["urdf", "isaaclab"],
        collision_default={"mode": "auto", "min_part_fraction": 0.02},
        notes=["UrdfFileCfg with collision_from_visuals=False: the importer only cooks our convex pieces.",
               "Gains/armature/friction/limits go in ImplicitActuatorCfg, not in the URDF.",
               "self_collision=True; filter pairs from the SRDF if needed (not applied automatically)."],
        checks=["yourdfpy"],
    ),
    "gazebo": Finish(
        outputs=["urdf", "gazebo"],
        collision_default={"mode": "auto", "min_part_fraction": 0.02},
        notes=["gazebo/<robot>.gazebo.urdf: package:// paths, world link, <gazebo> mu1/mu2/kp/kd, <ros2_control>.",
               "Package the output as <robot>_description so package:// resolves."],
        checks=["yourdfpy"],
    ),
    "pybullet": Finish(
        outputs=["urdf", "srdf"],
        collision_default={"mode": "auto", "min_part_fraction": 0.02},
        notes=["Load with flags URDF_USE_INERTIA_FROM_FILE | URDF_USE_SELF_COLLISION | "
               "URDF_USE_SELF_COLLISION_EXCLUDE_PARENT (inertia is recomputed otherwise).",
               "<mimic> is ignored: add p.JOINT_GEAR constraints (see cad2urdf/validate.py)."],
        checks=["pybullet"],
    ),
}


def front_end(cad: str, fmt: str) -> FrontEnd:
    key = (cad, fmt)
    if key not in FRONT_ENDS:
        raise KeyError(f"no route for cad={cad} format={fmt}; formats: native, step"
                       + (", urdf-export" if cad == "onshape" else ""))
    return FRONT_ENDS[key]


def plan(cad: str, fmt: str, sim: str) -> str:
    fe, fin = front_end(cad, fmt), FINISHES[sim]
    lines = [f"ROUTE  {cad} / {fmt} / {sim}", "",
             f"1. Front end: {fe.tool}" + ("" if fe.automatable_here else "   [runs inside the CAD tool]")]
    lines += [f"     - {h}" for h in fe.how]
    lines += [f"     needs: {n}" for n in fe.needs]
    lines += [f"     caveat: {c}" for c in fe.caveats]
    if sim in fe.direct:
        lines.append(f"     shortcut: {fe.direct[sim]} (skips our collision/SRDF/dynamics finish)")
    lines += ["", f"2. Finish (cad2urdf): writes {', '.join(fin.outputs)}; default collision {fin.collision_default}"]
    lines += [f"     - {n}" for n in fin.notes]
    lines += ["", f"3. Validate: python -m cad2urdf.validate OUT --sims {','.join(fin.checks)}"]
    return "\n".join(lines)


def matrix_markdown() -> str:
    """The full 3-layer table, for the docs."""
    rows = ["| CAD | Format | Front end | Runs on Linux | Direct sim output |", "|---|---|---|---|---|"]
    for (cad, fmt), fe in FRONT_ENDS.items():
        rows.append(f"| {cad} | {fmt} | {fe.tool} | {'yes' if fe.automatable_here else 'no (in CAD)'} | "
                    f"{'; '.join(f'{k}: {v}' for k, v in fe.direct.items()) or '—'} |")
    return "\n".join(rows)


if __name__ == "__main__":
    from cad2urdf.util import sandbox

    sandbox("cad2urdf.route")
    main()
