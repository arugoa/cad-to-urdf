"""Deterministic routing: (CAD software, export format, simulator) -> conversion plan.

    python -m cad2urdf.route --cad onshape --format native --sim maniskill            # print the plan
    python -m cad2urdf.route --cad onshape --format native --sim maniskill --run ...  # and execute it

Layer 1 (CAD) + layer 2 (format) choose the FRONT END: the tool that reads the CAD
and gives us links, joints, limits, inertia and per-part meshes. Where a mature
exporter exists we use it (it reads real mates); STEP is the universal fallback
(geometric joint inference + a spec).

Layer 3 (simulator) chooses the FINISH: collision defaults, which files to
write, and which checks to run. The finish is always cad2urdf's own compiler, so
every route ends in the same validated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        tool="cad2urdf Onshape front end (cad2urdf/onshape.py) via the Onshape REST API",
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
