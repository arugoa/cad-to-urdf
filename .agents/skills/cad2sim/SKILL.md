---
name: cad2sim
description: Convert a CAD robot (Onshape, SolidWorks, Fusion, Creo, a STEP file, or an existing URDF) into simulation-ready files for ManiSkill, MuJoCo, Isaac Lab, Gazebo or PyBullet using this repo's deterministic router (python -m cad2urdf.route). Use when the user wants a URDF/SRDF/MJCF/sim config from CAD, asks to "put this robot in <simulator>", or hands over a .step/.urdf/Onshape link to convert.
---

# cad2sim

The conversion itself is deterministic: `python -m cad2urdf.route`. Your job is only the parts code cannot decide:
1. which route to take, from the three layers;
2. getting the CAD-side export done, which sometimes has to happen inside the CAD tool;
3. resolving `REVIEW` items the code flags;
4. reading validation output and fixing the spec.

Never hand-edit generated URDF/MJCF/SRDF files. Change the spec and re-run. Never invent geometry, frames or inertia: those come from the CAD or the exporter.

Run everything from the repo root with ROS kept out of Python: `env -u PYTHONPATH .venv/bin/python -m ...`.

## 1. Pick the route (three layers)

Establish all three layers, from the user or the files present, before running anything:

| Layer | Options |
|---|---|
| CAD | `onshape`, `solidworks`, `fusion`, `creo`, `urdf` (already have one) |
| Format | `native` (the exporter reads real mates: preferred), `step` (fallback: joints inferred from geometry), `urdf-export` (Onshape's built-in URDF button) |
| Simulator | `maniskill`, `mujoco`, `isaaclab`, `gazebo`, `pybullet` |

How to choose the format:
- **Onshape:**
  - With API keys: `native` (reads every mate; no naming convention needed).
  - Without keys: `urdf-export` (Onshape's built-in export), else `step`.
- **SolidWorks / Fusion / Creo:** `native` if the user can run the exporter inside the CAD tool (sw2robot, ACDC4Robot, creo2urdf). Otherwise `step`.
- **A STEP file with no CAD access:** `step`.

Then print the plan and show the user its "needs" and "caveat" lines:

```bash
python -m cad2urdf.route --cad <cad> --format <fmt> --sim <sim>
python -m cad2urdf.route --list          # full matrix
```

## 2. Get the input

- **Onshape native:** confirm `ONSHAPE_ACCESS_KEY` and `ONSHAPE_SECRET_KEY` are exported (never ask for them to be pasted into chat or committed). `--input` is the assembly URL. The router reads mates, limits, masses and meshes through cad2urdf's own Onshape client (responses cached in `~/.cache/cad2urdf/onshape`).
- **In-CAD exporters** (the plan says `[runs inside the CAD tool]`): give the user the plan's steps. If a CAD MCP server is connected (Autodesk Fusion MCP, a SolidWorks COM MCP, CREOSON), you may drive the export yourself, then use the resulting URDF as `--input`.
- **STEP:** export the whole assembly as one file (AP242/AP214), not one file per part.

## 3. Run

```bash
python -m cad2urdf.route --cad <cad> --format <fmt> --sim <sim> --run \
    --input <file|url> --out build/<robot> [--spec build/<robot>.overrides.yaml]
```

Outputs in `--out`: `<robot>.urdf`, `.srdf`, `mjcf/`, `maniskill/`, `isaaclab/`, `gazebo/`, `meshes/`, `report.json` and `validation.json`. For STEP input it also writes `robot_spec.draft.yaml`.

## 4. Resolve REVIEW items (STEP input)

The draft spec comes from fixed rules: touching parts form one link, a running fit is a joint, a press fit is fixed. Every guess is listed at the top of the draft as `# REVIEW ...`. Resolve them in an overrides file. The overrides are merged over the draft, so include only what changes:

```yaml
# build/<robot>.overrides.yaml
materials: {aluminum: 2700, steel: 7850, pla: 1240}
part_materials: {"*bolt*": steel, "*": aluminum}
joints:
  base_to_turret: {limits: [-2.97, 2.97], effort: 40, velocity: 3}
  gripper_to_finger_2: {mimic: {joint: gripper_to_finger, multiplier: 1.0, offset: 0.0}, axis_sign: -1}
actuators: {gripper_to_finger_2: {kind: none}}
srdf: {group_states: {home: {group: arm, joints: {base_to_turret: 0}}}}
```

Where to get the answers:
1. Joint limits: the user, mate limits in the CAD, or the mechanism's hard stops. Never leave the ±π placeholder on a joint that has stops.
2. "cylindrical" fits (slide *or* spin): decide from the part names and the mechanism (a rail means prismatic; an axle means revolute).
3. Masses: the real materials; ask for measured masses of motors and batteries.
4. Mirrored joints (mimic): symmetric grippers, gear pairs.

Link and joint names come from part names. Rename them by editing the draft into a full spec (with `links:` and `joints:`) and passing that as `--spec`.

### 4b. When the draft can't find the joints

Signs: one giant link, joints that each move a single small solid, or `0 joints` on something that obviously moves. This happens when the CAD doesn't model the joint as a shaft in a bore. Common cases: motors or joint modules butted flat against a link, pivots modelled at zero clearance, V-wheel slides. Do not guess axes. Get them from the geometry:

```bash
python -m cad2urdf.step robot.step                  # solids grouped by part name
python -m cad2urdf.step robot.step --spec spec.yaml # grouped by your links
```

For each pair of part groups it prints the touching solids and every *shared axis*: coaxial cylindrical faces on both sides, at any radius or clearance, with spec-ready `axis:` / `origin:` (CAD units).

To write the spec:
1. **Links:** group the parts by the names (`0200_big_arm`, `J6`, `finger_left`, ...) using `fnmatch` patterns.
2. **Joints:** a joint module or motor is a *dense* shared axis, dozens to hundreds of face pairs over many radii. A bolt pattern is 1–3 face pairs of small radius. A cap on a tube is also dense but plainly fixed (read the names).
3. **Copy values:** give every joint explicit `axis` and `origin` from the inspector, so nothing is inferred.
4. **Orientation:** if the base joint's axis is +Y, the export is Y-up. Add `root_rpy: [1.5708, 0, 0]`.
5. **Passive spring loops** (gas springs, balance cylinders) can't close in a URDF tree. Put the barrel on one link and the rod on the other, and say so.
6. **Save the spec next to the STEP** (`<name>.spec.yaml`, `source:` relative) and build with `python -m cad2urdf <spec> -o build/<robot>`.

Worked example: `examples/random_step/Haro380.spec.yaml`.

Onshape mates the code handles on its own (do not re-decide them by hand):
- Planar mates between the same two bodies are combined. One planar mate is two slides and a spin (passive); two with different plane normals are a single slide along the planes' intersection; three with independent normals are a rigid joint.
- A screw, nut or washer mated with a slot, cylindrical or revolute mate is fixed to its part, never a joint. An explicitly named `dof_*` mate is the exception and is always honoured.

Onshape loops: mates named `closing_*` are loop closures (MJCF `<equality connect>`). Joints inside the loop become passive; only joints on the base keep motors. If a parallel mechanism's loop isn't closed in Onshape, ask the user to rename the closing mate `closing_<name>`.

## 5. Read validation.json and iterate

| Symptom | Likely cause | Fix in the spec |
|---|---|---|
| `penetrating_pairs_at_zero` not empty | collision shapes too fat at a joint | switch that link to `decompose` or `primitives` in `collision:` |
| `tracking_err_max` large, sim stable | weak gains or high joint friction/damping from the input URDF | raise `actuators.<joint>.kp`/`kv`, or override `dynamics` |
| `joints_missing` | joint not actuated in that sim | add an `actuators` entry |
| `mimic_err_max` > 1e-3 | missing or incorrect `mimic` | set `mimic` and `axis_sign` |
| GPU `spread_across_envs` > 0 | contact jitter against the ground or decorative parts | set that link's collision to `none` |
| MuJoCo URDF import fails, MJCF works | expected (massless dummy links, `package://` paths) | use `mjcf/<robot>.xml` for MuJoCo |

Report to the user:
1. the route used;
2. which REVIEW items you resolved and how;
3. the validation lines for the target simulator;
4. anything you could not verify (Isaac Lab and Gazebo need their own installs).

## Determinism

The same inputs must give byte-identical outputs: all sampling is seeded, ordering is sorted, and the router makes every decision that geometry or mates can settle. Your job is only what code cannot decide. Check that you stay on that side of the line:

| Decide in the spec, from the user or the mechanism | Never decide by hand |
|---|---|
| joint limits, effort, velocity | joint axes and origins (take them from the inspector or mates) |
| which joints are driven, passive or speed-controlled | which parts are fasteners, or which mates are rigid |
| materials, measured masses | inertia, collision shapes, mesh decimation |
| slide vs spin for a cylindrical fit; mimic and gear ratios | loop closures from `closing_*` mates |
| what a placeholder limit (+/-pi, +/-0.1 m) should be | joint names the exporter produced |

Rules:
- If two runs on the same input differ, that is a bug. Diff the output folders (`find . -type f | sort | xargs md5sum`) and report which file changed; do not paper over it in the spec.
- If you find yourself applying the same manual fix to a second robot, the fix belongs in the code as a rule with a test. Say so to the user instead of repeating it.
- Anything not settled by a rule goes in the spec as an explicit value, so the next run reproduces it.

## Out of scope

Static scenery such as competition fields: surface-only STEP files have no solids to build links from. Say so and don't force them through this route.
