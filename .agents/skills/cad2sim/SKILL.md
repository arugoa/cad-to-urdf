---
name: cad2sim
description: Convert a CAD robot (Onshape, SolidWorks, Fusion, Creo, a STEP file, or an existing URDF) into simulation-ready files for ManiSkill, MuJoCo, Isaac Lab, Gazebo or PyBullet using this repo's deterministic router (python -m cad2urdf.route). Use when the user wants a URDF/SRDF/MJCF/sim config from CAD, asks to "put this robot in <simulator>", or hands over a .step/.urdf/Onshape link to convert.
---

# cad2sim

The conversion is deterministic: `python -m cad2urdf.route`. The code settles everything that geometry or mates can settle. Your job is the rest:
1. which route to take;
2. getting the CAD-side export done, which sometimes has to happen inside the CAD tool;
3. judgment calls the code leaves out: what a part is (`part_classes`), which generated joints are real mechanisms, limits, materials, orientation;
4. reading validation output and fixing the spec.

Never hand-edit generated URDF/MJCF/SRDF files: change the spec and re-run. Never invent geometry, frames or inertia: those come from the CAD or the exporter. Run everything from the repo root with ROS kept out of Python: `env -u PYTHONPATH .venv/bin/python -m ...`.

## 1. Pick the route

Establish all three layers, from the user or the files present, before running anything:

| Layer | Options |
|---|---|
| CAD | `onshape`, `solidworks`, `fusion`, `creo`, `urdf` (already have one) |
| Format | `native` (the exporter reads real mates: preferred), `step` (fallback: joints inferred from geometry), `urdf-export` (Onshape's built-in URDF button) |
| Simulator | `maniskill`, `mujoco`, `isaaclab`, `gazebo`, `pybullet` |

Choosing the format:
- **Onshape:** with API keys, `native` (reads every mate, no naming convention needed). Without keys, `urdf-export`, else `step`.
- **SolidWorks / Fusion / Creo:** `native` if the user can run the exporter inside the CAD tool (sw2robot, ACDC4Robot, creo2urdf), otherwise `step`.
- **A STEP file with no CAD access:** `step`.

Print the plan and show the user its "needs" and "caveat" lines:

```bash
python -m cad2urdf.route --cad <cad> --format <fmt> --sim <sim>
python -m cad2urdf.route --list          # full matrix
```

## 2. Get the input

- **Onshape native:** confirm `ONSHAPE_ACCESS_KEY` and `ONSHAPE_SECRET_KEY` are set (never ask for them in chat or commit them). `--input` is the assembly tab's URL. Responses are cached in `~/.cache/cad2urdf/onshape`.
- **In-CAD exporters** (the plan says `[runs inside the CAD tool]`): give the user the plan's steps. If a CAD MCP server is connected (Autodesk Fusion MCP, a SolidWorks COM MCP, CREOSON), you may drive the export yourself, then use the resulting URDF as `--input`.
- **STEP:** export the whole assembly as one file (AP242/AP214), not one file per part.

## 3. Run

```bash
python -m cad2urdf.route --cad <cad> --format <fmt> --sim <sim> --run \
    --input <file|url> --out build/<robot> \
    --part-classes .agents/skills/cad2sim/part_classes.yaml [--spec build/<robot>.overrides.yaml]
```

Always pass the part-class library: the code has no name knowledge of its own, so without it nothing is dropped as a fastener and no bearing, servo or gear is recognised.

Outputs in `--out`: `<robot>.urdf`, `.srdf`, `mjcf/`, `maniskill/`, `isaaclab/`, `gazebo/`, `meshes/`, `report.json`, `validation.json`, and for STEP input `robot_spec.draft.yaml`.

### Part classes: what a part is

`part_classes.yaml` holds regexes for `fastener`, `not_fastener`, `non_physical`, `ignore`, `bearing`, `gear`, `servo` and `root`. Treat it as a starting point:
1. Skim the part names before running (`python -m cad2urdf.step <file> --part-classes <library>` for STEP, the assembly tree for Onshape).
2. Where a name misleads, add a pattern to `part_classes:` in the overrides file (merged per class over the library): a vendor part number for a servo, `hold-down clamp` as a fastener, `cone bearing` as a bearing.
3. Fasteners are dropped from visuals and collision (their mass stays) and their mates never become joints. A part that looks like a fastener but carries load, such as a lead screw, goes in `not_fastener`.
4. After the run, check that the REVIEW lines about bearings, servos and gears match the robot.

Record the choice in the spec. It is never a reason to edit the code.

## 4. Resolve REVIEW items (STEP input)

The draft comes from fixed rules: touching parts form one link, a running fit is a joint, a press fit is fixed. Every guess is a `# REVIEW ...` line at the top of the draft. Resolve them in an overrides file, merged over the draft, so include only what changes:

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

Where the answers come from:
1. **Joint limits:** the user, mate limits in the CAD, or the mechanism's hard stops. Never leave the ±π placeholder on a joint that has stops.
2. **Cylindrical fits (slide or spin):** the part names and the mechanism (a rail means prismatic, an axle revolute).
3. **Masses:** the real materials; ask for measured masses of motors and batteries.
4. **Mirrored joints (mimic):** symmetric grippers, gear pairs.

Link and joint names come from part names. To rename them, edit the draft into a full spec (with `links:` and `joints:`) and pass it as `--spec`.

### When the draft can't find the joints

Signs: one giant link, joints that each move a single small solid, or `0 joints` on something that obviously moves. The CAD doesn't model the joint as a shaft in a bore (motors butted against a link, zero-clearance pivots, V-wheel slides). Do not guess axes; get them from the geometry:

```bash
python -m cad2urdf.step robot.step --part-classes <library>                  # solids grouped by part name
python -m cad2urdf.step robot.step --part-classes <library> --spec spec.yaml # grouped by your links
```

For each pair of part groups it prints the touching solids and every shared axis (coaxial cylindrical faces at any radius or clearance) with spec-ready `axis:` / `origin:` in CAD units. Then write the spec:
1. **Links:** group parts by name with `fnmatch` patterns.
2. **Joints:** a joint module or motor is a dense shared axis (dozens to hundreds of face pairs over many radii). A bolt pattern is 1–3 face pairs of small radius. A cap on a tube is dense but plainly fixed.
3. **Values:** give every joint explicit `axis` and `origin` from the inspector, so nothing is inferred.
4. **Passive spring loops** (gas springs, balance cylinders) can't close in a URDF tree: put the barrel on one link and the rod on the other, and say so.
5. **Save the spec next to the STEP** (`<name>.spec.yaml`, relative `source:`) and build with `python -m cad2urdf <spec> -o build/<robot>`.

Worked example: `examples/step/Haro380.spec.yaml`.

## 5. Onshape specifics

The code handles these on its own; do not re-decide them by hand:
- **Planar mates** between the same two bodies are combined: one is two slides and a spin (passive), two with different plane normals are a single slide along the planes' intersection, three with independent normals are a rigid joint.
- **Fasteners:** a part in the `fastener` class mated with a slot, cylindrical or revolute mate is fixed to its part. An explicitly named `dof_*` mate is always honoured.
- **Mates to the assembly origin** (including mates that list only one entity) move the body against the grounded part, or the first part of that sub-assembly. A REVIEW line names the stand-in.
- **Parts named like FOV cones, frustums or keep-out volumes**, or matching the `ignore` class, are not imported.
- **Loops:** mates named `closing_*` are loop closures (MJCF `<equality connect>`). Joints inside the loop become passive; only joints on the base keep motors. If a parallel mechanism's loop isn't closed in Onshape, ask the user to rename the closing mate `closing_<name>`.

Judgment calls that stay with you:
- **Orientation:** Onshape does not report a document's up axis. If the robot loads sideways, set `root_rpy: [1.5708, 0, 0]` (Y-up documents). STEP exports are often Y-up too: check the base joint's axis.

## 6. Trimming extra joints

Mates and shaft fits also produce joints that are not mechanisms: bearings, shaft collars, pulleys, screws, a part that rides a shaft. Which joints matter is a semantic call: decide it with the user and record it in the spec. Do not add a keep-list option to the code, and do not edit generated files.

1. List what was generated: `grep -E '<joint |<parent|<child|<axis' <robot>.urdf`. Note each joint's name, parent, child, type and axis, and the mass each one carries.
2. Decide which are real. If the user names the working joints, those are the keep list. Otherwise judge from names, the carried mass and the mechanism: wheels, yaw and pitch, flywheels, arm joints, grippers and slides are real; bearing, collar, spacer, pulley and fastener joints are not. A planar chain (`<name>_x`, `_y`, `_z`) is one mechanism: keep or fix all three. If unsure, ask.
3. Write every other joint into the overrides file as fixed (the same entry works for Onshape, STEP and exporter-URDF input):
   ```yaml
   joints:
     bearing_center_1_cylindrical_2: {type: fixed}
     shaft_collar_1_planar_1_x: {type: fixed}
   ```
4. Re-run with `--spec`. Fixed joints stay in the URDF as `type="fixed"` (simulators merge them) and disappear from the MJCF, actuators, SRDF groups and Isaac Lab / ManiSkill files.
5. Check that the non-fixed joints in the URDF are exactly your keep list. A joint name can change between runs when the CAD or the rules change, and an override for a name that no longer exists is silently ignored. Regenerate the list from the latest build, then confirm `validation.json` still passes. Tell the user which joints you fixed and why.

## 7. Read validation.json and iterate

| Symptom | Likely cause | Fix in the spec |
|---|---|---|
| `penetrating_pairs_at_zero` not empty | collision shapes too fat at a joint | switch that link to `decompose` or `primitives` in `collision:` |
| `tracking_err_max` large, sim stable | weak gains or high joint friction/damping from the input URDF | raise `actuators.<joint>.kp`/`kv`, or override `dynamics` |
| `joints_missing` | joint not actuated in that sim | add an `actuators` entry |
| `mimic_err_max` > 1e-3 | missing or incorrect `mimic` | set `mimic` and `axis_sign` |
| GPU `spread_across_envs` > 0 | contact jitter against the ground or decorative parts | set that link's collision to `none` |
| MuJoCo URDF import fails, MJCF works | expected (massless dummy links, `package://` paths) | use `mjcf/<robot>.xml` for MuJoCo |
| robot loads sideways | the document or export is Y-up | `root_rpy: [1.5708, 0, 0]` |

Report to the user: the route used; which REVIEW items you resolved and how; the validation lines for the target simulator; anything you could not verify (Isaac Lab and Gazebo need their own installs).

## 8. Determinism

The same inputs must give byte-identical outputs: all sampling is seeded, ordering is sorted, and the router makes every decision that geometry or mates can settle. Stay on the right side of this line:

| Decide in the spec, from the user or the mechanism | Never decide by hand: the code settles it from geometry or mates |
|---|---|
| joint limits, effort, velocity | joint axes and origins (take them from the inspector or the mates) |
| which joints are driven, passive or speed-controlled | which mates are rigid; how several planar mates combine |
| materials, measured masses | inertia, collision shapes, mesh decimation |
| which parts are fasteners, bearings, servos, gears or non-physical (`part_classes`) | which parts form links and joints (shaft/bore fits) |
| which generated joints are real mechanisms (fix the rest) | loop closures from `closing_*` mates |
| slide vs spin for a cylindrical fit; mimic and gear ratios | joint names the exporter produced |
| the up axis when a robot loads sideways (`root_rpy`) | self-collision matrix and SRDF excludes |
| what a placeholder limit (±π, ±0.1 m) should be | which output files are written |

Rules:
- If two runs on the same input differ, that is a bug. Diff the output folders (`find . -type f | sort | xargs md5sum`) and report which file changed; do not paper over it in the spec.
- If you apply the same manual fix to a second robot, the fix belongs in the code as a rule with a test. Say so to the user instead of repeating it.
- Anything a rule doesn't settle goes in the spec as an explicit value, so the next run reproduces it.

## Out of scope

Static scenery such as competition fields: surface-only STEP files have no solids to build links from. Say so and don't force them through this route.
