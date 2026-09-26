# cad-to-urdf

Convert CAD assemblies from **Onshape, SolidWorks, Fusion, Creo or plain STEP** into **simulation-ready** robot descriptions for **ManiSkill, MuJoCo, Isaac Lab, Gazebo and PyBullet**:

- **URDF + SRDF:** a self-collision matrix sampled MoveIt-style.
- **Native MJCF:** actuators, armature, mimic joints as equality constraints, contact excludes, keyframes.
- **Per-simulator side files:** a ManiSkill agent class, an Isaac Lab `ArticulationCfg`, Gazebo `ros2_control` controllers.
- **Convex collision geometry chosen per link:** primitives where they fit, CoACD where they don't, ≤ 64 vertices per hull for GPU PhysX. It is scored against the real CAD volume.
- **Static scenery** (competition fields, arenas) from surface-only STEP exports.
- **Validation:** every output is loaded and driven in each simulator installed on your machine.

The conversion is **deterministic**: the same inputs always give the same outputs. The few judgment calls (joint limits, materials, running an exporter inside a CAD tool) are handled by a small spec file, or by the included [`cad2sim` agent skill](#using-it-with-claude-code--codex).

![arm4: visual vs collision](docs/img/arm4_visual_vs_collision.png)

---

## Contents

- [How it works](#how-it-works)
- [Setup](#setup)
- [Onshape setup](#onshape-setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [The spec file](#the-spec-file)
- [Using it with Claude Code / Codex](#using-it-with-claude-code--codex)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Status and limitations](#status-and-limitations)

---

## How it works

Pick three things: **CAD package → export format → simulator**.

```
 CAD + format ──► front end ──────────────► finish (cad2urdf) ──────────────► validate
 onshape/native    Onshape REST API (mates)   links, inertia, collision,        MuJoCo, PyBullet,
 onshape/urdf-export  Onshape URDF export      SRDF, URDF, MJCF, ManiSkill,      SAPIEN, ManiSkill
 solidworks/native    sw2robot (in SW)         Isaac Lab, Gazebo files           (CPU + GPU)
 fusion/native        ACDC4Robot (in Fusion)
 creo/native          creo2urdf (in Creo)
 */step               geometric joint inference + drafted spec
 urdf/native          any existing URDF
```

**Front ends** read the CAD. When an exporter can read your real mates, it's used, because mates are the ground truth for what moves. STEP is the fallback: it keeps the shapes but loses the mates, so joints are inferred from geometry.

**The finish** is the same for every route, so every simulator gets the same links, inertia, collision and SRDF.

`python -m cad2urdf.route --list` prints the full matrix, and [`docs/RESEARCH.md` §9](docs/RESEARCH.md#9-routing-cad--format--simulator) explains the reasoning behind each route.

---

## Setup

Requires **Python 3.10–3.12** on Linux (tested on Ubuntu 22.04, Python 3.11). A CUDA GPU is optional; it's used for the ManiSkill GPU check.

```bash
git clone https://github.com/arugoa/cad-to-urdf.git
cd cad-to-urdf

# with uv (fast)
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt
# or with plain pip
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

> **ROS users:** if `/opt/ros/*/setup.bash` is sourced, ROS's `PYTHONPATH` leaks into the venv and breaks pytest and some imports. Prefix commands with `env -u PYTHONPATH`, or use a shell without ROS sourced.

Check the install:

```bash
pytest -q                                    # ~5 s, 17 tests
python -m cad2urdf.route --list              # prints the routing matrix
```

Optional simulator installs, for running the outputs rather than only generating them:
- **Isaac Lab:** follow the [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- **Gazebo + ros2_control:** `sudo apt install ros-$ROS_DISTRO-gz-ros2-control`.

---

## Onshape setup

Onshape has two routes. Pick one.

### Option A: Onshape URDF export (no setup)

In Onshape, right-click the **assembly tab** → **Export** → **URDF** (mesh format GLTF or STL). Every mate becomes a joint. Unzip the result, then:

```bash
python -m cad2urdf.route --cad onshape --format urdf-export --sim maniskill --run \
    --input path/to/unzipped/robot.urdf --out build/myrobot
```

### Option B: live document through the API (reads mates directly)

cad2urdf has its own Onshape client (`cad2urdf/onshape.py`). It reads the assembly's mates as they are, with **no naming convention**:
- fastened mates and rigid sub-assemblies → one link;
- revolute → revolute (continuous if the mate has no limits);
- slider → prismatic;
- gear / rack-and-pinion / screw relations → mimic joints;
- mate limits → joint limits.

Masses come from the materials you assigned in Onshape, and meshes are fetched per part. Responses are cached in `~/.cache/cad2urdf/onshape`, so re-runs work offline.

**1. Get API keys.** They're managed in your Onshape settings, not the old Developer portal:
- **Personal account:** user icon → **My account** → **Developer** → **API keys** → **Create new API key** (read permissions only).
- **Company / Enterprise account:** only an **admin** can create keys. They go to user icon → **Enterprise settings** → **Developer** → **API keys** → **Create new API key**, assign it to you, and send you both values.
- The secret key is shown only once. No admin available? Use Option A instead.

**2. Export them in your shell.** Never commit them. Put them in `~/.bashrc` or an untracked `.env` you `source`.

```bash
export ONSHAPE_ACCESS_KEY=<access key>
export ONSHAPE_SECRET_KEY=<secret key>
# optional: ONSHAPE_API=https://yourteam.onshape.com  (defaults to the document URL's domain)
```

**3. Run with the assembly's URL.** The `/e/...` part must be the assembly tab:

```bash
python -m cad2urdf.route --cad onshape --format native --sim maniskill --run \
    --input "https://cad.onshape.com/documents/<doc>/w/<workspace>/e/<assembly>" --out build/myrobot
```

Anything the client can't map (ball or planar mates, kinematic loops, sliders without limits) is printed as a `REVIEW` line and can be fixed with `--spec`. More detail: [`docs/ONSHAPE_API_KEYS.md`](docs/ONSHAPE_API_KEYS.md).

---

## Usage

All generated files go to `build/` (git-ignored). Everything can be regenerated from the inputs.

### Robots: one command per route

```bash
# See the plan for a route without running anything
python -m cad2urdf.route --cad onshape --format native --sim maniskill

# STEP file (any CAD): joints inferred from geometry, spec drafted automatically
python -m cad2urdf.route --cad solidworks --format step --sim maniskill --run \
    --input robot.step --out build/robot

# URDF from an exporter (sw2robot, ACDC4Robot, creo2urdf, Onshape export, ...) → finish for MuJoCo
python -m cad2urdf.route --cad fusion --format native --sim mujoco --run \
    --input exported/robot.urdf --out build/robot

# Same, with corrections merged on top of the generated spec
python -m cad2urdf.route --cad onshape --format step --sim isaaclab --run \
    --input robot.step --out build/robot --spec robot.overrides.yaml
```

| `--sim` | Main file to load | Also written |
|---|---|---|
| `maniskill` | `maniskill/<robot>_agent.py` (registers an agent uid) | URDF + SRDF |
| `mujoco` | `mjcf/<robot>.xml` | URDF |
| `isaaclab` | `isaaclab/<robot>_cfg.py` (`<ROBOT>_CFG`) | URDF |
| `gazebo` | `gazebo/<robot>.gazebo.urdf` + `gazebo/config/controllers.yaml` | — |
| `pybullet` | `<robot>.urdf` (load with `URDF_USE_INERTIA_FROM_FILE`) | SRDF |

Every route writes all of these files. `--sim` only picks the collision defaults and which checks to run.

**Your own spec instead of the drafted one:** run the compiler directly. The fully worked example is `examples/arm4/robot_spec.yaml`.

```bash
python -m cad2urdf examples/arm4/robot_spec.yaml -o build/arm4 [--study]   # --study compares collision modes
```

### Static scenes (fields, arenas)

For scenery without joints, including surface-only exports that aren't solids:

```bash
python -m cad2urdf.scene field.step -o build/field
```

This writes `<name>.urdf` (one fixed link), `mjcf/<name>.xml`, and `maniskill/<name>_scene.py`, which exposes `build_<name>(scene)` for your env's `_load_scene`. It also runs a **drop test**: balls dropped across the scene must land on the visual surface. Results go to `validation.json`.

### Validate

The route runs validation automatically. To re-run it:

```bash
python -m cad2urdf.validate build/robot --sims yourdfpy,mujoco,pybullet,sapien,maniskill
```

Each simulator drives every actuated joint to a test pose and reports stability, tracking error, mimic-joint error, penetration at rest and GPU consistency.

### View it in the simulator

```bash
python -m cad2urdf.scene examples/field_cad/2026_ARC_3v3.step -o build/field      # generate first
python tests/view_urdf.py build/field --sim maniskill                            # SAPIEN viewer window
python tests/view_urdf.py build/field --sim mujoco                               # MuJoCo viewer
python tests/view_urdf.py build/robot --sim maniskill --scene build/field --at 0 0 0.02   # robot on the field
python tests/view_urdf.py build/robot                                            # browser: sliders, collision toggle, SRDF poses
python tests/view_urdf.py build/field --sim maniskill --screenshot field.png     # save an image, no window
```

Robots open holding their first keyframe under PD control.
- **MuJoCo:** keys `2`/`3` toggle the visual and collision groups; the *Control* panel moves the joints.
- **Browser mode** serves http://localhost:8080.

---

## Outputs

```
build/robot/
├── <robot>.urdf                  neutral URDF: relative mesh paths, full inertia, dynamics, mimic
├── <robot>.srdf                  groups, poses, end effector, sampled disable_collisions
├── mjcf/<robot>.xml              MuJoCo: actuators, armature, equality, excludes, keyframes
├── maniskill/<robot>_agent.py    ManiSkill 3 BaseAgent (PD + mimic controllers, keyframes)
├── isaaclab/<robot>_cfg.py       Isaac Lab ArticulationCfg (UrdfFileCfg + actuators)
├── gazebo/<robot>.gazebo.urdf    package:// paths, <gazebo> friction, <ros2_control>
├── gazebo/config/controllers.yaml
├── meshes/visual/*.stl           one mesh per link per material
├── meshes/collision/*.stl        each file = one convex piece (≤ 64 vertices)
├── robot_spec.yaml               the exact spec used (reproducible)
├── robot_spec.draft.yaml         STEP input only: drafted spec with REVIEW notes
├── report.json                   joint candidates, masses, inertia checks, collision IoU per link
└── validation.json               what each simulator actually loaded and did
```

---

## The spec file

The spec is the only hand-edited input. For a STEP file it is **drafted for you** and every guess is marked `# REVIEW`. Pass corrections with `--spec`; they are merged over the draft, so you only write what changes:

```yaml
# robot.overrides.yaml
materials: {aluminum: 2700, steel: 7850, pla: 1240}          # kg/m^3
part_materials: {"*bolt*": steel, "*": aluminum}              # first match wins
joints:
  base_to_turret: {limits: [-2.97, 2.97], effort: 40, velocity: 3}
  gripper_to_finger_2: {mimic: {joint: gripper_to_finger}, axis_sign: -1}
dynamics:  {default: {damping: 0.5, friction: 0.05, armature: 0.01}}
actuators: {default: {kind: position, kp: 200, kv: 10}, gripper_to_finger_2: {kind: none}}
collision: {default: {mode: auto}, finger: {mode: decompose, threshold: 0.03}}
srdf:
  group_states: {home: {group: arm, joints: {base_to_turret: 0}}}
```

For exporter URDFs, the useful keys are `package_dirs` (resolve `package://`) and `root_rpy` (re-orient Y-up SolidWorks exports: `[1.5708, 0, 0]`).

Collision modes, cheapest to most faithful: `none`, `box`, `primitives`, `hull`, `auto` (default), `decompose`, `keep` (use the input URDF's own collision geometry).

---

## Using it with Claude Code / Codex

The repo ships an agent skill, **`cad2sim`**, in `.agents/skills/cad2sim/SKILL.md` (symlinked into `.claude/skills/`, so both Claude Code and Codex find it). Ask something like *"convert examples/robot_cad/hero.step for ManiSkill"*. The agent will:

1. pick the route;
2. run the router;
3. resolve the `REVIEW` items, asking you for joint limits and materials where needed;
4. read `validation.json` and fix the spec until the checks pass.

The conversion itself stays deterministic. The agent only edits the spec, never the generated files.

---

## Repository layout

```
cad2urdf/
  route.py, routes.py    3-layer router (CAD × format × simulator) and its CLI
  cad.py                 STEP loading, B-rep mass properties, tessellation
  draft.py               deterministic spec draft for mate-less STEP
  joints.py              shaft/bore joint inference (k-d tree indexed)
  onshape.py             Onshape REST client: mates, limits, mass properties, meshes → internal model
  ingest.py              exporter URDF → internal model
  model.py               internal model (links, joints, frames, inertia)
  collision.py           per-link collision modes + IoU scoring
  urdf.py, srdf.py, mjcf.py, targets.py   writers
  scene.py               static scenes (fields) + drop test
  validate.py            robot-agnostic checks in every local simulator
examples/arm4/           parametric sample robot (build_cad.py → STEP → spec → outputs)
tests/                   pytest suite + view_urdf.py (ManiSkill / MuJoCo / browser viewer)
docs/RESEARCH.md         research: tools, AI/MCP, joints, collision, dynamics, per-simulator needs
docs/ONSHAPE_API_KEYS.md
.agents/skills/cad2sim/  agent skill (Claude Code + Codex)
```

---

## Tests

```bash
pytest -q
```

Covers:
- joint inference;
- inertia (checked against an analytic box);
- collision fitting and vertex caps;
- the SRDF matrix;
- the drafted spec on the sample STEP;
- the URDF round trip (identical kinematics);
- rotated-frame (Y-up) ingest;
- the Onshape front end against a fake API (mates → links and joints, limits, masses, MuJoCo load).

---

## Status and limitations

| Route | Status |
|---|---|
| STEP → any sim (sample arm) | ✅ fully automatic; 7 links and 6 joint types recovered from geometry; runs in MuJoCo, PyBullet, SAPIEN, ManiSkill CPU + GPU |
| Exporter URDF → any sim (SolidWorks-exported infantry) | ✅ loads and runs in SAPIEN, ManiSkill (CPU + GPU) and MuJoCo (via our MJCF) |
| Static scene (ARC 3v3 field, surface-only STEP) | ✅ 925 collision shapes; 0 of 400 drop-test balls fall through |
| Onshape native (own REST client) | ⚠️ tested offline against recorded API response shapes; not yet run against a live document |
| Isaac Lab / Gazebo | ⚠️ files generated; Isaac Lab not executed; Gazebo checked with `gz sdf` conversion only |

Known limitations:
- **STEP loses mates.** Inferring joints from geometry works when running fits are drawn with clearance. On large real assemblies, parts that touch everywhere weld links together: a bearing touches both the shaft and the housing, and gears and belts touch each other. On a 1,142-part robot the draft found only one joint. For real robots, use a route that reads the mates (Onshape export/API, sw2robot, ACDC4Robot, creo2urdf).
- **Joint limits and real masses** aren't in the geometry. Set them in the spec.
- **Large STEP files** are slow the first time (about 5 min for 1,000+ parts). The contact search is then cached in `~/.cache/cad2urdf/`.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full survey of existing tools, AI/MCP options, and how each simulator consumes URDF/SRDF and dynamics.
