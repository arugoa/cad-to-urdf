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
- [Onshape setup](#onshape-setup) ([API key guide](docs/ONSHAPE_API_KEYS.md))
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

Configure secrets (only needed for the Onshape API route):

```bash
cp .env.example .env        # .env is git-ignored; fill in your Onshape API keys
```

`.env.example` is the committed template and lists every setting cad2urdf reads. Your real `.env` stays local and is never committed. cad2urdf loads it automatically, so there's nothing to `source`.

Check the install:

```bash
pytest -q                                    # ~5 s
python -m cad2urdf.route --list              # prints the routing matrix
```

**Out-of-memory protection is on by default.** `cad2urdf.route`, `python -m cad2urdf`, `cad2urdf.scene`, `cad2urdf.validate` and `tests/view_urdf.py` re-launch themselves in the systemd user slice `cad2urdf.slice`:
- **One shared memory cap** (total RAM − 5 GB, no swap) covers *all* cad2urdf jobs together. A batch, a validation and a viewer running at once can't add up past it. Over the cap, only cad2urdf jobs are stopped (exit 137, with a message).
- **Killed first:** every cad2urdf process sets `oom_score_adj=1000`, so if the machine still runs out of memory for another reason, the kernel kills our jobs before your editor or browser.
- **Lowest CPU priority** (`nice 19`): every idle core is used, but other apps win.

Settings: `CAD2URDF_RESERVE_GB=6` keeps more RAM free; `CAD2URDF_NO_SANDBOX=1` disables the sandbox (CI, containers). For any other heavy command (pip installs, Isaac Sim), `scripts/run_safely.sh <command>` puts it in the same capped slice.

Optional simulator installs, for running the outputs rather than only generating them:
- **Isaac Lab:** follow the [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- **Gazebo + ros2_control:** `sudo apt install ros-$ROS_DISTRO-gz-ros2-control`.

---

## Onshape setup

> 🔑 **Step-by-step API key guide (personal and Enterprise accounts): [`docs/ONSHAPE_API_KEYS.md`](docs/ONSHAPE_API_KEYS.md)**

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

**1. Get API keys** (full guide: [`docs/ONSHAPE_API_KEYS.md`](docs/ONSHAPE_API_KEYS.md)). They're managed in your Onshape settings, not the old Developer portal:
- **Personal account:** user icon → **My account** → **Developer** → **API keys** → **Create new API key** (read permissions only).
- **Company / Enterprise account:** only an **admin** can create keys. They go to user icon → **Enterprise settings** → **Developer** → **API keys** → **Create new API key**, assign it to you, and send you both values.
- The secret key is shown only once. No admin available? Use Option A instead.

**2. Put them in `.env`.** Copy the template, then fill in the two values:

```bash
cp .env.example .env
# .env
ONSHAPE_ACCESS_KEY=<access key>
ONSHAPE_SECRET_KEY=<secret key>
```

`.env` is git-ignored. Never paste keys into code, commits or chat. Exporting the same variables in your shell also works; shell values take precedence.

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
docs/ONSHAPE_API_KEYS.md  how to get Onshape API keys (personal and Enterprise accounts)
scripts/run_safely.sh    run heavy conversions at full idle CPU without starving other apps (memory cap)
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

Tested on an RTX 3070 Ti laptop, Ubuntu 22.04. Check results are written to each output's `validation.json`.

| Route / input | Result |
|---|---|
| **Onshape (own REST client), 7 public robots** | ✅ joint counts match the reference exactly: 2-wheeler 2, adjustable arm 4, quadruped 12, dog 12, Sigmaban humanoid 20, Orbita parallel 7, RSK soccer 64 (3 wheels + 60 omni rollers + kicker) |
| Onshape, pneumatic cylinder | ✅ slider limits and direction from the mates; checked in MuJoCo, PyBullet, SAPIEN, ManiSkill (CPU + GPU) |
| STEP, SO-100 arm (servo-driven, saved folded) | ✅ drafted automatically: the reference's exact 7 links / 6 joints |
| STEP, sample arm4 | ✅ 7 links / 6 joints from geometry alone |
| Exporter URDF (SolidWorks infantry) | ✅ runs in SAPIEN, ManiSkill (CPU + GPU) and MuJoCo |
| Static scene (ARC 3v3 field, surface-only STEP) | ✅ 925 collision shapes; 0 of 400 drop-test balls fall through |
| Big STEP robots (TR hero, Infantry 2026 chassis) | ⚠️ drafts still wrong: 1,000+ parts with axle screws, belts and suspension linkages. Use the Onshape route |

Simulators checked by `cad2urdf.validate`:
- MuJoCo (URDF import and native MJCF), PyBullet, SAPIEN, ManiSkill 3 (CPU and GPU PhysX) and yourdfpy;
- **Gazebo Classic 11**: URDF → SDF, headless `gzserver`, physics stepped, joint angles and link poses read back. Mimic joints are dropped for this check, because Classic needs a plugin for them;
- Isaac Sim: an import-and-drive probe exists (`cad2urdf/isaac_probe.py`), but Isaac Sim 5.1's RTX renderer crashes at startup on this laptop's NVIDIA 595 driver. Isaac Sim 6.1 is being tried.

Known limitations:
- **STEP loses mates.** Geometry rules cover shaft/bore fits, bearings, servos (horn vs body), fasteners, gears, press fits and folded poses (incidental contacts are cut with a minimum cut). Dense 1,000-part robots still need the mates route.
- **Onshape Enterprise documents** (e.g. `yourteam.onshape.com`) need a key issued by an Enterprise admin; a personal key gets 403. See [`docs/ONSHAPE_API_KEYS.md`](docs/ONSHAPE_API_KEYS.md).
- **Onshape conventions honoured:**
  - if any mate is named `dof_*`, top-level instances are the links, only `dof_*` mates are joints, fastened mates join, and other mates are ignored (`_inv` flips an axis);
  - otherwise every mate is read as-is;
  - instance-pattern copies stay rigid with their seed, composite parts are included, and `frame_*` markers are skipped.
- **Joint limits and real masses** aren't in STEP geometry. Set them in the spec.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full survey of existing tools, AI/MCP options, and how each simulator consumes URDF/SRDF and dynamics.
