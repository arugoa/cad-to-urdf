# Sample: `arm4`, a 4-DOF arm with a parallel gripper

This is a worked example going from CAD to simulation files. Research context is in [`docs/RESEARCH.md` §8](../../docs/RESEARCH.md#8-worked-sample-arm4).

## 1. The CAD

`build_cad.py` models **24 named solids** parametrically with build123d (OpenCascade) in millimetres, posed at the zero configuration (arm straight up). It writes them as one STEP assembly: `cad/arm4.step`.

The STEP file keeps names, placements and exact B-rep, but **no mates**, just like STEP exports from Onshape, SolidWorks, Fusion or Creo. Every moving interface is a real shaft in a real bore with 0.2 mm diametral clearance. That geometric signature is what joint inference uses.

| Rigid group | Parts |
|---|---|
| base | `base_plate`, `base_housing` (bore Ø29.8), `base_bolt_1..4` |
| turret | `turret_disc` (with Ø29.6 shaft), `turret_clevis_left/right` (Ø12.2 holes), `shoulder_motor` |
| upper arm | `upper_arm_beam` (slotted, Ø10.2 elbow bore), `shoulder_pin` (Ø12), `elbow_motor` |
| forearm | `forearm_plate_left/right`, `forearm_spacer` (Ø16.2 wrist bore), `elbow_pin` (Ø10) |
| gripper base | `wrist_flange` (Ø16 shaft), `gripper_palm`, `gripper_rail` (Ø8), `gripper_rail_post_left/right` |
| fingers | `finger_left`, `finger_right` (carriage with Ø8.2 bore on the rail + blade) |

## 2. The granularity spec

`robot_spec.yaml` is the only hand-written (or LLM-written) input. It sets:

- **Materials** and part→material patterns, which drive mass and inertia.
- **Links** as part-name patterns.
- **Joints**: type, limits, effort, velocity, sign, mimic. Axis and origin come from geometry.
- **Dynamics**: damping, friction, armature.
- **Actuators**: position servos with kp/kv.
- **Contact** friction.
- **Collision mode per link**, chosen from the collision study.
- **SRDF semantics**: groups, named states, end effector, passive joints.

## 3. Run

```bash
python -m cad2urdf examples/arm4/robot_spec.yaml -o examples/arm4/output --study
python -m cad2urdf.validate examples/arm4/output
```

## 4. What comes out (`output/`)

| File | For | Notes |
|---|---|---|
| `arm4.urdf` | MoveIt, ManiSkill/SAPIEN, PyBullet, MuJoCo fallback, yourdfpy | relative mesh paths; full inertia tensors; `dynamics`, `limit`, `mimic` |
| `arm4.srdf` | MoveIt, SAPIEN, mplib | groups `arm`/`gripper`, states `home`/`ready`/`open`/`closed`, end effector, passive `finger_right`, sampled `disable_collisions` (6 Adjacent + 12 Never) |
| `mjcf/arm4.xml` | MuJoCo / MJX / MuJoCo Warp / Newton | armature, frictionloss, position actuators, `equality` for the mimic finger, `contact/exclude`, keyframes, `tcp` site |
| `gazebo/arm4.gazebo.urdf` + `config/controllers.yaml` | Gazebo (gz-sim) + gz_ros2_control | `package://arm4_description/...` paths, `world` link, `<gazebo>` surface tags, `<ros2_control>` |
| `isaaclab/arm4_cfg.py` | Isaac Lab | `UrdfFileCfg` import options + `ImplicitActuatorCfg` per joint (armature, friction, limits) |
| `maniskill/arm4_agent.py` | ManiSkill 3 | `BaseAgent` with PD arm controller + mimic gripper controller, keyframes, materials |
| `meshes/visual/*.stl` | all | one mesh per link per material |
| `meshes/collision/*.stl` | all | every file is a single convex piece with ≤ 64 vertices |
| `report.json` | you | joint candidates, masses, inertia checks, collision metrics, collision study, SRDF sampling stats |
| `validation.json` | you | what each simulator actually loaded (see below) |

## 5. Results

**Joints recovered from geometry:** all 6. The rail interfaces come back as `cylindrical` (could slide or spin), and the spec resolves them to prismatic.

**Collision** (IoU vs exact CAD volume): base 0.97, turret 0.81, upper arm 0.76, forearm 0.97, gripper base 0.85, fingers 0.98. That is 43 collision geoms in total, versus 10.5k visual triangles.

**Validation:**

| Target | Outcome |
|---|---|
| MuJoCo (MJCF) | no penetration at home; stable; tracks `ready` within 0.013 rad; mimic fingers symmetric |
| MuJoCo (URDF) | loads; visuals dropped by default, no actuators, armature 0, mimic → equality |
| PyBullet | needs `URDF_USE_INERTIA_FROM_FILE` (60% inertia error otherwise); mimic via gear constraint |
| SAPIEN / ManiSkill | loads headless without visuals; reads SRDF but only applies `reason="Default"` pairs |
| yourdfpy | loads; FK matches CAD |

![collision modes](../../docs/img/collision_modes.png)
