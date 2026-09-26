# cad-to-urdf

Turn feature-based / parametric CAD assemblies into **simulation-ready robot descriptions**: URDF + SRDF, native MJCF, and the side files that Isaac Lab, ManiSkill and Gazebo need. You set granularity per link (which parts form a link, which interfaces are joints, how detailed each link's collision geometry is) in a small YAML spec.

- 📄 **Research notes:** [`docs/RESEARCH.md`](docs/RESEARCH.md) covers:
  - existing CAD→URDF/SRDF tools (Onshape, SolidWorks, Fusion, Creo, STEP)
  - Claude / GPT-6 Astra and MCP capabilities
  - how to identify joints in each CAD package
  - collision meshes for multi-part assemblies
  - how dynamics reach each simulator
  - how URDF/SRDF needs differ across MuJoCo, Isaac Lab, ManiSkill, Gazebo, PyBullet, Drake and Genesis
  - a proposed architecture
- 🤖 **Worked sample:** [`examples/arm4/`](examples/arm4/) goes from a parametric 4-DOF arm + gripper to a STEP file, then to URDF/SRDF/MJCF/Gazebo/Isaac Lab/ManiSkill files, validated in MuJoCo, PyBullet and SAPIEN.

![arm4](docs/img/arm4_visual_vs_collision.png)

## Convert your robot (3 layers: CAD → format → simulator)

```bash
python -m cad2urdf.route --list                                          # the routing matrix
python -m cad2urdf.route --cad onshape --format native --sim maniskill   # print the plan for a route
python -m cad2urdf.route --cad onshape --format step --sim maniskill --run \
    --input my_robot.step --out build/my_robot [--spec overrides.yaml]   # run it (deterministic)
python -m cad2urdf.route --cad solidworks --format native --sim mujoco --run \
    --input exported/robot.urdf --out build/robot                         # finish an exporter's URDF
```

The conversion is deterministic. The judgment calls (joint limits, materials and mimic couplings for STEP input, and running exporters inside the CAD tool) are handled by the **`cad2sim` skill** in `.agents/skills/cad2sim/`, which is symlinked into `.claude/skills/` so both Claude Code and Codex find it. See [`docs/RESEARCH.md` §9](docs/RESEARCH.md#9-routing-cad--format--simulator).

If ROS is sourced in your shell, run with `env -u PYTHONPATH`: ROS's pytest plugins and packages leak into the venv.

## Quick start (sample)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python examples/arm4/build_cad.py                                   # (re)build the sample STEP
python -m cad2urdf examples/arm4/robot_spec.yaml -o examples/arm4/output --study   # ~4 min (CoACD)
python -m cad2urdf.validate examples/arm4/output                    # load in MuJoCo / PyBullet / SAPIEN / yourdfpy
python examples/arm4/render.py                                      # figures in docs/img
pytest -q                                                           # fast tests (no CoACD)
```

## What the pipeline does

```
STEP (+ robot_spec.yaml)
  → parts (exact B-rep, names)          cad2urdf/cad.py
  → links (spec patterns), mass/COM/inertia from B-rep × density, validity checks
  → joint candidates from shaft/bore geometry; spec resolves type/limits      cad2urdf/joints.py, model.py
  → collision per link: none | box | primitives | hull | decompose(CoACD) | mesh, scored by IoU    collision.py
  → URDF (neutral + Gazebo/ros2_control)   urdf.py
  → SRDF with a sampled self-collision matrix (MoveIt-style, MuJoCo as checker)   srdf.py
  → MJCF (armature, actuators, mimic→equality, excludes, keyframes)   mjcf.py
  → Isaac Lab ArticulationCfg, ManiSkill agent, gz_ros2_control YAML   targets.py
  → report.json; validation.json (validate.py)
```

## Status

A research prototype. It uses the STEP adapter only. Native Onshape/Fusion/SolidWorks/Creo adapters are sketched in the research doc (§4). Isaac Lab, ManiSkill-agent and Gazebo outputs are generated but were not executed here, because this environment has no GPU or ROS.
