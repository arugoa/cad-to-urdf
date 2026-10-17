# Robot Model CI

`robot-model-ci` is an artifact-level quality gate for CAD2Sim output folders. It deliberately complements the existing `cad2urdf.validate`, `asset_test`, and `sim2sim` commands rather than replacing them. It checks common structural and packaging defects in URDF/MJCF and compares generated output to a previously known-good artifact tree.

## Install

From the `cad-to-urdf` repository root:

```bash
uv pip install -e tools/robot-model-ci
robot-model-ci validate build/robot --json build/robot/model_ci_report.json
```

Run it after the normal CAD2Sim pipeline. CAD2Sim currently emits the URDF at the output root, MJCF under `mjcf/`, and meshes under `meshes/`, which this tool understands.

Compare a known-good model directory with a candidate:

```bash
robot-model-ci compare artifacts/known-good build/robot --json build/robot/model_ci_diff.json
```

The comparison intentionally treats removed links/joints and changed joint topology/type as breaking. Axis, limit, inertial, and mesh changes are reported for review instead of automatically blocked because some are expected CAD edits. Candidate validation errors fail the command.

## Checks in v0.1

- XML parsing, root element checks, duplicate link/joint/body names
- URDF parent/child references, graph roots/connectivity, joint axes and finite origin values
- movable-joint limits, ordered ranges, positive effort/velocity values
- positive mass (a zero-mass frame link is allowed behind a fixed joint) and positive-semidefinite URDF inertia tensor checks
- local mesh/include references and warnings for machine-specific absolute paths
- MJCF joint range and actuator inventory checks
- cross-format movable-joint inventory/type warnings across URDF and MJCF with the same model name
- baseline/candidate inventory diff and SHA-256 mesh content checks

This is not a full physics proof or a replacement for loading the model in MuJoCo, Newton, Isaac Sim, Gazebo, or PyBullet. Keep `cad2urdf.validate`, `asset_test`, and `sim2sim` in the pipeline for simulator-specific checks.

## Local tests

```bash
python -m pip install pytest
python -m pytest tools/robot-model-ci/tests
```

No runtime dependencies are required; pytest is needed only to run the tests.
