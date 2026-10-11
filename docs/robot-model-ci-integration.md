# Applying Robot Model CI to `arugoa/cad-to-urdf`

This directory is a drop-in overlay for your existing repository. It does not modify the converter, its generated formats, or the current `cad2urdf.validate`, `asset_test`, and `sim2sim` tools. The public repository's output layout already puts URDF at the output root, MJCF under `mjcf/`, and mesh assets under `meshes/`, which Robot Model CI can inspect.

## 1. Apply the overlay

From a local checkout of `https://github.com/arugoa/cad-to-urdf`:

```bash
bash /path/to/cad2sim-model-ci-overlay/apply_overlay.sh "$PWD"
python -m pip install -e tools/robot-model-ci
python -m pytest tools/robot-model-ci/tests -q
```

If the bundle directory is already the current working directory, use its absolute path in the command above.

## 2. Run it after CAD2Sim

Keep the existing pipeline, then add:

```bash
python -m cad2urdf.route --cad solidworks --format step --sim mujoco --run \
  --input robot.step --out build/robot \
  --part-classes .agents/skills/cad2sim/part_classes.yaml

robot-model-ci validate build/robot --json build/robot/model-ci-report.json
python -m cad2urdf.validate build/robot --sims mujoco,pybullet
python -m cad2urdf.asset_test build/robot --sims mujoco,newton
python -m cad2urdf.sim2sim build/robot --sims mujoco,newton
```

The CAD2Sim command is just the repository's documented example; use the route, input, and simulator combination that applies to your robot. Keep the existing simulator tests. The new checker adds fast structure/asset linting and a model inventory diff; it does not replace simulation.

## 3. Add it to CI

The supplied workflow validates the checker itself and a small known-good fixture on every push/PR. It checks `build/robot` if an earlier workflow step has generated that directory. For genuine end-to-end CI, add your real model-generation command before the `Validate generated CAD2Sim model` step. Large licensed CAD files, Onshape credentials, Isaac Sim installation, and heavy simulation dependencies vary by project, so the overlay does not guess those choices.

## 4. Baseline comparison

Keep a known-good generated output directory as a versioned or CI artifact baseline, then compare it with the candidate:

```bash
robot-model-ci compare artifacts/model-baseline build/robot --json build/model-diff.json
```

Removed links/joints and changed joint type/topology fail the comparison. Changes to axes, limits, inertia, and mesh bytes are reported for review rather than automatically blocked; expected CAD changes can legitimately affect them. Candidate validation errors always fail.

## Current limits

- Local URDF/MJCF structure and asset checks; not a full URDF parser or physics proof.
- Mesh URI resolution is best effort for package layouts and custom URI schemes.
- Use existing simulator validation and sim-to-sim tests to catch issues that need the actual physics engine.
