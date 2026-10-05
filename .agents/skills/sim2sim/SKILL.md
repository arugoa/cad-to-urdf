---
name: sim2sim
description: Check that a generated robot asset behaves the same across simulators (MuJoCo, PyBullet, Newton, Isaac Sim) by running one scripted scenario in each and comparing the trajectories. Use when an asset works in one simulator but not another, before moving a policy between simulators (for example PhysX to Newton), or to find which parameter differs between engines.
---

# sim2sim

`python -m cad2urdf.sim2sim` runs the same scenario in several simulators and writes `sim2sim.json`. You read the differences, find the parameter behind each one, fix the spec (or report a code bug) and re-run. The asset must already pass the cad2sim skill's asset test.

Run from the repo root with ROS kept out of Python: `env -u PYTHONPATH .venv/bin/python -m ...`.

## 1. Run

```bash
python -m cad2urdf.sim2sim build/<robot>                              # MuJoCo (reference) and PyBullet
python -m cad2urdf.sim2sim build/<robot> --sims mujoco,pybullet,newton
python -m cad2urdf.sim2sim build/<robot> --sims mujoco,newton,isaac   # starts Isaac Sim
```

Isaac Sim needs about 9 GB of free RAM and about a minute. Ask the user before every launch, run it alone, and never beside another heavy job. Without permission, say the Isaac comparison was not run.

## 2. What is compared

Every simulator loads the same asset (MJCF, URDF or USD), steps at 2 ms, and applies the same joint torques: `tau = clip(kp (target - q) - kd qdot, effort)`, from the MJCF's gains. The torques come from the script, not from each engine's drive, so the comparison tests the asset's dynamics: inertia, armature, damping, friction, limits. Joints are matched by name, never by index, because engines order a branched robot's joints differently.

The scenario is a hold, a step to half range, a bang-bang square wave (the high-acceleration case) and a return. The reference is the baseline (MuJoCo by default); each other simulator is scored per joint as RMSE over the joint's range, with the worst joint and the step and bang-bang phases reported separately. Under 5% is `info`, over is a `warning`; a trajectory that goes non-finite is an `error`.

## 3. Expected differences

| Simulator | Expect a gap from |
|---|---|
| PyBullet | no armature (a URDF can't carry it) and no mimic joints: this shows in the bang-bang phase and on mimic followers |
| Newton | solver differences; it reads armature only from the `mujoco` Physics variant, so a `physx`-only asset has zero armature there |
| Isaac Sim (PhysX) | a different solver and contact model; similar, not identical, trajectories. PhysX drives mimic followers as well as leaders, Newton drives one joint |

A gap larger than these explain is a finding.

## 4. Find the cause

| Signature | Likely cause | Action |
|---|---|---|
| one engine blows up in the bang-bang phase only | armature missing or zero in that engine | run the asset test with that engine; the writers must set armature in its namespace |
| every engine agrees except in the step phase | a limit differs, or one engine ignores it | compare limits across the files with the asset test |
| a joint is far off in one engine, all others agree | wrong axis sign or a unit error in that engine's file | compare world axes (asset test); report as a writer bug |
| slow drift in one engine | friction or damping not applied | compare `damping` and `friction` per file |
| a follower joint is off | mimic handled differently | expected for PyBullet; check `NewtonMimicAPI` or `PhysxMimicJointAPI` for the others |
| the trajectories match but a contact scene differs | friction models differ between engines | do not copy friction numbers across engines; check collision shapes and `condim` first |

Do not change the code or hand-edit files. Fix gains, armature, damping, effort or collision in the spec, regenerate, and re-run both this and the asset test.

## 5. Checks before a policy transfer

The simulators must agree on the contract around the policy: ordered joint names, action scale, offsets and clipping, PD gains, physics dt and decimation, and the observation layout. If a locomotion policy falls within a few dozen steps in the second engine, suspect joint order first. Expect similar behaviour, not identical trajectories.

## 6. Running the engines in parallel

MuJoCo, PyBullet and Newton are light: run each in its own subagent and compare the JSONs. Isaac runs alone.

## 7. Report

Give the user: which simulators ran, the per-simulator RMSE and the worst joint, each gap you explained and its cause, each one you could not explain, and anything not run (Isaac, Newton).
