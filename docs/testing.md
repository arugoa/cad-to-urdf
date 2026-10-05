# Testing and status

## Asset tests

```bash
python -m cad2urdf.asset_test build/robot --sims mujoco,newton    # add isaac to start Isaac Sim (~9 GB)
```

Checks mass, inertia and limits; compares the URDF, MJCF and each USD variant (armature, gains, limits, efforts, masses, joint axes); then drives the asset in MuJoCo with a hold, step responses, a bang-bang acceleration test (a missing armature makes it explode here) and a random sweep. `newton` adds what Newton's USD import reads. `isaac` reads back what Isaac Sim applied and compares it with the file. Flags are `error`, `warning` or `info`, each with a hint, and go to `asset_test.json`. The [`cad2sim` skill](../.agents/skills/cad2sim/SKILL.md) reads them and fixes the spec.

## Repository tests

```bash
pytest -q
```

Covers joint inference, inertia, collision fitting, the SRDF matrix, the STEP draft, the URDF round trip, `root_rpy`, loop closures, part classes, the Onshape front end against a fake API (planar, origin and screw mates), the USD asset (structure, joint frames, units, armature per engine, Newton's read-back), and the asset tester catching deliberately broken assets.

## Status

Tested on an RTX 3070 Ti laptop with Ubuntu 22.04.

| Input | Result |
|---|---|
| 7 public Onshape robots | joint counts match the references: 2-wheeler 2, adjustable arm 4, quadruped 12, dog 12, Sigmaban 20, Orbita 7 (+2 loop closures), RSK soccer 64 |
| Open Duck Mini v2 (Onshape) | 14 joints; runs in MuJoCo and PyBullet |
| SO-100 arm (STEP) | drafted automatically: 7 links, 6 joints |
| Haro380 arm, printed parallel gripper, Ender 3 V2 (STEP, tested locally, not bundled) | Haro380: hand-written spec with 6 joints and the gas-spring loop; gripper: 5 links, 4 pivots, four-bar loops reported; Ender 3: no joints (slides aren't detected) |
| Kaya base (STEP) | one solid, no joints |
| ARC 3v3 field (surface-only STEP) | 925 collision shapes; 0 of 400 drop-test balls fall through |
| Triton Infantry 2026 (Onshape, 101 parts) | 62 links, 61 joints: wheels, yaw, pitch and flywheels from the real mates, plus bearing and collar joints for the agent to fix; the STEP draft of the same robot is wrong |
| 1,000-part robots from STEP | drafts are unreliable; use the Onshape route |

Simulators checked: MuJoCo (URDF and MJCF), PyBullet, SAPIEN, ManiSkill 3 (CPU and GPU), yourdfpy, Gazebo Classic 11 (headless), Newton 1.5 (USD import and armature read-back), Isaac Sim 6.1 (USD loaded, read back and stress-tested).

## Limitations

- STEP has no mates. Shaft/bore fits, loose pins, bearings, servo horns, gears and press fits are recognised; joints the CAD doesn't model come out rigid.
- Joint limits and real masses aren't in STEP geometry. Set them in the spec.
- Loop closures reach the MJCF and the USD. URDF-based simulators need them added by hand.
- Which joints are real mechanisms, and which parts are fasteners or bearings, are not decided in the code. The agent skill decides.

[RESEARCH.md](RESEARCH.md) surveys existing tools and how each simulator reads URDF, SRDF and dynamics.
