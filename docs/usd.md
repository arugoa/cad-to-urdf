# USD asset

`usd/<robot>.usda` uses the Isaac Sim 6.x asset structure, as Isaac Lab loads it ([Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/asset_structure.html)):

```
usd/
  <robot>.usda               interface layer: references configuration/base.usda, variant set "Physics"
  configuration/
    base.usda                link hierarchy, one Xform per link
    instances.usda           visual meshes and collision shapes per link
    geometries.usda          mesh data
    materials.usda           materials
    robot.usda               robot metadata
    Physics/
      physics.usda           engine-neutral UsdPhysics
      physx.usda             physics.usda plus PhysX attributes
      mujoco.usda            physics.usda plus MuJoCo and Newton attributes
```

All files are text USD.

## Physics variants

| Variant | Holds |
|---|---|
| `physx` (default) | neutral physics plus `physxJoint:armature`, `physxJoint:jointFriction`, `physxJoint:maxJointVelocity`, `PhysxMimicJointAPI`, self collision |
| `mujoco` | neutral physics plus `mjc:armature`, `mjc:frictionloss`, `newton:armature`, `NewtonMimicAPI` |
| `physics` | neutral UsdPhysics only |
| `none` | geometry only |

The neutral layer has rigid bodies, mass and principal inertia, colliders (convex hulls and primitives, with a physics material using the MJCF friction), joints and drives, the articulation root, a fixed root joint for a fixed base, filtered collision pairs from the SRDF, and loop closures as spherical joints excluded from the articulation.

## Units

UsdPhysics uses degrees: revolute limits are in degrees and angular drive gains are per degree. The writer converts from the spec's radians. Prismatic joints are in meters.

## Isaac Lab and Newton

```python
sim_utils.UsdFileCfg(usd_path="usd/robot.usda", variants={"Physics": "physx"})    # PhysX
sim_utils.UsdFileCfg(usd_path="usd/robot.usda", variants={"Physics": "mujoco"})   # Newton, MJWarp
```

The generated `isaaclab/<robot>_cfg.py` uses the first. Isaac Lab actuator configs override the USD's drive values at spawn, so the config carries the same stiffness, damping and armature as the USD.

## Armature

Each engine reads armature from its own namespace and ignores the others. Newton's default USD import reads `newton:*` only, so an asset with only `physxJoint:armature` has zero armature in Newton and goes unstable at high accelerations. The writer sets armature in all three namespaces, and `python -m cad2urdf.asset_test --sims newton,isaac` reads back what each engine applied. Armature is at least `16 kp dt^2`, which keeps an explicit position spring stable at a 2 ms step.

Newton does not enforce `physxJoint:maxJointVelocity`; limit speed with damping.
