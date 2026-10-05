# Simulators

| Simulator | Load |
|---|---|
| MuJoCo | `mjcf/<robot>.xml` |
| Isaac Lab, Isaac Sim | `isaaclab/<robot>_cfg.py` (spawns `usd/<robot>.usda`, variant `physx`; a second config uses the URDF importer) |
| Newton (MJWarp) | `usd/<robot>.usda`, variant `mujoco` |
| ManiSkill | `maniskill/<robot>_agent.py` |
| Gazebo | `gazebo/<robot>.gazebo.urdf`, `gazebo/config/controllers.yaml` |
| PyBullet | `<robot>.urdf` with `URDF_USE_INERTIA_FROM_FILE` |

The USD asset is described in [usd.md](usd.md).

## Validation

```bash
python -m cad2urdf.validate build/robot --sims mujoco,pybullet,sapien,maniskill,gazebo
python -m cad2urdf.validate build/robot --sims isaac      # starts Isaac Sim, needs ~9 GB free
```

Each simulator loads the asset, drives a test pose, and reports tracking error, penetrations at rest, mimic error and stability in `validation.json`. Isaac Sim runs only when named. `isaac` loads the USD, reads back the values Isaac applied (armature, gains, limits, efforts, masses) and runs a high-acceleration stress; `isaac_urdf` uses the URDF importer instead. Gazebo needs `ros-$ROS_DISTRO-gz-ros2-control`. More thorough checks are in [testing.md](testing.md).

## Viewing

```bash
python tests/view_urdf.py build/robot                    # browser: joint sliders, collision toggle
python tests/view_urdf.py build/robot --sim mujoco       # MuJoCo viewer, keys 2/3 toggle visual/collision
python tests/view_urdf.py build/robot --sim maniskill    # SAPIEN viewer
python tests/view_urdf.py build/robot --sim isaac        # Isaac Sim window, close other apps first
```

## Per-simulator notes

- **Loop closures** (Orbita, the Haro pump) are `connect` equalities in the MJCF and spherical joints outside the articulation in the USD. A URDF can't hold them.
- **Mimic joints:** MJCF `equality`, PhysX `PhysxMimicJointAPI`, Newton `NewtonMimicAPI`. PyBullet ignores `<mimic>`; the validator uses gear constraints. PhysX drives leader and follower, Newton drives one.
- **Armature** has no URDF field. It is written to the MJCF, the USD and the Isaac Lab config, at least `16 kp dt^2` so the position spring is stable at the timestep.
- **Joint order** differs between engines on branched robots. Match by name.
- **Memory:** Isaac Sim needs about 9 GB free. See [development.md](development.md) for the memory cap.
