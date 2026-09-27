"""Isaac Sim check, run with the Isaac Sim Python (.venv-isaac), not the main venv.

    .venv-isaac/bin/python cad2urdf/isaac_probe.py ROBOT.urdf '{"joint": target, ...}' [--floating]

Imports the URDF with Isaac Sim's own URDF importer (the same one Isaac Lab's UrdfFileCfg uses),
steps PhysX with position drives towards the targets, and prints one JSON line with what happened.
Kept free of cad2urdf imports so it runs inside Isaac's environment.
"""

import json
import sys

import numpy as np
from isaacsim import SimulationApp

# multi_gpu off: on hybrid laptops (AMD iGPU + NVIDIA) Isaac's RTX renderer can crash probing both GPUs
app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0})

import omni.kit.commands  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
app.update()

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402

urdf, targets = sys.argv[1], json.loads(sys.argv[2])
floating = "--floating" in sys.argv
out = {}
try:
    ok, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.fix_base = not floating
    cfg.make_default_prim = True
    cfg.create_physics_scene = True
    cfg.import_inertia_tensor = True
    cfg.convex_decomp = False  # our collision pieces are already convex
    cfg.default_drive_type = 1  # position drive
    cfg.default_drive_strength = 1e4
    cfg.default_position_drive_damping = 1e3
    ok, prim = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf, import_config=cfg,
                                         get_articulation_root=True)
    out["imported"] = bool(ok and prim)
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 240)
    world.scene.add_default_ground_plane()
    art = world.scene.add(SingleArticulation(prim_path=prim, name="robot"))
    world.reset()
    names = list(art.dof_names)
    out["dofs"] = len(names)
    q_target = np.array([targets.get(n, 0.0) for n in names], dtype=np.float32)
    from isaacsim.core.utils.types import ArticulationAction

    for _ in range(720):  # 3 s
        art.apply_action(ArticulationAction(joint_positions=q_target))
        world.step(render=False)
    q = np.asarray(art.get_joint_positions(), dtype=float)
    driven = [i for i, n in enumerate(names) if n in targets]
    out.update(
        joints_checked=len(driven),
        tracking_err_max=round(float(np.max(np.abs(q[driven] - q_target[driven]))) if driven else 0.0, 4),
        finite=bool(np.all(np.isfinite(q))),
        root_z=round(float(art.get_world_pose()[0][2]), 4),
    )
    out["ok"] = True
except Exception as e:  # noqa: BLE001
    out.update(ok=False, error=f"{type(e).__name__}: {e}")
print("ISAAC_RESULT " + json.dumps(out), flush=True)
app.close()
