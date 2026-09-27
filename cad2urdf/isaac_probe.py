"""Isaac Sim check, run with the Isaac Sim Python (.venv-isaac6 / .venv-isaac), not the main venv.

    .venv-isaac6/bin/python cad2urdf/isaac_probe.py ROBOT.urdf '{"joint": target, ...}' [--floating]

Imports the URDF with Isaac Sim's own URDF importer (the one Isaac Lab's UrdfFileCfg uses), steps PhysX
with position drives towards the targets, and prints one ``ISAAC_RESULT {json}`` line.
Supports Isaac Sim 6.x (URDFImporter + isaacsim.core.experimental) and 5.x (commands + isaacsim.core.api).
Kept free of cad2urdf imports so it runs inside Isaac's environment.
"""

import json
import os
import sys

import numpy as np
from isaacsim import SimulationApp

# multi_gpu off: on hybrid laptops (AMD iGPU + NVIDIA) the RTX renderer can crash probing both GPUs
app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0})

import omni.kit.commands  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
app.update()

urdf, targets = sys.argv[1], json.loads(sys.argv[2])
floating = "--floating" in sys.argv
STEPS = 720  # 3 s at 240 Hz
out = {}


def _np(x):
    x = x.numpy() if hasattr(x, "numpy") else x
    return np.asarray(x, dtype=float).reshape(-1)


def run_isaac6():
    import omni.timeline
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, UsdGeom, UsdPhysics

    usd_dir = os.path.join(os.path.dirname(urdf), "_isaac_usd")
    cfg = URDFImporterConfig(urdf_path=urdf, usd_path=usd_dir, fix_base=not floating, merge_fixed_joints=False,
                             collision_from_visuals=False, collision_type="Convex Hull", allow_self_collision=False,
                             joint_drive_type="force", joint_target_type="position",
                             override_joint_stiffness=1e4, override_joint_damping=1e3)
    usd = URDFImporter(cfg).import_urdf()
    out["usd"] = os.path.basename(usd)
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    plane = UsdGeom.Plane.Define(stage, "/World/ground")
    plane.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    stage_utils.add_reference_to_stage(usd, "/World/robot")
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not roots:
        raise RuntimeError("no articulation root in the imported USD")
    SimulationManager.set_physics_dt(1 / 240)
    omni.timeline.get_timeline_interface().play()
    app.update()
    art = Articulation(roots[0])
    names = list(art.dof_names)
    q_target = np.array([[targets.get(n, 0.0) for n in names]], dtype=np.float32)
    art.set_dof_position_targets(q_target)
    for _ in range(STEPS):
        SimulationManager.step()
        app.update()
    return names, q_target[0], _np(art.get_dof_positions())


def run_isaac5():
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

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
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 240)
    world.scene.add_default_ground_plane()
    art = world.scene.add(SingleArticulation(prim_path=prim, name="robot"))
    world.reset()
    names = list(art.dof_names)
    q_target = np.array([targets.get(n, 0.0) for n in names], dtype=np.float32)
    for _ in range(STEPS):
        art.apply_action(ArticulationAction(joint_positions=q_target))
        world.step(render=False)
    return names, q_target, _np(art.get_joint_positions())


try:
    try:
        import isaacsim.asset.importer.urdf as _imp

        new_api = hasattr(_imp, "URDFImporter")
    except ImportError:
        new_api = False
    out["isaac_api"] = "6.x" if new_api else "5.x"
    names, q_target, q = run_isaac6() if new_api else run_isaac5()
    driven = [i for i, n in enumerate(names) if n in targets]
    out.update(
        dofs=len(names),
        joints_checked=len(driven),
        tracking_err_max=round(float(np.max(np.abs(q[driven] - q_target[driven]))) if driven else 0.0, 4),
        finite=bool(np.all(np.isfinite(q))),
        ok=True,
    )
except Exception as e:  # noqa: BLE001
    import traceback

    out.update(ok=False, error=f"{type(e).__name__}: {e}", where=traceback.format_exc().splitlines()[-3][:200])
print("ISAAC_RESULT " + json.dumps(out), flush=True)
app.close()
