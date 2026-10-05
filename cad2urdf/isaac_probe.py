"""Load a robot in Isaac Sim, drive it and report. No cad2urdf imports, so it runs in any Python with Isaac Sim.

    isaac_probe.py ROBOT.urdf '{"joint": target, ...}' [--floating] [--gui]
    isaac_probe.py ROBOT.usda '{"joint": target, ...}' --usd [--expect EXPECT.json]
    isaac_probe.py ROBOT.usda '{}' --usd --trajectory SPEC.json

A URDF goes through Isaac Sim's URDF importer (as Isaac Lab's UrdfFileCfg does). With ``--usd`` the asset is
referenced into the stage the way Isaac Lab spawns it, the per-joint values Isaac applied (armature, gains,
limits, efforts, masses) are read back and compared with ``--expect`` (a JSON of what the file says), and a
high-acceleration stress is run: that is where a missing armature shows up as an explosion.

``--trajectory`` runs the sim2sim scenario instead: the drive stiffness is zeroed and the spec's joint torques are
applied as efforts every physics step, and the joint trajectory is returned.

Prints one ``ISAAC_RESULT {json}`` line; ``--gui`` keeps the window open instead. Supports Isaac Sim 6.x and 5.x
(5.x for URDF only).
"""

import json
import os
import sys

import numpy as np
from isaacsim import SimulationApp

# multi_gpu off: the RTX renderer can crash probing both GPUs of a hybrid laptop
GUI = "--gui" in sys.argv
app = SimulationApp({"headless": not GUI, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0})

import omni.kit.commands  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
app.update()

asset, targets = sys.argv[1], json.loads(sys.argv[2])
floating = "--floating" in sys.argv
USD = "--usd" in sys.argv
expect = json.load(open(sys.argv[sys.argv.index("--expect") + 1])) if "--expect" in sys.argv else {}
STEPS = 720  # 3 s at 240 Hz
STRESS_STEPS, STRESS_PERIOD = 480, 12  # 2 s of +-targets flipping every 50 ms: a bang-bang acceleration test
out = {"mode": "usd" if USD else "urdf"}

def _np(x):
    x = x.numpy() if hasattr(x, "numpy") else x
    return np.asarray(x, dtype=float).reshape(-1)


def _arrays(x):
    """A getter's result as a list of float arrays: Isaac returns an array or a tuple of arrays."""
    items = x if isinstance(x, (tuple, list)) else [x]
    return [np.asarray(i.numpy() if hasattr(i, "numpy") else i, dtype=float) for i in items]


def _read(art, getters):
    for g in getters:
        if hasattr(art, g):
            return _arrays(getattr(art, g)())[0]
    return None


def _damping(art):
    """Per-DOF drive damping: ``get_dof_gains()`` returns (stiffness, damping) in this Isaac Sim release."""
    if hasattr(art, "get_dof_gains"):
        arrays = _arrays(art.get_dof_gains())
        if len(arrays) == 2:
            return arrays[1].reshape(-1)
    return _read(art, ("get_dof_dampings",))


def _limits(art, n):
    """(lower, upper) per DOF: the array is (1, n, 2) or (2, n) depending on the release, so the layout is detected."""
    if not hasattr(art, "get_dof_limits"):
        return None
    arrays = _arrays(art.get_dof_limits())
    if len(arrays) == 2 and arrays[0].size == n:  # a (lower, upper) tuple
        return arrays[0].reshape(-1), arrays[1].reshape(-1)
    a = arrays[0].reshape(-1)
    if a.size != 2 * n:
        return None
    rows = a.reshape(2, n)
    if np.all(rows[0] <= rows[1]):  # [all lowers, all uppers]
        return rows[0], rows[1]
    pairs = a.reshape(n, 2)
    return pairs[:, 0], pairs[:, 1]


def readback(art, names):
    """{joint: {armature, stiffness, damping, effort, velocity, friction, lower, upper}} as Isaac holds them."""
    n = len(names)
    got, missing = {name: {} for name in names}, []
    for field, getters in (("armature", ("get_dof_armatures",)), ("effort", ("get_dof_max_efforts",)),
                           ("velocity", ("get_dof_max_velocities",))):
        arr = _read(art, getters)
        if arr is None:
            missing.append(field)
            continue
        for i, name in enumerate(names):
            got[name][field] = float(arr.reshape(-1)[i])
    if hasattr(art, "get_dof_gains"):
        gains = _arrays(art.get_dof_gains())
        for field, arr in zip(("stiffness", "damping"), gains):
            for i, name in enumerate(names):
                got[name][field] = float(arr.reshape(-1)[i])
    else:
        missing += ["stiffness", "damping"]
    if hasattr(art, "get_dof_friction_properties"):
        props = _arrays(art.get_dof_friction_properties())
        # Isaac's static/dynamic/viscous friction model: a different attribute from physxJoint:jointFriction, so it
        # is reported, not compared
        out["friction_properties"] = [[round(float(v), 6) for v in p.reshape(-1)[:n]] for p in props]
    lim = _limits(art, n)
    if lim is None:
        missing.append("limits")
    else:
        for i, name in enumerate(names):
            got[name]["lower"], got[name]["upper"] = float(lim[0][i]), float(lim[1][i])
    out["api_missing"] = missing
    out["api_get"] = sorted(m for m in dir(art) if m.startswith("get_dof"))[:40]
    return got


def compare(got, want):
    """Mismatches between what Isaac applied and what the file says. Gains may be per radian or per degree."""
    bad, units = [], {}
    for joint, w in want.items():
        g = got.get(joint)
        if g is None:
            bad.append({"joint": joint, "field": "joint", "expected": "present", "got": "missing"})
            continue
        for field, exp in w.items():
            val = g.get(field)
            if val is None or not np.isfinite(val):
                continue
            tol = 1e-3 * max(abs(exp), 1.0) if field in ("lower", "upper") else 2e-3 * abs(exp) + 1e-6
            ok = abs(val - exp) <= tol
            if not ok and field in ("stiffness", "damping") and exp:
                for unit, factor in (("per_deg", np.pi / 180), ("per_rad", 180 / np.pi)):
                    if abs(val - exp * factor) <= 2e-3 * abs(exp * factor) + 1e-6:
                        units[field] = unit
                        ok = True
            if not ok:
                bad.append({"joint": joint, "field": field, "expected": round(exp, 6), "got": round(val, 6)})
    out["gain_units_vs_file"] = units
    return bad


def stress(art, names, q_target, SimulationManager):
    """Bang-bang targets at 80% of each joint's range: finite state and bounded speed, or it blew up."""
    lo = hi = None
    lim = _limits(art, len(names))
    if lim is not None:
        lo, hi = lim
    amp = np.full(len(names), 0.6)
    if lo is not None:
        span = np.where(np.isfinite(hi - lo) & ((hi - lo) < 50), 0.4 * (hi - lo), 0.6)
        amp = np.minimum(span, 1.2)
    max_v, finite = 0.0, True
    for k in range(STRESS_STEPS):
        sign = 1.0 if (k // STRESS_PERIOD) % 2 == 0 else -1.0
        art.set_dof_position_targets(np.array([sign * amp], dtype=np.float32))
        SimulationManager.step()
        app.update()
        qd = _np(art.get_dof_velocities())
        finite = finite and bool(np.all(np.isfinite(qd)))
        if finite:
            max_v = max(max_v, float(np.max(np.abs(qd))))
    out["stress"] = {"finite": finite, "max_joint_speed": round(max_v, 2), "exploded": (not finite) or max_v > 500.0}


def trajectory(art, names, spec, SimulationManager):
    """The sim2sim scenario: explicit torques tau = clip(kp (target - q) - kv qdot, effort), joints matched by name."""
    n = len(names)
    idx = [names.index(j) for j in spec["names"]]
    kv = np.array(spec["kv"])
    passive = np.zeros(n)
    got = _damping(art)
    if got is not None:
        passive[idx] = np.maximum(got[idx] - kv, 0.0)   # the USD's drive damping is kv plus passive damping
    art.set_dof_gains(np.zeros((1, n), dtype=np.float32), passive.reshape(1, n).astype(np.float32))
    kp, effort = np.array(spec["kp"]), np.array(spec["effort"])
    out_q = []
    for k in range(spec["steps"]):
        q = _np(art.get_dof_positions())[idx]
        qd = _np(art.get_dof_velocities())[idx]
        tgt = np.array(spec["targets"][min(k // spec["control_every"], len(spec["targets"]) - 1)])
        tau = np.zeros(n)
        tau[idx] = np.clip(kp * (tgt - q) - kv * qd, -effort, effort)
        art.set_dof_efforts(tau.reshape(1, n).astype(np.float32))
        SimulationManager.step()  # one physics step. No app.update() here: with the timeline playing it would advance
        #                           physics by a whole rendering frame and shift the clock against the other engines
        if k % spec["control_every"] == 0:
            out_q.append(_np(art.get_dof_positions())[idx].tolist())
    out["trajectory"] = out_q


def diagnose(art, names, spec, SimulationManager):
    """A constant 1 N m on the first spec joint with the drive off: what the engine does with a plain push."""
    import omni.timeline

    n = len(names)
    j = names.index(spec["names"][0])
    kv = np.array(spec["kv"])
    got = _damping(art)
    passive = np.zeros(n)
    if got is not None:
        passive[j] = max(float(got[j]) - kv[0], 0.0)
    art.set_dof_gains(np.zeros((1, n), dtype=np.float32), passive.reshape(1, n).astype(np.float32))
    info = {"joint": names[j], "passive_damping_set": float(passive[j])}
    for field, getter in (("armature", "get_dof_armatures"), ("damping_after", "get_dof_dampings"),
                          ("stiffness_after", "get_dof_stiffnesses"), ("max_velocity", "get_dof_max_velocities")):
        arr = _read(art, (getter,))
        info[field] = None if arr is None else float(arr.reshape(-1)[j])
    if hasattr(art, "get_dof_gains"):
        info["gains_after"] = [[round(float(v), 6) for v in a.reshape(-1)] for a in _arrays(art.get_dof_gains())]
    tl = omni.timeline.get_timeline_interface()
    t0 = tl.get_current_time()
    trace = []
    for k in range(300):
        tau = np.zeros(n, dtype=np.float32)
        tau[j] = 1.0
        art.set_dof_efforts(tau.reshape(1, n))
        SimulationManager.step()
        if k % 25 == 24:
            trace.append(round(float(_np(art.get_dof_positions())[j]), 6))
    info["q_trace_every_25_steps"] = trace
    release = []  # then no torque at all: damping alone would coast to a stop, a spring would pull the joint back
    for k in range(300):
        art.set_dof_efforts(np.zeros((1, n), dtype=np.float32))
        SimulationManager.step()
        if k % 25 == 24:
            release.append(round(float(_np(art.get_dof_positions())[j]), 6))
    info["q_release_every_25_steps"] = release
    info["sim_time_after_300_steps"] = round(float(tl.get_current_time() - t0), 6)
    info["physics_dt"] = float(SimulationManager.get_physics_dt()) if hasattr(SimulationManager, "get_physics_dt") else None
    out["diagnose"] = info


def run_isaac6():
    import omni.timeline
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, UsdGeom, UsdPhysics

    if USD:
        usd = asset
    else:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        usd_dir = os.path.join(os.path.dirname(asset), "_isaac_usd")
        cfg = URDFImporterConfig(urdf_path=asset, usd_path=usd_dir, fix_base=not floating, merge_fixed_joints=False,
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
    # the ground goes just under the robot, as in the MJCF: a robot whose CAD origin is mid-body would otherwise
    # start inside a ground at z = 0 and be thrown out of it
    from pxr import Usd

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                       UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
    lowest = float(cache.ComputeWorldBound(stage.GetPrimAtPath("/World/robot")).ComputeAlignedRange().GetMin()[2])
    plane.CreateWidthAttr(20.0)
    plane.CreateLengthAttr(20.0)
    UsdGeom.Xformable(plane).AddTranslateOp().Set(Gf.Vec3d(0, 0, min(0.0, lowest) - 0.01))
    out["ground_z"] = round(min(0.0, lowest) - 0.01, 4)
    # a Physics variant set (Isaac Sim 6.x importer output, and our USD) may have no default selection
    vsets = stage.GetPrimAtPath("/World/robot").GetVariantSets()
    if vsets.HasVariantSet("Physics"):
        vs = vsets.GetVariantSet("Physics")
        if not vs.GetVariantSelection():
            vs.SetVariantSelection("physx")
        out["physics_variant"] = vs.GetVariantSelection()
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not roots:
        raise RuntimeError("no articulation root in the USD")
    out["articulation_roots"] = len(roots)
    for flag, attr, value in (("--no-joint-friction", "physxJoint:jointFriction", 0.0),
                              ("--no-velocity-cap", "physxJoint:maxJointVelocity", 1.0e6)):
        if flag in sys.argv:  # diagnostics: neutralise one PhysX attribute on every joint
            for prim in stage.Traverse():
                a = prim.GetAttribute(attr) if prim.HasAttribute(attr) else None
                if a:
                    a.Set(value)
            out[flag.lstrip("-").replace("-", "_")] = True
    if "--no-self-collision" in sys.argv:  # a diagnostic: is a mismatch caused by the links colliding with each other?
        attr = stage.GetPrimAtPath(roots[0]).GetAttribute("physxArticulation:enabledSelfCollisions")
        if attr:
            attr.Set(False)
        out["self_collision"] = "off"
    dt = json.load(open(sys.argv[sys.argv.index("--trajectory") + 1]))["dt"] if "--trajectory" in sys.argv else 1 / 240
    SimulationManager.set_physics_dt(dt)  # before play: it cannot change while the simulation runs
    omni.timeline.get_timeline_interface().play()
    app.update()
    art = Articulation(roots[0])
    names = list(art.dof_names)
    out["dof_names"] = names
    if USD:
        got = readback(art, names)
        if expect:
            out["mismatches"] = compare(got, expect.get("joints", {}))
            out["mismatch_count"] = len(out["mismatches"])
            if hasattr(art, "get_link_masses"):
                total = float(np.sum(_np(art.get_link_masses())))
                out["mass_total"] = round(total, 5)
                out["mass_expected"] = expect.get("total_mass")
    if "--diagnose" in sys.argv:
        diagnose(art, names, json.load(open(sys.argv[sys.argv.index("--trajectory") + 1])), SimulationManager)
        return names, np.zeros(len(names)), _np(art.get_dof_positions())
    if "--trajectory" in sys.argv:
        trajectory(art, names, json.load(open(sys.argv[sys.argv.index("--trajectory") + 1])), SimulationManager)
        return names, np.zeros(len(names)), _np(art.get_dof_positions())
    q_target = np.array([[targets.get(n, 0.0) for n in names]], dtype=np.float32)
    art.set_dof_position_targets(q_target)
    if GUI:  # physics steps with the timeline until the window closes
        print(f"Isaac Sim: {len(names)} joints {names}; close the window to exit", flush=True)
        while app.is_running():
            app.update()
        return names, q_target[0], _np(art.get_dof_positions())
    for _ in range(STEPS):
        SimulationManager.step()
        app.update()
    q = _np(art.get_dof_positions())
    if USD:
        stress(art, names, q_target, SimulationManager)
    return names, q_target[0], q


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
    ok, prim = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=asset, import_config=cfg,
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
    if USD and not new_api:
        raise RuntimeError("--usd needs Isaac Sim 6.x")
    names, q_target, q = run_isaac6() if new_api else run_isaac5()
    driven = [i for i, n in enumerate(names) if n in targets]
    out["tracking_detail"] = {names[i]: [round(float(q_target[i]), 4), round(float(q[i]), 4)] for i in driven}
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
