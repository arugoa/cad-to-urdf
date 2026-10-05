"""Run one scripted scenario in several simulators and compare the trajectories (sim-to-sim transfer of an asset).

    python -m cad2urdf.sim2sim build/robot [--sims mujoco,pybullet,newton,isaac] [--ref mujoco] [--seconds 6]

Every simulator loads the same asset (MJCF, URDF or USD), steps at the same dt and applies the same joint torques,
computed here from the MJCF's gains: tau = clip(kp (q_target - q) - kd qdot, +-effort). Driving the torques
explicitly tests what the asset says about the *dynamics* (inertia, armature, damping, friction, limits), not how
each engine implements a position drive. Contacts are off in every simulator, so a difference comes from the asset's dynamics and not from two collision
pipelines (contact transfer is not tested here). Joints are matched by name, never by index: engines order the joints
of a branched robot differently.

The scenario: hold, a step to half range, a bang-bang square wave (the high-acceleration case where a missing
armature shows), and a return. The reference simulator's trajectory is the baseline; every other one is scored
against it per joint. Writes ``sim2sim.json`` and the raw ``sim2sim_trajectories.npz`` next to the asset. ``isaac`` launches Isaac Sim (ask first, ~9 GB).

Known, expected differences are reported as info: PyBullet has no armature and ignores mimic joints; Newton's
default USD import reads only the ``mujoco`` Physics variant's armature (``physx`` gives it zero).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .asset_test import read_mjcf, read_urdf

DT = 0.002  # physics step for every simulator
CONTROL_EVERY = 5  # steps between target updates (10 ms)
HOLD, STEP, BANG, RETURN = 1.0, 2.0, 2.0, 1.0  # seconds per phase
BANG_PERIOD = 0.2
RMSE_WARN = 0.05  # of the joint's range (or 1 rad)


class Contract:
    """What every simulator must agree on: ordered joint names, gains, limits, and the scripted targets."""

    def __init__(self, asset: Path):
        self.asset = asset
        self.urdf_path = next(p for p in sorted(asset.glob("*.urdf")) if not p.name.startswith("_"))
        self.urdf = read_urdf(self.urdf_path)
        self.mjcf_path = next(iter(sorted((asset / "mjcf").glob("*.xml"))))
        self.mjcf = read_mjcf(self.mjcf_path)
        self.usda = asset / "usd" / f"{self.urdf_path.stem}.usda"
        j = self.mjcf["joints"]
        self.names = [n for n in j if j[n]["kp"] > 0 and not self.urdf["joints"].get(n, {}).get("mimic")]
        self.kp = np.array([j[n]["kp"] for n in self.names])
        self.kv = np.array([j[n]["kv"] for n in self.names])
        self.effort = np.array([j[n]["effort"] or 1e6 for n in self.names])
        lo = np.array([j[n]["lower"] for n in self.names])
        hi = np.array([j[n]["upper"] for n in self.names])
        self.limited = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 50)
        self.lo, self.hi = np.where(self.limited, lo, -1.0), np.where(self.limited, hi, 1.0)
        self.scale = np.where(self.limited, self.hi - self.lo, 1.0)  # what "5% of range" means
        self.steps = int(round((HOLD + STEP + BANG + RETURN) / DT))

    def target(self, t: float) -> np.ndarray:
        edge = np.where(np.abs(self.hi) >= np.abs(self.lo), self.hi, self.lo)
        half = np.where(self.limited, 0.5 * edge, 0.5)  # an unlimited joint steps half a radian
        if t < HOLD:
            return np.zeros(len(self.names))
        if t < HOLD + STEP:
            return half
        if t < HOLD + STEP + BANG:
            amp = 0.8 * np.where(self.limited, np.minimum(np.abs(self.lo), np.abs(self.hi)), 0.6)
            amp = np.where(amp > 1e-6, amp, 0.4 * np.abs(half))
            return amp * (1.0 if int((t - HOLD - STEP) / BANG_PERIOD) % 2 == 0 else -1.0)
        return np.zeros(len(self.names))

    def torque(self, q: np.ndarray, qd: np.ndarray, t: float, target: np.ndarray | None = None) -> np.ndarray:
        tgt = self.target(t) if target is None else target
        return np.clip(self.kp * (tgt - q) - self.kv * qd, -self.effort, self.effort)


def _result(q, qd, dt, **extra) -> dict:
    q = np.asarray(q)
    return {"q": q, "dt": dt, "finite": bool(np.all(np.isfinite(q))), **extra}


# ---------- runners: each returns {q: (T, n) per control step, finite, notes} ----------

def run_mujoco(c: Contract) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(c.mjcf_path))
    m.opt.timestep = DT
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT   # joint dynamics only: see the module docstring
    m.actuator_gainprm[:, 0] = 0.0  # torques come from here, not from the MJCF actuators
    m.actuator_biasprm[:, :3] = 0.0
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in c.names]
    dadr = [m.jnt_dofadr[m.joint(n).id] for n in c.names]
    mujoco.mj_forward(m, d)
    mass = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, mass)
    bare = np.array([mass[i, i] - m.dof_armature[i] for i in dadr])   # reflected inertia without the armature
    out = []
    for k in range(c.steps):
        t = k * DT
        d.qfrc_applied[dadr] = c.torque(d.qpos[qadr], d.qvel[dadr], t)
        mujoco.mj_step(m, d)
        if k % CONTROL_EVERY == 0:
            out.append(d.qpos[qadr].copy())
        if not np.all(np.isfinite(d.qpos)):
            out.extend([np.full(len(qadr), np.nan)] * ((c.steps - k) // CONTROL_EVERY))
            break
    return _result(out, None, DT * CONTROL_EVERY, notes=["armature and joint damping from the MJCF"],
                   inertia_without_armature=bare.tolist())


def run_pybullet(c: Contract) -> dict:
    import pybullet as p

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(DT, physicsClientId=cid)
        rid = p.loadURDF(str(c.urdf_path), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)
        index = {p.getJointInfo(rid, i, physicsClientId=cid)[1].decode(): i for i in range(p.getNumJoints(rid, physicsClientId=cid))}
        idx = [index[n] for n in c.names]
        p.setJointMotorControlArray(rid, idx, p.VELOCITY_CONTROL, forces=[0.0] * len(idx), physicsClientId=cid)
        out = []
        for k in range(c.steps):
            st = p.getJointStates(rid, idx, physicsClientId=cid)
            q, qd = np.array([s[0] for s in st]), np.array([s[1] for s in st])
            p.setJointMotorControlArray(rid, idx, p.TORQUE_CONTROL, forces=c.torque(q, qd, k * DT).tolist(), physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)
            if k % CONTROL_EVERY == 0:
                out.append(q)
        notes = ["no armature (URDF cannot carry it)", "mimic joints are ignored"]
        order = [p.getJointInfo(rid, i, physicsClientId=cid)[1].decode() for i in range(p.getNumJoints(rid, physicsClientId=cid))]
        return _result(out, None, DT * CONTROL_EVERY, notes=notes, native_order=order)
    finally:
        p.disconnect(cid)


def run_newton(c: Contract, variant: str = "mujoco") -> dict:
    """Newton (MuJoCo-Warp solver) on the USD's ``mujoco`` Physics variant, torques through ``control.joint_f``."""
    import newton
    import warp as wp
    from pxr import Usd

    if not c.usda.exists():
        return {"error": "no USD in the asset folder"}
    stage = Usd.Stage.Open(str(c.usda))
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection(variant)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(stage)
    device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    model = builder.finalize(device=device)
    n_dof = model.joint_dof_count
    keys = [str(k).split("/")[-1] for k in model.joint_label] if hasattr(model, "joint_label") else list(model.joint_key)
    starts = model.joint_qd_start.numpy()
    dof_of = {}
    for ji, key in enumerate(keys):
        if key in c.names:
            dof_of[key] = int(starts[ji])
    missing = [n for n in c.names if n not in dof_of]
    if missing:
        return {"error": f"joints not found in Newton's model by name: {missing[:4]} (has {keys[:6]})"}
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    ke[:] = 0.0  # no native drive: the torques below are the only actuation
    for i, name in enumerate(c.names):  # the USD's drive damping is kv plus the passive joint damping: keep the latter
        d = dof_of[name]
        kd[d] = max(float(kd[d]) - c.kv[i], 0.0)
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True, integrator="implicitfast")
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    dofs = [dof_of[n] for n in c.names]
    qadr = [int(model.joint_q_start.numpy()[keys.index(n)]) for n in c.names]
    out = []
    for k in range(c.steps):
        q = s0.joint_q.numpy()[qadr]
        qd = s0.joint_qd.numpy()[dofs]
        f = np.zeros(n_dof, dtype=np.float32)
        f[dofs] = c.torque(q, qd, k * DT)
        control.joint_f.assign(f)
        s0.clear_forces()
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0
        if k % CONTROL_EVERY == 0:
            out.append(s0.joint_q.numpy()[qadr].copy())
    notes = [f"Physics variant {variant}", f"device {device}", "armature read from newton:armature"]
    return _result(out, None, DT * CONTROL_EVERY, notes=notes, native_order=keys)


def run_isaac(c: Contract) -> dict:
    """Isaac Sim (PhysX) on the USD, torques through the articulation efforts. Launches Isaac Sim."""
    import os
    import subprocess

    from .validate import isaac_python, meminfo_gb

    py = isaac_python()
    if py is None:
        return {"error": "Isaac Sim is not installed"}
    free = meminfo_gb("MemAvailable")
    need = float(os.environ.get("CAD2URDF_ISAAC_MIN_FREE_GB", "10"))
    if free < need:
        return {"error": f"Isaac Sim needs ~9 GB; {free:.1f} GB free (< {need:g}): close apps first"}
    spec = {"names": c.names, "kp": c.kp.tolist(), "kv": c.kv.tolist(), "effort": c.effort.tolist(), "dt": DT,
            "control_every": CONTROL_EVERY, "steps": c.steps, "targets": [c.target(k * DT).tolist() for k in range(0, c.steps, CONTROL_EVERY)]}
    spec_path = c.asset / "usd" / "_sim2sim_spec.json"
    spec_path.write_text(json.dumps(spec))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "DISPLAY", "WAYLAND_DISPLAY")}
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if icd.exists():
        env["VK_ICD_FILENAMES"] = str(icd)
    probe = Path(__file__).with_name("isaac_probe.py")
    r = subprocess.run([str(py), "-P", str(probe), str(c.usda), "{}", "--usd", "--trajectory", str(spec_path)],
                       capture_output=True, text=True, env=env, timeout=1800)
    line = next((x for x in r.stdout.splitlines() if x.startswith("ISAAC_RESULT ")), None)
    if line is None:
        return {"error": f"Isaac Sim {'crashed' if 'Crash detected' in r.stdout + r.stderr else 'failed'} (exit {r.returncode})"}
    res = json.loads(line[len("ISAAC_RESULT "):])
    if not res.get("ok") or "trajectory" not in res:
        return {"error": res.get("error", "Isaac Sim returned no trajectory")}
    return _result(res["trajectory"], None, DT * CONTROL_EVERY, notes=["PhysX via the USD physx variant"],
                   native_order=res.get("dof_names"))


RUNNERS = {"mujoco": run_mujoco, "pybullet": run_pybullet, "newton": run_newton, "isaac": run_isaac}
EXPECTED = {
    "pybullet": "PyBullet has no armature and ignores mimic joints, so a gap here is expected for those",
    "newton": "Newton reads the mujoco variant's armature; solver settings are not numerically portable between engines",
    "isaac": "PhysX applies the physx variant; expect similar behaviour, not identical trajectories (on arm4 it settles a few percent short of the target; joint friction and the velocity cap were ruled out, the cause is not found)",
}


def compare(ref: np.ndarray, other: np.ndarray, c: Contract) -> dict:
    n = min(len(ref), len(other))
    err = other[:n] - ref[:n]
    per = np.sqrt(np.nanmean(err**2, axis=0))
    phase = {"step": slice(int(HOLD / (DT * CONTROL_EVERY)), int((HOLD + STEP) / (DT * CONTROL_EVERY))),
             "bang_bang": slice(int((HOLD + STEP) / (DT * CONTROL_EVERY)), int((HOLD + STEP + BANG) / (DT * CONTROL_EVERY)))}
    by_phase = {k: float(np.nanmax(np.sqrt(np.nanmean(err[s] ** 2, axis=0)) / c.scale)) for k, s in phase.items()}
    return {"rmse_per_joint": {nm: round(float(v), 5) for nm, v in zip(c.names, per)},
            "rmse_over_range_max": round(float(np.nanmax(per / c.scale)), 4),
            "worst_joint": c.names[int(np.nanargmax(per / c.scale))],
            "phase_rmse_over_range": {k: round(v, 4) for k, v in by_phase.items()},
            "final_error_max": round(float(np.nanmax(np.abs(err[-1]))), 5)}


def unstable_without_armature(c: Contract, bare) -> list[str]:
    """Joints where explicit damping is unstable at this step without armature: it needs kv * dt / I < 2."""
    return [n for n, i, kv in zip(c.names, bare, c.kv) if i > 0 and kv * DT / i >= 2.0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cad2urdf.sim2sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="a robot output folder")
    ap.add_argument("--sims", default="mujoco,pybullet", help="mujoco, pybullet, newton, isaac (launches Isaac Sim)")
    ap.add_argument("--ref", default="mujoco", help="the baseline simulator (default mujoco)")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    c = Contract(out)
    sims = args.sims.split(",")
    if args.ref not in sims:
        sims.insert(0, args.ref)
    result = {"robot": c.urdf_path.stem, "joints": c.names, "dt": DT, "reference": args.ref, "sims": {}, "flags": []}
    runs = {}
    for name in sims:
        t0 = time.time()
        try:
            res = RUNNERS[name](c)
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        res["seconds"] = round(time.time() - t0, 1)
        runs[name] = res
        entry = {k: v for k, v in res.items() if k not in ("q",)}
        result["sims"][name] = entry
        if "error" in res:
            result["flags"].append({"severity": "error", "sim": name, "what": res["error"]})
        elif not res["finite"]:
            result["flags"].append({"severity": "error", "sim": name, "what": "the trajectory went non-finite (exploded)"})
    ref = runs.get(args.ref)
    if ref and "q" in ref:
        rq = np.asarray(ref["q"])
        for name, res in runs.items():
            if name == args.ref or "q" not in res:
                continue
            cmp_ = compare(rq, np.asarray(res["q"]), c)
            result["sims"][name]["vs_reference"] = cmp_
            bad = cmp_["rmse_over_range_max"] > RMSE_WARN
            sev = "warning" if bad else "info"
            what = (f"RMSE {cmp_['rmse_over_range_max'] * 100:.1f}% of range (worst joint {cmp_['worst_joint']}; "
                    f"step {cmp_['phase_rmse_over_range']['step'] * 100:.1f}%, bang-bang {cmp_['phase_rmse_over_range']['bang_bang'] * 100:.1f}%)")
            if bad and name in EXPECTED:
                what += f". {EXPECTED[name]}"
            bare = runs.get(args.ref, {}).get("inertia_without_armature")
            if name == "pybullet" and bare is not None:
                risky = unstable_without_armature(c, bare)
                if risky:
                    what += (f". Without armature, explicit damping is unstable (kv*dt/I >= 2) on {len(risky)} of "
                             f"{len(c.names)} joints, e.g. {risky[0]}: this gap is the armature, not a bug")
                    cmp_["unstable_without_armature"] = risky
            result["flags"].append({"severity": sev, "sim": name, "what": what})
    saved = {name: np.asarray(res["q"]) for name, res in runs.items() if "q" in res}
    saved["targets"] = np.array([c.target(k * DT * CONTROL_EVERY) for k in range(len(next(iter(saved.values()), [])))])
    np.savez_compressed(out / "sim2sim_trajectories.npz", joints=np.array(c.names), **saved)
    errors = sum(f["severity"] == "error" for f in result["flags"])
    result["ok"] = errors == 0
    (out / "sim2sim.json").write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{result['robot']}: reference {args.ref}, {len(sims)} simulator(s), {errors} error(s)  ->  {out / 'sim2sim.json'}")
    for f in result["flags"]:
        print(f"  [{f['severity']}] {f['sim']}: {f['what']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
