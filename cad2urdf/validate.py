"""Load the generated files in every simulator available locally and check what each one reads.

    python -m cad2urdf.validate OUTDIR [--sims mujoco,pybullet,sapien,maniskill,yourdfpy]

Robot-agnostic: every check drives the actuated joints to the same modest test
pose (from 0 by min(0.4 rad | 2 cm, 30 % of the range), 0.4 rad for continuous
joints, mimic joints following their leader) and reports tracking error, stability and mimic error.

Checks: MuJoCo (URDF import and native MJCF), PyBullet, SAPIEN, ManiSkill 3
(through the generated agent, GPU if available) and yourdfpy.
Isaac Sim/Lab and Gazebo are not exercised here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- URDF helpers
def _urdf_joints(urdf: Path) -> list[dict]:
    out = []
    for j in ET.parse(urdf).getroot().findall("joint"):
        lim, mim = j.find("limit"), j.find("mimic")
        out.append(dict(
            name=j.get("name"), type=j.get("type"),
            lower=float(lim.get("lower", 0)) if lim is not None else 0.0,
            upper=float(lim.get("upper", 0)) if lim is not None else 0.0,
            mimic=None if mim is None else (mim.get("joint"), float(mim.get("multiplier", 1)), float(mim.get("offset", 0))),
        ))
    return out


def test_pose(urdf: Path) -> dict[str, float]:
    """Target for every non-fixed joint (mimic followers included)."""
    q = {}
    joints = [j for j in _urdf_joints(urdf) if j["type"] != "fixed"]
    for j in joints:
        if j["mimic"] is None:
            if j["type"] == "continuous":
                q[j["name"]] = 0.4
                continue
            lo, hi = j["lower"], j["upper"]
            mid = 0.0 if lo <= 0.0 <= hi else (lo + hi) / 2
            step = min(0.4 if j["type"] == "revolute" else 0.02, 0.3 * (hi - lo))
            q[j["name"]] = float(np.clip(mid + step, lo, hi))
    for j in joints:
        if j["mimic"]:
            lead, mult, off = j["mimic"]
            q[j["name"]] = q.get(lead, 0.0) * mult + off
    return q


def _mimics(urdf: Path):
    return [(j["name"], *j["mimic"]) for j in _urdf_joints(urdf) if j["mimic"]]


def _mimic_err(q: dict[str, float], urdf: Path) -> float:
    errs = [abs(q[f] - (q[l] * m + o)) for f, l, m, o in _mimics(urdf) if f in q and l in q]
    return round(float(max(errs)), 5) if errs else 0.0


def _masses(urdf: Path) -> dict[str, float]:
    out = {}
    for l in ET.parse(urdf).getroot().findall("link"):
        m = l.find("inertial/mass")
        out[l.get("name")] = float(m.get("value")) if m is not None else 0.0
    return out


def _fixed_base(out_dir: Path, urdf: Path) -> bool:
    srdf = urdf.with_suffix(".srdf")
    if srdf.exists():
        vj = ET.parse(srdf).getroot().find("virtual_joint")
        if vj is not None:
            return vj.get("type") != "floating"
    return True


def _summary(q: dict[str, float], target: dict[str, float], urdf: Path) -> dict:
    leaders = [n for n in target if n not in {f for f, *_ in _mimics(urdf)}]
    missing = [n for n in leaders if n not in q]
    err = max((abs(q[n] - target[n]) for n in leaders if n in q), default=float("nan"))
    return {"joints_checked": len(leaders) - len(missing), **({"joints_missing": missing} if missing else {}),
            "tracking_err_max": round(float(err), 4), "mimic_err_max": _mimic_err(q, urdf),
            "finite": bool(np.all(np.isfinite(list(q.values()))))}


# ---------------------------------------------------------------- checks
def check_yourdfpy(urdf: Path) -> dict:
    import yourdfpy

    r = yourdfpy.URDF.load(str(urdf), build_scene_graph=True, load_meshes=True, load_collision_meshes=True)
    r.update_cfg({n: 0.0 for n in r.actuated_joint_names})
    T = [r.get_transform(l) for l in r.link_map]
    return {"ok": True, "links": len(r.link_map), "actuated_joints": len(r.actuated_joint_names),
            "fk_finite": bool(np.all(np.isfinite(T)))}


def check_mujoco_urdf(urdf: Path) -> dict:
    """What MuJoCo's own URDF importer does with the neutral URDF."""
    import mujoco

    out = {}
    try:
        mujoco.MjModel.from_xml_path(str(urdf))
        out["raw_urdf"] = "loaded"
    except Exception as e:
        out["raw_urdf"] = f"FAILED: {str(e).splitlines()[0][:160]}"

    def with_block(**compiler):
        root = ET.parse(urdf).getroot()
        mj = ET.Element("mujoco")
        ET.SubElement(mj, "compiler", {k: str(v) for k, v in compiler.items()})
        root.insert(0, mj)
        path = urdf.parent / "_mj_probe.urdf"
        ET.ElementTree(root).write(path)
        try:
            return mujoco.MjModel.from_xml_path(str(path))
        finally:
            path.unlink()

    try:
        m = with_block(strippath="false")
        m_keep = with_block(strippath="false", discardvisual="false", fusestatic="false")
        out["import"] = {"ngeom": m.ngeom, "ngeom_with_visuals": m_keep.ngeom, "nbody": m.nbody,
                         "neq (mimic->equality)": m.neq, "nu (actuators)": m.nu,
                         "armature_all_zero": bool(np.all(m.dof_armature == 0))}
    except Exception as e:
        out["import"] = f"FAILED: {str(e).splitlines()[0][:200]}"
    gz = urdf.parent / "gazebo" / f"{urdf.stem}.gazebo.urdf"
    if gz.exists():
        try:
            mujoco.MjModel.from_xml_path(str(gz))
            out["package_uri_urdf"] = "loaded"
        except Exception as e:
            out["package_uri_urdf"] = f"FAILED: {str(e).splitlines()[0][:120]}"
    return out


def check_mujoco_mjcf(xml: Path, urdf: Path) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    mujoco.mj_forward(m, d)
    pen = sorted({tuple(sorted((bodies[m.geom_bodyid[c.geom1]], bodies[m.geom_bodyid[c.geom2]])))
                  for c in d.contact[: d.ncon] if c.dist < -1e-4 and "world" not in
                  (bodies[m.geom_bodyid[c.geom1]], bodies[m.geom_bodyid[c.geom2]])})
    target = test_pose(urdf)
    act = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    for i, name in enumerate(act):
        jid = m.actuator_trnid[i, 0]
        d.ctrl[i] = target.get(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid), 0.0)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    q = {}
    for jid in range(m.njnt):
        if int(m.jnt_type[jid]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            q[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)] = float(d.qpos[m.jnt_qposadr[jid]])
    unactuated = sorted(set(target) - {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
                                       for i in range(m.nu)} - {f for f, *_ in _mimics(urdf)})
    tgt = {k: v for k, v in target.items() if k not in unactuated}
    return {"nbody": m.nbody, "ngeom": m.ngeom, "nu": m.nu, "neq": m.neq, "nkey": m.nkey,
            "penetrating_pairs_at_zero": pen, "unactuated_joints": unactuated, **_summary(q, tgt, urdf)}


def check_pybullet(urdf: Path, fixed: bool) -> dict:
    import pybullet as p

    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    res = {}
    rid = p.loadURDF(str(urdf), useFixedBase=fixed)
    n = p.getNumJoints(rid)
    def_I = [p.getDynamicsInfo(rid, i)[2] for i in range(n)]
    p.removeBody(rid)
    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    rid = p.loadURDF(str(urdf), useFixedBase=fixed, flags=flags)
    file_I = [p.getDynamicsInfo(rid, i)[2] for i in range(n)]
    rel = [abs(max(a) - max(b)) / max(max(b), 1e-12) for a, b in zip(def_I, file_I) if max(b) > 0]
    res["inertia_err_without_USE_INERTIA_FROM_FILE"] = round(float(max(rel, default=0)), 3)
    info = [p.getJointInfo(rid, i) for i in range(n)]
    idx = {i[1].decode(): i[0] for i in info}
    lname = {i[0]: i[12].decode() for i in info}
    p.performCollisionDetection()
    res["self_contacts_at_zero"] = sorted({tuple(sorted((lname.get(c[3], "base"), lname.get(c[4], "base"))))
                                           for c in p.getContactPoints(rid, rid) if c[8] < -1e-4})
    followers = set()
    for f, l, mult, off in _mimics(urdf):  # pybullet ignores <mimic>: emulate with gear constraints
        c = p.createConstraint(rid, idx[l], rid, idx[f], p.JOINT_GEAR, [0, 0, 1], [0, 0, 0], [0, 0, 0])
        p.changeConstraint(c, gearRatio=-mult, maxForce=1000)
        followers.add(f)
    target = test_pose(urdf)
    for name, j in idx.items():
        if info[j][2] == p.JOINT_FIXED:
            continue
        if name in followers:
            p.setJointMotorControl2(rid, j, p.VELOCITY_CONTROL, force=0)
        else:
            p.setJointMotorControl2(rid, j, p.POSITION_CONTROL, targetPosition=target.get(name, 0.0),
                                    force=float(info[j][10]) or 1000.0)
    for _ in range(240 * 3):
        p.stepSimulation()
    q = {name: p.getJointState(rid, j)[0] for name, j in idx.items() if info[j][2] != p.JOINT_FIXED}
    p.disconnect(cid)
    return {**res, **_summary(q, target, urdf)}


def check_sapien(urdf: Path, fixed: bool) -> dict:
    import sapien  # noqa: I001 (import before mujoco/pybullet in a fresh process is safest)

    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):  # headless CPU check: no render device needed without visuals
        for v in link.findall("visual"):
            link.remove(v)
    tmp = urdf.parent / "_sapien_probe.urdf"
    ET.ElementTree(root).write(tmp)
    if urdf.with_suffix(".srdf").exists():
        shutil.copy(urdf.with_suffix(".srdf"), tmp.with_suffix(".srdf"))
    try:
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(1 / 500)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = fixed
        if not any(j.get("type") != "fixed" for j in root.findall("joint")):
            # no moving joints: SAPIEN loads this as a plain rigid object, not an articulation
            arts, actors, _ = loader.parse(str(tmp))
            return {"ok": True, "note": "no moving joints: loaded as a rigid object",
                    "objects": len(arts) + len(actors)}
        robot = loader.load(str(tmp))
    finally:
        tmp.unlink()
        tmp.with_suffix(".srdf").unlink(missing_ok=True)
    target = test_pose(urdf)
    for j in robot.get_active_joints():
        j.set_drive_properties(stiffness=1000.0, damping=50.0, force_limit=1e4)
        j.set_drive_target(target.get(j.name, 0.0))
    for _ in range(1500):
        robot.set_qf(robot.compute_passive_force(gravity=True))
        scene.step()
    q = dict(zip([j.name for j in robot.get_active_joints()], robot.get_qpos().tolist()))
    s = _summary(q, target, urdf)
    s["mimic_err_max"] = "n/a (SAPIEN treats mimic joints as independent; ManiSkill uses a mimic controller)"
    return {"srdf_pairs_applied (reason=Default only)": [sorted(p) for p in loader.ignore_pairs],
            "total_mass": round(float(sum(l.mass for l in robot.get_links())), 4), **s}


MANISKILL_PROBE = r"""
import importlib.util, json, sys, torch, gymnasium as gym
spec = importlib.util.spec_from_file_location("agent", sys.argv[1]); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import mani_skill.envs, sapien
from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.utils.registration import register_env
target = json.loads(sys.argv[3])
backend = sys.argv[2]
lift = float(sys.argv[4])
n = 64 if backend == "physx_cuda" else 1
cls = next(v for v in vars(m).values() if isinstance(v, type) and hasattr(v, "uid") and v.__module__ == "agent")

@register_env("Cad2SimProbe-v1", max_episode_steps=10**9, override=True)
class Probe(EmptyEnv):  # Empty-v1 has a floor at z=0: lift the robot so it doesn't start inside it
    def _load_agent(self, options):
        super(EmptyEnv, self)._load_agent(options, sapien.Pose(p=[0, 0, lift]))

env = gym.make("Cad2SimProbe-v1", robot_uids=cls.uid, num_envs=n, sim_backend=backend, control_mode="pd_joint_pos")
env.reset(seed=0)
agent = env.unwrapped.agent
act = []
for name, c in agent.controller.controllers.items():
    mimic = getattr(c.config, "mimic", {}) or {}
    act += [target.get(j, 0.0) for j in c.config.joint_names if j not in mimic]
a = torch.tensor(act, dtype=torch.float32).repeat(n, 1)
for _ in range(150):
    env.step(a)
q = agent.robot.get_qpos()
names = [j.name for j in agent.robot.active_joints]
print(json.dumps({"backend": backend, "num_envs": n, "action_dim": len(act),
                  "q": dict(zip(names, q[0].cpu().tolist())),
                  "spread_across_envs": float((q.max(0).values - q.min(0).values).max())}))
"""


def ground_clearance(urdf: Path, margin: float = 0.01) -> float:
    """Height to lift a fixed-base robot so its collision geometry starts above a z=0 floor."""
    import yourdfpy

    r = yourdfpy.URDF.load(str(urdf), build_scene_graph=False, build_collision_scene_graph=True,
                           load_meshes=False, load_collision_meshes=True)
    b = r.collision_scene.bounds if r.collision_scene is not None and r.collision_scene.geometry else None
    return max(0.0, -float(b[0][2]) + margin) if b is not None else 0.0


def check_maniskill(out: Path, urdf: Path) -> dict:
    agent = next((out / "maniskill").glob("*_agent.py"), None)
    if agent is None:
        return {"ok": False, "error": "no maniskill/*_agent.py"}
    target = test_pose(urdf)
    if not target:
        return {"skipped": "no moving joints: nothing for a ManiSkill agent to control (load it as an actor)"}
    lift = ground_clearance(urdf)
    res = {"lifted_above_floor_m": round(lift, 4)}
    import os

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for backend in ("physx_cpu", "physx_cuda"):
        r = subprocess.run([sys.executable, "-c", MANISKILL_PROBE, str(agent), backend, json.dumps(target), str(lift)],
                           capture_output=True, text=True, env=env, timeout=900)
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode or not lines:
            res[backend] = {"ok": False, "error": (r.stderr.strip().splitlines() or ["?"])[-1][:300]}
            continue
        d = json.loads(lines[-1])
        res[backend] = {"num_envs": d["num_envs"], "action_dim": d["action_dim"],
                        "spread_across_envs": round(d["spread_across_envs"], 6), **_summary(d["q"], target, urdf)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="examples/arm4/output")
    ap.add_argument("--sims", default="yourdfpy,mujoco,pybullet,sapien,maniskill")
    args = ap.parse_args(argv)
    out = Path(args.out)
    urdf = next(p for p in out.glob("*.urdf") if not p.name.startswith("_"))
    fixed = _fixed_base(out, urdf)
    sims = args.sims.split(",")
    checks = []
    if "yourdfpy" in sims:
        checks.append(("yourdfpy", lambda: check_yourdfpy(urdf)))
    if "mujoco" in sims:
        checks += [("mujoco_urdf_import", lambda: check_mujoco_urdf(urdf)),
                   ("mujoco_mjcf", lambda: check_mujoco_mjcf(out / "mjcf" / f"{urdf.stem}.xml", urdf))]
    if "pybullet" in sims:
        checks.append(("pybullet", lambda: check_pybullet(urdf, fixed)))
    if "sapien" in sims:
        checks.append(("sapien", lambda: check_sapien(urdf, fixed)))
    if "maniskill" in sims:
        checks.append(("maniskill", lambda: check_maniskill(out, urdf)))
    results = {"test_pose": test_pose(urdf)}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"== {name}\n{json.dumps(results[name], indent=2)}")
    (out / "validation.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
