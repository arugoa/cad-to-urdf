"""Load the generated files in every simulator available locally and check what each one reads.

    python -m cad2urdf.validate examples/arm4/output

Checks: MuJoCo (raw URDF, URDF + <mujoco> block, native MJCF), PyBullet,
SAPIEN (the ManiSkill backend) and yourdfpy (a ROS-free URDF parser).
Isaac Sim/Lab and Gazebo need GPUs / ROS installs and are not exercised here.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _urdf_info(urdf: Path):
    root = ET.parse(urdf).getroot()
    masses = {l.get("name"): float(l.find("inertial/mass").get("value")) for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}
    return root, masses, joints


def check_yourdfpy(urdf: Path) -> dict:
    import yourdfpy

    r = yourdfpy.URDF.load(str(urdf), build_scene_graph=True, load_meshes=True, load_collision_meshes=True)
    cfg = {n: 0.0 for n in r.actuated_joint_names}
    r.update_cfg(cfg)
    T = r.get_transform("gripper_base", "base_link")
    return {"ok": True, "links": len(r.link_map), "actuated_joints": r.actuated_joint_names,
            "gripper_base_z_at_home": round(float(T[2, 3]), 4)}


def check_mujoco_urdf(urdf: Path) -> dict:
    """What MuJoCo's URDF importer does with a plain, neutral URDF."""
    import mujoco

    out = {}
    try:
        mujoco.MjModel.from_xml_path(str(urdf))
        out["raw_urdf"] = "loaded"
    except Exception as e:  # expected: meshes not found because strippath drops 'meshes/visual/'
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

    m_default = with_block(strippath="false")
    m_keep = with_block(strippath="false", discardvisual="false", fusestatic="false")
    _, masses, _ = _urdf_info(urdf)
    out["with_strippath_false"] = {
        "ngeom": m_default.ngeom, "nbody": m_default.nbody, "neq (mimic->equality?)": m_default.neq,
        "nu (actuators)": m_default.nu,
        "dof_damping": np.round(m_default.dof_damping, 4).tolist(),
        "dof_frictionloss": np.round(m_default.dof_frictionloss, 4).tolist(),
        "dof_armature": np.round(m_default.dof_armature, 4).tolist(),
    }
    out["with_discardvisual_false"] = {"ngeom": m_keep.ngeom}
    # a fixed-base root link is fused into the world body (fusestatic), so compare moving links only
    moving = sum(v for k, v in masses.items() if k != ET.parse(urdf).getroot().find("link").get("name"))
    out["moving_mass_match"] = bool(np.isclose(sum(m_default.body_mass[1:]), moving, rtol=1e-4))
    out["root_link_fused_into_world"] = bool(m_default.nbody == len(masses))
    gz = urdf.parent / "gazebo" / f"{urdf.stem}.gazebo.urdf"
    if gz.exists():
        try:
            mujoco.MjModel.from_xml_path(str(gz))
            out["package_uri_urdf"] = "loaded"
        except Exception as e:
            out["package_uri_urdf"] = f"FAILED: {str(e).splitlines()[0][:120]}"
    return out


def check_mujoco_mjcf(xml: Path) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    names = lambda obj, n: [mujoco.mj_id2name(m, obj, i) for i in range(n)]
    bodies = names(mujoco.mjtObj.mjOBJ_BODY, m.nbody)

    # 1. initial penetration at home between non-excluded bodies
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    pen = sorted({tuple(sorted((bodies[m.geom_bodyid[c.geom1]], bodies[m.geom_bodyid[c.geom2]])))
                  for c in d.contact[: d.ncon] if c.dist < -1e-4})

    # 2. track the "ready" keyframe with the position servos for 3 s
    k_ready = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
    target = m.key_qpos[k_ready].copy()
    d.ctrl[:] = m.key_ctrl[k_ready]
    finger_l = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_left")]
    finger_r = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_right")]
    ok = True
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            ok = False
            break
    err = np.abs(d.qpos - target)
    # 3. open the gripper and check the equality constraint keeps the fingers symmetric
    i_f = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)].index("finger_left")
    d.ctrl[i_f] = 0.008
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return {
        "nbody": m.nbody, "ngeom": m.ngeom, "nu": m.nu, "neq": m.neq, "nkey": m.nkey,
        "penetrating_pairs_at_home": pen,
        "stable": ok,
        "ready_tracking_err_rad_max": round(float(err[:4].max()), 4),
        "gravity_sag_note": "position servos with finite kp settle with a small steady-state error under gravity",
        "fingers_after_open": [round(float(d.qpos[finger_l]), 5), round(float(d.qpos[finger_r]), 5)],
    }


def check_pybullet(urdf: Path) -> dict:
    import pybullet as p

    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    root = ET.parse(urdf).getroot()
    res = {}
    for label, flags in [("default_flags", 0), ("USE_INERTIA_FROM_FILE", p.URDF_USE_INERTIA_FROM_FILE)]:
        rid = p.loadURDF(str(urdf), useFixedBase=True, flags=flags)
        link_names = {p.getJointInfo(rid, i)[12].decode(): i for i in range(p.getNumJoints(rid))}
        izz = {n: p.getDynamicsInfo(rid, i)[2] for n, i in link_names.items()}
        # compare principal moments (pybullet stores the diagonalised inertia)
        rel = [abs(max(izz[n]) - max(np.linalg.eigvalsh(_inertia(root, n)))) / max(np.linalg.eigvalsh(_inertia(root, n)))
               for n in link_names]
        res[label] = {"max_rel_inertia_error": round(float(max(rel)), 3)}
        p.removeBody(rid)

    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    rid = p.loadURDF(str(urdf), useFixedBase=True, flags=flags)
    n = p.getNumJoints(rid)
    info = [p.getJointInfo(rid, i) for i in range(n)]
    res["joint_damping_friction_read"] = {i[1].decode(): (round(i[6], 3), round(i[7], 3)) for i in info}
    p.performCollisionDetection()
    pairs = sorted({tuple(sorted((info[c[3]][12].decode() if c[3] >= 0 else "base_link",
                                  info[c[4]][12].decode() if c[4] >= 0 else "base_link")))
                    for c in p.getContactPoints(rid, rid) if c[8] < -1e-4})
    res["self_collision_pairs_at_home (parent pairs excluded by flag)"] = pairs
    # mimic is NOT enforced by pybullet: emulate with a gear constraint
    idx = {i[1].decode(): i[0] for i in info}
    c = p.createConstraint(rid, idx["finger_left"], rid, idx["finger_right"], p.JOINT_GEAR, [0, 1, 0], [0, 0, 0], [0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=100)
    targets = {"shoulder_pitch": 0.6, "elbow": 1.2, "finger_left": 0.008}
    for name, j in idx.items():
        if info[j][2] != p.JOINT_FIXED and name != "finger_right":
            p.setJointMotorControl2(rid, j, p.POSITION_CONTROL, targetPosition=targets.get(name, 0.0),
                                    force=float(info[j][10]))
    p.setJointMotorControl2(rid, idx["finger_right"], p.VELOCITY_CONTROL, force=0)  # free, follows the gear
    for _ in range(240 * 3):
        p.stepSimulation()
    q = {name: p.getJointState(rid, j)[0] for name, j in idx.items()}
    res["after_3s"] = {k: round(v, 4) for k, v in q.items()}
    p.disconnect(cid)
    return res


def _inertia(root, link):
    el = root.find(f"link[@name='{link}']/inertial/inertia")
    g = lambda k: float(el.get(k))
    return np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]])


def check_sapien(urdf: Path) -> dict:
    import shutil

    import sapien

    # headless: SAPIEN needs a GPU render device for visual shapes, so load a visual-free copy
    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):
        for v in link.findall("visual"):
            link.remove(v)
    tmp = urdf.parent / "_sapien_probe.urdf"
    ET.ElementTree(root).write(tmp)
    shutil.copy(urdf.with_suffix(".srdf"), tmp.with_suffix(".srdf"))
    try:
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(1 / 500)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(tmp))  # also reads <same name>.srdf
    finally:
        tmp.unlink()
        tmp.with_suffix(".srdf").unlink()
    ignored = [sorted(p) for p in loader.ignore_pairs]
    joints = robot.get_active_joints()
    targets = {"shoulder_pitch": 0.6, "elbow": 1.2}
    for j in joints:
        j.set_drive_properties(stiffness=200.0, damping=10.0, force_limit=40.0)
        j.set_drive_target(targets.get(j.name, 0.0))
    for _ in range(1500):
        robot.set_qf(robot.compute_passive_force(gravity=True))
        scene.step()
    q = dict(zip([j.name for j in joints], np.round(robot.get_qpos(), 4).tolist()))
    total_mass = sum(l.mass for l in robot.get_links())
    return {"active_joints": [j.name for j in joints],
            "srdf_pairs_applied (only reason=Default is honoured)": ignored,
            "total_mass": round(float(total_mass), 4), "qpos_after_3s": q,
            "mimic_note": "finger_right is an independent active joint in SAPIEN; ManiSkill uses a mimic controller"}


def main(out_dir: str):
    out = Path(out_dir)
    urdf = next(out.glob("*.urdf"))
    results = {}
    for name, fn, arg in [
        ("yourdfpy", check_yourdfpy, urdf),
        ("mujoco_urdf_import", check_mujoco_urdf, urdf),
        ("mujoco_mjcf", check_mujoco_mjcf, out / "mjcf" / f"{urdf.stem}.xml"),
        ("pybullet", check_pybullet, urdf),
        ("sapien_maniskill", check_sapien, urdf),
    ]:
        try:
            results[name] = fn(arg)
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"== {name}\n{json.dumps(results[name], indent=2)}")
    (out / "validation.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/arm4/output")
