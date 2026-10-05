"""Stress-test a generated asset and cross-check its files: the checks that catch converter mistakes.

    python -m cad2urdf.asset_test build/robot [--sims mujoco,newton,isaac] [--seconds 3]

Static audit (mass, inertia, limits), consistency between the URDF, the MJCF and every USD variant (armature,
gains, limits, efforts, masses, joint axes), then dynamics in MuJoCo: a hold, step responses, a bang-bang
acceleration test (where a missing armature makes the asset explode) and a random sweep. ``newton`` adds what
Newton's USD import reads; ``isaac`` adds what Isaac Sim applied to the USD (it launches Isaac Sim).

Writes ``asset_test.json`` next to the asset and prints a summary. Every flag has a severity (error, warning,
info) and a hint; the exit code is 1 when there is an error. Nothing here edits the asset: fix the spec and re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEG = 180.0 / np.pi
DT = 0.002  # the MJCF timestep: the armature floor 16*kp*dt^2 is stated against it
BLOWUP_SPEED = 500.0  # rad/s or m/s: past this the asset has exploded


def _flag(flags: list, severity: str, where: str, what: str, hint: str = "") -> None:
    flags.append({"severity": severity, "where": where, "what": what, "hint": hint})


def _close(a: float, b: float, rel: float = 1e-3, absolute: float = 1e-6) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b)) + absolute


# ---------- readers: one dict per joint, SI units (radians) ----------

def read_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    links = {}
    for l in root.findall("link"):
        mass = l.find("inertial/mass")
        inertia = l.find("inertial/inertia")
        links[l.get("name")] = {
            "mass": float(mass.get("value")) if mass is not None else 0.0,
            "inertia": None if inertia is None else np.array(
                [[float(inertia.get(k)) for k in ("ixx", "ixy", "ixz")], [float(inertia.get(k)) for k in ("ixy", "iyy", "iyz")],
                 [float(inertia.get(k)) for k in ("ixz", "iyz", "izz")]]),
        }
    joints = {}
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            continue
        lim, dyn = j.find("limit"), j.find("dynamics")
        joints[j.get("name")] = {
            "type": j.get("type"),
            "lower": float(lim.get("lower", "nan")) if lim is not None else float("nan"),
            "upper": float(lim.get("upper", "nan")) if lim is not None else float("nan"),
            "effort": float(lim.get("effort", 0)) if lim is not None else 0.0,
            "velocity": float(lim.get("velocity", 0)) if lim is not None else 0.0,
            "damping": float(dyn.get("damping", 0)) if dyn is not None else 0.0,
            "friction": float(dyn.get("friction", 0)) if dyn is not None else 0.0,
            "mimic": j.find("mimic") is not None,
        }
    return {"links": links, "joints": joints}


def read_mjcf(path: Path) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(path))
    joints = {}
    for i in range(m.njnt):
        if m.jnt_type[i] not in (2, 3):  # slide, hinge
            continue
        dof = m.jnt_dofadr[i]
        joints[m.joint(i).name] = {
            "armature": float(m.dof_armature[dof]), "damping": float(m.dof_damping[dof]),
            "friction": float(m.dof_frictionloss[dof]),
            "lower": float(m.jnt_range[i][0]) if m.jnt_limited[i] else float("nan"),
            "upper": float(m.jnt_range[i][1]) if m.jnt_limited[i] else float("nan"),
            "kp": 0.0, "kv": 0.0, "effort": 0.0,
        }
    for a in range(m.nu):
        j = m.joint(int(m.actuator_trnid[a, 0])).name
        if j in joints:
            joints[j]["kp"] = float(m.actuator_gainprm[a, 0])
            joints[j]["kv"] = float(-m.actuator_biasprm[a, 2])
            if m.actuator_forcelimited[a]:
                joints[j]["effort"] = float(m.actuator_forcerange[a, 1])
    return {"joints": joints, "mass": {m.body(i).name: float(m.body_mass[i]) for i in range(1, m.nbody)},
            "timestep": float(m.opt.timestep)}


def read_usd(path: Path, variant: str) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection(variant)
    joints, mass = {}, {}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAttribute("physics:mass"):
            mass[p.GetName()] = float(p.GetAttribute("physics:mass").Get())
        revolute = p.IsA(UsdPhysics.RevoluteJoint)
        if not (revolute or p.IsA(UsdPhysics.PrismaticJoint)):
            continue
        k = DEG if revolute else 1.0
        drive = "angular" if revolute else "linear"

        def get(n, p=p):
            return float(p.GetAttribute(n).Get()) if p.HasAttribute(n) and p.GetAttribute(n).Get() is not None else None

        joint = UsdPhysics.Joint(p)
        body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        q = p.GetAttribute("physics:localRot1").Get()
        from scipy.spatial.transform import Rotation

        R1 = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        T = np.array(UsdGeom.Xformable(body1).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        joints[p.GetName()] = {
            "armature": get("physxJoint:armature") if variant == "physx" else get("mjc:armature"),
            "newton_armature": get("newton:armature"),
            "friction": get("physxJoint:jointFriction") if variant == "physx" else get("mjc:frictionloss"),
            "kp": (get(f"drive:{drive}:physics:stiffness") or 0.0) * k,
            "damping_total": (get(f"drive:{drive}:physics:damping") or 0.0) * k,
            "effort": get(f"drive:{drive}:physics:maxForce"),
            "lower": None if get("physics:lowerLimit") is None else get("physics:lowerLimit") / k,
            "upper": None if get("physics:upperLimit") is None else get("physics:upperLimit") / k,
            "axis_world": (T[:3, :3] @ R1)[:, 0],
        }
    return {"joints": joints, "mass": mass}


# ---------- static audit and consistency ----------

def audit(urdf: dict, flags: list) -> dict:
    masses = []
    for name, l in urdf["links"].items():
        if l["mass"] <= 0:
            _flag(flags, "error", f"urdf link {name}", "non-positive mass", "a link needs mass: check the part materials")
            continue
        masses.append(l["mass"])
        I = l["inertia"]
        if I is not None:
            w = np.linalg.eigvalsh((I + I.T) / 2)
            if w.min() <= 0:
                _flag(flags, "error", f"urdf link {name}", "inertia is not positive definite",
                      "degenerate geometry or a zero-volume part")
            elif w[0] + w[1] < w[2] * (1 - 1e-6):
                _flag(flags, "error", f"urdf link {name}", "inertia violates the triangle inequality")
    real = [m for m in masses if m > 1e-3]
    if real and max(real) / min(real) > 1000:
        _flag(flags, "warning", "urdf", f"link mass ratio {max(real) / min(real):.0f}:1",
              "extreme ratios make contacts and gain tuning fragile; check tiny links or a missing material")
    for name, j in urdf["joints"].items():
        if j["type"] in ("revolute", "prismatic"):
            if not j["lower"] < j["upper"]:
                _flag(flags, "error", f"urdf joint {name}", f"limits {j['lower']}..{j['upper']} are not an interval")
            if j["effort"] <= 0 or j["velocity"] <= 0:
                _flag(flags, "warning", f"urdf joint {name}", "no effort or velocity limit",
                      "set joints.<name>.effort/velocity in the spec")
    return {"links": len(urdf["links"]), "joints": len(urdf["joints"]), "total_mass": round(sum(masses), 4)}


def consistency(urdf: dict, mjcf: dict | None, usd: dict[str, dict], flags: list) -> None:
    floor_note = "raise armature to at least 16*kp*dt^2, or lower kp"
    for name, u in urdf["joints"].items():
        if u["mimic"]:
            continue
        m = mjcf["joints"].get(name) if mjcf else None
        for variant, usd_d in usd.items():
            s = usd_d["joints"].get(name)
            if s is None:
                _flag(flags, "error", f"usd[{variant}] joint {name}", "missing from the USD")
                continue
            if m is not None:
                if s["armature"] is None or not _close(s["armature"], m["armature"], 1e-3, 1e-7):
                    _flag(flags, "error", f"usd[{variant}] joint {name}",
                          f"armature {s['armature']} differs from the MJCF's {m['armature']:.6g}",
                          "an engine ignores armature in a namespace it does not read: author it in every layer")
                if variant == "mujoco" and (s["newton_armature"] is None or not _close(s["newton_armature"], m["armature"], 1e-3, 1e-7)):
                    _flag(flags, "error", f"usd[{variant}] joint {name}", "newton:armature missing or different",
                          "Newton's default USD import reads newton:* only")
                if not _close(s["kp"], m["kp"], 1e-3, 1e-6):
                    _flag(flags, "error", f"usd[{variant}] joint {name}", f"stiffness {s['kp']:.6g}/rad vs MJCF kp {m['kp']:.6g}",
                          "angular drive gains are per degree in UsdPhysics")
                total = m["damping"] + m["kv"]
                if not _close(s["damping_total"], total, 1e-3, 1e-6):
                    _flag(flags, "error", f"usd[{variant}] joint {name}",
                          f"drive damping {s['damping_total']:.6g}/rad vs MJCF damping+kv {total:.6g}")
                if s["effort"] is not None and m["effort"] and not _close(s["effort"], m["effort"]):
                    _flag(flags, "error", f"usd[{variant}] joint {name}", f"effort {s['effort']:.6g} vs MJCF {m['effort']:.6g}")
            if not np.isnan(u["lower"]) and s["lower"] is not None and not (_close(s["lower"], u["lower"], 1e-4, 1e-5)
                                                                           and _close(s["upper"], u["upper"], 1e-4, 1e-5)):
                _flag(flags, "error", f"usd[{variant}] joint {name}",
                      f"limits {s['lower']:.4f}..{s['upper']:.4f} vs URDF {u['lower']:.4f}..{u['upper']:.4f}",
                      "USD angular limits are degrees")
            if s["armature"] is not None and s["kp"] and s["armature"] < 16 * s["kp"] * DT**2 * (1 - 1e-6):
                _flag(flags, "error", f"usd[{variant}] joint {name}",
                      f"armature {s['armature']:.6g} is below the stability floor {16 * s['kp'] * DT**2:.6g}", floor_note)
        if m is not None:
            if not np.isnan(u["lower"]) and not (np.isnan(m["lower"]) or (_close(m["lower"], u["lower"], 1e-4, 1e-5)
                                                                         and _close(m["upper"], u["upper"], 1e-4, 1e-5))):
                _flag(flags, "error", f"mjcf joint {name}", f"range {m['lower']:.4f}..{m['upper']:.4f} vs URDF "
                                                            f"{u['lower']:.4f}..{u['upper']:.4f}")
            if m["kp"] and m["armature"] < 16 * m["kp"] * m.get("dt", DT) ** 2 * (1 - 1e-6):
                _flag(flags, "error", f"mjcf joint {name}", f"armature {m['armature']:.6g} is below the stability floor "
                                                            f"{16 * m['kp'] * DT**2:.6g}", floor_note)
    if mjcf:
        for body, mass in mjcf["mass"].items():
            if body in urdf["links"] and not _close(mass, urdf["links"][body]["mass"], 1e-3, 1e-7):
                _flag(flags, "error", f"mjcf body {body}", f"mass {mass:.6g} vs URDF {urdf['links'][body]['mass']:.6g}")
        for variant, usd_d in usd.items():
            for body, mass in usd_d["mass"].items():
                if body in urdf["links"] and not _close(mass, urdf["links"][body]["mass"], 1e-3, 1e-7):
                    _flag(flags, "error", f"usd[{variant}] body {body}", f"mass {mass:.6g} vs URDF {urdf['links'][body]['mass']:.6g}")
        _axes(mjcf, usd, urdf, flags)


def _axes(mjcf: dict, usd: dict, urdf: dict, flags: list) -> None:
    """Joint axes in the world at the zero configuration: MuJoCo's against the USD's."""
    import mujoco

    path = mjcf.get("path")
    if path is None:
        return
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    for variant, usd_d in usd.items():
        for name, s in usd_d["joints"].items():
            try:
                i = m.joint(name).id
            except KeyError:
                continue
            a, b = d.xaxis[i], s["axis_world"]
            if min(np.linalg.norm(a - b), np.linalg.norm(a + b)) > 1e-4:  # the sign is the actuator's convention
                _flag(flags, "error", f"usd[{variant}] joint {name}", f"world axis {np.round(b, 3).tolist()} vs MJCF "
                                                                    f"{np.round(a, 3).tolist()}", "a frame or axis conversion is wrong")
            elif np.linalg.norm(a - b) > 1e-4:
                _flag(flags, "warning", f"usd[{variant}] joint {name}", "axis sign differs from the MJCF",
                      "fine if the limits and targets match; check the joint direction in the sim")


# ---------- dynamics in MuJoCo ----------

def _drive(m, d, ctrl_fn, seconds: float, speed_limit: float) -> dict:
    import mujoco

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    max_v = max_a = 0.0
    finite, min_dist = True, 1.0
    steps = int(seconds / m.opt.timestep)
    for k in range(steps):
        d.ctrl[:] = ctrl_fn(k * m.opt.timestep)
        mujoco.mj_step(m, d)
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            finite = False
            break
        max_v = max(max_v, float(np.abs(d.qvel).max(initial=0.0)))
        max_a = max(max_a, float(np.abs(d.qacc).max(initial=0.0)))
        for c in d.contact[: d.ncon]:
            min_dist = min(min_dist, float(c.dist))
    exploded = (not finite) or max_v > BLOWUP_SPEED
    return {"finite": finite, "max_speed": round(max_v, 3), "max_accel": round(max_a, 1),
            "deepest_contact_m": round(min_dist, 5) if min_dist < 1.0 else None,
            "exploded": exploded, "over_speed_limit": bool(speed_limit and max_v > 3 * speed_limit)}


def dynamics(xml: Path, urdf: dict, seconds: float, flags: list) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    acts = [(a, m.joint(int(m.actuator_trnid[a, 0])).name) for a in range(m.nu) if m.actuator_biastype[a] != 0]
    follower = {j for j, d in urdf["joints"].items() if d["mimic"]}  # driven through its leader, not on its own
    lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
    unbounded = ~(m.actuator_ctrllimited.astype(bool))
    lo[unbounded], hi[unbounded] = -1.0, 1.0
    vmax = max([urdf["joints"][j]["velocity"] for _, j in acts if j in urdf["joints"]] or [0.0])
    out = {"actuated_joints": len(acts)}

    out["hold"] = r = _drive(m, d, lambda t: np.zeros(m.nu), seconds, vmax)
    if r["exploded"]:
        _flag(flags, "error", "mujoco hold", "unstable at rest", "check inertia, armature and contact softness")

    steps = []
    for a, j in acts:
        if j in follower:
            continue
        target = np.zeros(m.nu)
        target[a] = np.clip(0.6 * (hi[a] if abs(hi[a]) >= abs(lo[a]) else lo[a]), lo[a], hi[a])
        res = _drive(m, d, lambda t, tg=target: tg, seconds, vmax)
        err = float(abs(d.qpos[m.jnt_qposadr[m.joint(j).id]] - target[a]))
        peak = float(res["max_speed"])
        res.update(joint=j, final_error=round(err, 5))
        steps.append(res)
        if res["exploded"]:
            _flag(flags, "error", f"mujoco step response {j}", "exploded", "armature/damping too low for kp, or kp too high")
        elif err > 0.1 * max(abs(target[a]), 1e-3) and abs(target[a]) > 1e-3:
            _flag(flags, "warning", f"mujoco step response {j}", f"settles {err:.4f} from the target",
                  "effort limit too low for the load, or the target hits a joint stop or a contact")
        if res["over_speed_limit"]:
            _flag(flags, "warning", f"mujoco step response {j}", f"peak speed {peak:.2f} is over 3x the velocity limit",
                  "raise joint damping or lower kp")
    out["steps"] = steps

    period = 0.1
    amp = 0.8 * np.minimum(np.abs(hi), np.abs(lo))
    amp = np.where(amp > 1e-6, amp, 0.5 * np.maximum(np.abs(hi), np.abs(lo)))
    out["bang_bang"] = r = _drive(m, d, lambda t: amp * (1.0 if int(t / period) % 2 == 0 else -1.0), seconds, vmax)
    if r["exploded"]:
        _flag(flags, "error", "mujoco bang-bang", f"exploded (peak speed {r['max_speed']}, finite={r['finite']})",
              "high accelerations need armature >= 16*kp*dt^2 and damping; check every engine's armature")
    elif r["over_speed_limit"]:
        _flag(flags, "warning", "mujoco bang-bang", f"peak speed {r['max_speed']} over 3x the velocity limit")

    rng = np.random.default_rng(0)
    sweep = []
    for _ in range(4):
        tgt = lo + (hi - lo) * rng.uniform(0.15, 0.85, m.nu)
        sweep.append(_drive(m, d, lambda t, tg=tgt: tg, seconds, vmax))
    out["random_sweep"] = {"runs": len(sweep), "exploded": sum(r["exploded"] for r in sweep),
                           "deepest_contact_m": min([r["deepest_contact_m"] for r in sweep if r["deepest_contact_m"] is not None]
                                                    or [None], default=None) if any(r["deepest_contact_m"] is not None for r in sweep) else None,
                           "max_speed": max(r["max_speed"] for r in sweep)}
    if out["random_sweep"]["exploded"]:
        _flag(flags, "error", "mujoco random sweep", f"{out['random_sweep']['exploded']} of {len(sweep)} runs exploded",
              "usually a collision shape pair that should be excluded, or a missing armature")
    deepest = out["random_sweep"]["deepest_contact_m"]
    if deepest is not None and deepest < -0.01:
        _flag(flags, "warning", "mujoco random sweep", f"contact penetrates {-deepest * 1000:.1f} mm",
              "fat collision shapes at a joint: switch that link to decompose or primitives")
    return out


# ---------- other engines ----------

def newton_import(usda: Path, joints_expected: dict, flags: list) -> dict:
    """What Newton's default USD import reads: the physx variant gives it zero armature, the mujoco variant the real one."""
    try:
        import newton
        from pxr import Usd
    except ImportError:
        return {"skipped": "newton is not installed"}
    out = {}
    for variant in ("physx", "mujoco"):
        stage = Usd.Stage.Open(str(usda))
        stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection(variant)
        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize(device="cpu")
        # by joint name: Newton can hold more DOFs than the articulation's joints (loop-closure joints)
        labels = [str(k).split("/")[-1] for k in model.joint_label]
        starts, arm = model.joint_qd_start.numpy(), model.joint_armature.numpy()
        by_name = {n: round(float(arm[int(starts[i])]), 6) for i, n in enumerate(labels)
                   if i + 1 < len(starts) and starts[i + 1] > starts[i]}
        out[variant] = {"dofs": int(model.joint_dof_count), "armature": by_name}
    for name, expected in joints_expected.items():
        if not expected.get("armature"):
            continue
        got = out["mujoco"]["armature"].get(name)
        if got is None or not np.isclose(got, expected["armature"], rtol=1e-3, atol=1e-7):
            _flag(flags, "error", f"newton[mujoco variant] joint {name}",
                  f"Newton reads armature {got} but the file says {expected['armature']:.6g}",
                  "author newton:armature / mjc:armature on every joint")
    if any(v.get("armature") for v in joints_expected.values()) and all(v == 0 for v in out["physx"]["armature"].values()):
        _flag(flags, "info", "newton[physx variant]", "Newton's default import ignores physx armature",
              "run Newton on the Physics=mujoco variant (or pass SchemaResolverPhysx)")
    return out


def isaac_readback(out_dir: Path, urdf: Path, flags: list) -> dict:
    from .validate import check_isaac

    res = check_isaac(urdf, fixed=True)
    if res.get("skipped"):
        return res
    if not res.get("ok"):
        _flag(flags, "error", "isaac", res.get("error", "Isaac Sim failed"))
        return res
    for mm in res.get("mismatches", []):
        _flag(flags, "error", f"isaac joint {mm['joint']}", f"{mm['field']}: file says {mm['expected']}, Isaac applied {mm['got']}",
              "the loader lost or converted a value: compare units and the Physics variant")
    st = res.get("stress", {})
    if st.get("exploded"):
        _flag(flags, "error", "isaac stress", f"exploded (peak joint speed {st.get('max_joint_speed')}, finite={st.get('finite')})",
              "armature or damping not applied in Isaac: see the mismatches")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cad2urdf.asset_test", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="a robot output folder (from cad2urdf.route or cad2urdf)")
    ap.add_argument("--sims", default="mujoco", help="mujoco (always), newton, isaac (launches Isaac Sim)")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    urdf_path = next(p for p in sorted(out.glob("*.urdf")) if not p.name.startswith("_"))
    xml = next(iter(sorted((out / "mjcf").glob("*.xml"))), None)
    usda = out / "usd" / f"{urdf_path.stem}.usda"
    flags: list = []
    urdf = read_urdf(urdf_path)
    result = {"robot": urdf_path.stem, "audit": audit(urdf, flags)}
    mjcf = read_mjcf(xml) if xml else None
    if mjcf:
        mjcf["path"] = xml
    usd = {v: read_usd(usda, v) for v in ("physx", "mujoco")} if usda.exists() else {}
    if not usda.exists():
        _flag(flags, "warning", "usd", "no USD asset in the folder", "re-run the compile: it writes usd/<robot>.usda")
    consistency(urdf, mjcf, usd, flags)
    sims = args.sims.split(",")
    if xml and "mujoco" in sims:
        result["mujoco"] = dynamics(xml, urdf, args.seconds, flags)
    if "newton" in sims and usda.exists():
        try:
            result["newton"] = newton_import(usda, usd["physx"]["joints"], flags)
        except Exception as e:  # noqa: BLE001  (one engine's failure is a finding, not a crash)
            _flag(flags, "error", "newton", f"could not import the USD: {type(e).__name__}: {e}",
                  "see the traceback with python -m cad2urdf.asset_test --sims newton")
            result["newton"] = {"error": str(e)}
    if "isaac" in sims:
        result["isaac"] = isaac_readback(out, urdf_path, flags)
    result["flags"] = flags
    errors = sum(f["severity"] == "error" for f in flags)
    result["ok"] = errors == 0
    (out / "asset_test.json").write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{result['robot']}: {errors} error(s), {sum(f['severity'] == 'warning' for f in flags)} warning(s), "
          f"{sum(f['severity'] == 'info' for f in flags)} info  ->  {out / 'asset_test.json'}")
    for f in flags:
        print(f"  [{f['severity']}] {f['where']}: {f['what']}" + (f"  (hint: {f['hint']})" if f["hint"] else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
