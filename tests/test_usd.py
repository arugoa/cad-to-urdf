"""The USD asset: structure, frames, units, per-engine attributes, and what Newton reads back."""

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from cad2urdf.writers import TIMESTEP, _servo_dynamics

@pytest.fixture(scope="module")
def asset(arm4_out):
    robot, out = arm4_out
    return robot, out / "usd" / f"{robot.name}.usda"


def _stage(path, variant="physx"):
    stage = Usd.Stage.Open(str(path))
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection(variant)
    return stage


def _world(prim_xf):
    return np.array(prim_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def test_isaac_sim_asset_structure(asset):
    robot, path = asset
    root = path.parent
    for rel in ("configuration/base.usda", "configuration/instances.usda", "configuration/geometries.usda",
                "configuration/materials.usda", "configuration/robot.usda", "configuration/Physics/physics.usda",
                "configuration/Physics/physx.usda", "configuration/Physics/mujoco.usda"):
        assert (root / rel).read_text().startswith("#usda 1.0"), rel  # text USD, not binary crate
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetDefaultPrim()
    assert prim.GetName() == robot.name
    vs = prim.GetVariantSets().GetVariantSet("Physics")
    assert set(vs.GetVariantNames()) == {"none", "physics", "physx", "mujoco"} and vs.GetVariantSelection() == "physx"
    assert prim.GetAttribute("isaac:namespace").Get() == robot.name
    none = _stage(path, "none")
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in none.Traverse())  # the "none" variant has no physics


def test_bodies_mass_and_articulation(asset):
    robot, path = asset
    stage = _stage(path)
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert set(bodies) == set(robot.links)
    for name, link in robot.links.items():
        assert bodies[name].GetAttribute("physics:mass").Get() == pytest.approx(link.mass, rel=1e-5)
    roots = [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [robot.root]


def test_joint_frames_reproduce_the_link_poses(asset):
    """Both joint frames must coincide in the world at the zero configuration, and the axis must be X."""
    robot, path = asset
    stage = _stage(path)
    for j in robot.joints.values():
        prim = stage.GetPrimAtPath(f"/{robot.name}/joints/{j.name}")
        joint = UsdPhysics.Joint(prim)
        b0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        b1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])

        def frame(body, pos_attr, rot_attr):
            T = _world(UsdGeom.Xformable(body))
            q = prim.GetAttribute(rot_attr).Get()
            R = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
            local = np.eye(4)
            local[:3, :3], local[:3, 3] = R, np.array(prim.GetAttribute(pos_attr).Get())
            return T @ local

        F0, F1 = frame(b0, "physics:localPos0", "physics:localRot0"), frame(b1, "physics:localPos1", "physics:localRot1")
        np.testing.assert_allclose(F0, F1, atol=1e-5, err_msg=j.name)
        if j.type != "fixed":
            assert prim.GetAttribute("physics:axis").Get() == "X"
            np.testing.assert_allclose(F1[:3, 0], j.axis / np.linalg.norm(j.axis), atol=1e-5, err_msg=j.name)


def test_units_limits_in_degrees_and_gains_per_degree(asset):
    robot, path = asset
    stage = _stage(path)
    for j in robot.moving_joints():
        prim = stage.GetPrimAtPath(f"/{robot.name}/joints/{j.name}")
        revolute = j.type in ("revolute", "continuous")
        scale = 180 / np.pi if revolute else 1.0
        if j.type != "continuous":
            assert prim.GetAttribute("physics:lowerLimit").Get() == pytest.approx(j.lower * scale, rel=1e-5)
            assert prim.GetAttribute("physics:upperLimit").Get() == pytest.approx(j.upper * scale, rel=1e-5)
        drive = "angular" if revolute else "linear"
        kp = j.actuator.get("kp", 0.0) if j.actuator.get("kind") == "position" else 0.0
        stiffness = prim.GetAttribute(f"drive:{drive}:physics:stiffness").Get()
        assert stiffness == pytest.approx(kp / scale, rel=1e-5)


def test_armature_is_stated_in_every_engines_namespace(asset):
    robot, path = asset
    for variant, names in (("physx", ["physxJoint:armature"]), ("mujoco", ["mjc:armature", "newton:armature"])):
        stage = _stage(path, variant)
        for j in robot.moving_joints():
            prim = stage.GetPrimAtPath(f"/{robot.name}/joints/{j.name}")
            _, armature = _servo_dynamics(j, TIMESTEP)
            for n in names:
                assert prim.GetAttribute(n).Get() == pytest.approx(armature, rel=1e-5), (variant, j.name, n)
            kp = j.actuator.get("kp", 0.0) if j.actuator.get("kind") == "position" else 0.0
            assert armature >= 16 * kp * TIMESTEP**2 - 1e-12  # the explicit-spring stability floor


def test_mimic_in_both_engine_layers(asset):
    robot, path = asset
    followers = [j for j in robot.joints.values() if j.mimic]
    assert followers, "arm4's right finger mimics the left"
    f = followers[0]
    leader = f"/{robot.name}/joints/{f.mimic['joint']}"
    mj_stage = _stage(path, "mujoco")  # keep the stage alive: its prims expire with it
    mj = mj_stage.GetPrimAtPath(f"/{robot.name}/joints/{f.name}")
    assert [str(t) for t in mj.GetRelationship("newton:mimicJoint").GetTargets()] == [leader]
    assert mj.GetAttribute("newton:mimicCoef1").Get() == pytest.approx(f.mimic.get("multiplier", 1.0))
    px_stage = _stage(path, "physx")
    px = px_stage.GetPrimAtPath(f"/{robot.name}/joints/{f.name}")
    authored = px.GetMetadata("apiSchemas").explicitItems  # GetAppliedSchemas hides schemas without a plugin here
    assert any(str(t).startswith("PhysxMimicJointAPI") for t in authored)
    assert px.GetAttribute("physxMimicJoint:transX:gearing").Get() == pytest.approx(-f.mimic.get("multiplier", 1.0))


def test_newton_reads_the_armature_from_the_mujoco_variant(asset):
    """Newton's default import reads newton:* only: the physx variant gives it zero armature (the 'explodes at
    high acceleration' failure), the mujoco variant gives it the real values."""
    newton = pytest.importorskip("newton")
    robot, path = asset
    want = np.array([_servo_dynamics(j, TIMESTEP)[1] for j in robot.moving_joints()])
    got = {}
    for variant in ("physx", "mujoco"):
        builder = newton.ModelBuilder()
        builder.add_usd(_stage(path, variant))
        got[variant] = np.array(builder.finalize(device="cpu").joint_armature.numpy())
    assert np.allclose(got["physx"], 0.0)  # silently ignored: why the asset test checks every engine
    assert sorted(got["mujoco"]) == pytest.approx(sorted(want), rel=1e-4)


def test_colliders_share_a_physics_material_with_the_mjcf_friction(asset):
    robot, path = asset
    stage = _stage(path)
    mat = stage.GetPrimAtPath(f"/{robot.name}/PhysicsMaterials/default")
    mu = float((robot.spec.get("contact", {}).get("friction") or [1.0])[0])
    assert mat.GetAttribute("physics:staticFriction").Get() == pytest.approx(mu)
    from pxr import UsdShade

    collider = stage.GetPrimAtPath(f"/{robot.name}/{robot.root}/collisions/c0")
    bound = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")[0]
    assert bound.GetPath() == mat.GetPath()
