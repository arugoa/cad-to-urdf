"""USDA asset in the Isaac Sim 6.x asset structure, written from the same IR as the URDF and MJCF.

    usd/
      <robot>.usda                 interface layer: references configuration/base.usda, variant set "Physics"
      configuration/
        base.usda                  kinematic hierarchy: one Xform per link, referencing instances.usda
        instances.usda             per link: visual meshes and collision shapes (meshes reference geometries.usda)
        geometries.usda            mesh data
        materials.usda             UsdPreviewSurface materials
        robot.usda                 robot metadata (sublayer of the interface)
        Physics/
          physics.usda             neutral UsdPhysics: bodies, mass, colliders, joints, drives, articulation
          physx.usda               physics.usda + PhysX attributes (armature, joint friction, max velocity, mimic)
          mujoco.usda              physics.usda + MuJoCo/Newton attributes (armature, frictionloss, mimic)

The Physics variants are none, physics, physx (default) and mujoco. Each engine layer states armature in its own
namespace (physxJoint:armature, mjc:armature, newton:armature) because an engine ignores the others': Newton's
default USD import reads only ``newton:*``. Angular quantities follow UsdPhysics (degrees; drive gains per degree).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation

from .geometry import budget_mesh
from .model import Link, Robot
from .writers import TIMESTEP, _mass_inertia, _servo_dynamics

DEG = 180.0 / np.pi
DEFAULT_VARIANT = "physx"
LAYERS = "configuration"  # the folder Isaac Sim / Isaac Lab assets keep their layers in
VARIANTS = ("none", "physics", "physx", "mujoco")


def _ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return s if re.match(r"[A-Za-z_]", s) else "_" + s


def _quat(R: np.ndarray) -> Gf.Quatf:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _frame_with_x(axis: np.ndarray) -> np.ndarray:
    """Rotation whose first column is ``axis``: UsdPhysics joints act about the X axis of their frame."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, a)
    y /= np.linalg.norm(y)
    return np.column_stack([a, y, np.cross(a, y)])


def _stage(path: Path, root: str | None = None) -> Usd.Stage:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    if root:
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, f"/{root}").GetPrim())
    return stage


def _set_pose(xf: UsdGeom.Xformable, pos, R) -> None:
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(_quat(R))


def _mesh(stage: Usd.Stage, path: str, m) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())]))


def _attr(prim: Usd.Prim, name: str, typ, value) -> None:
    prim.CreateAttribute(name, typ).Set(value)


def _names(robot: Robot) -> dict:
    """Prim names for every visual and every mesh collision, unique per link."""
    out = {}
    for link in robot.links.values():
        for key in link.visuals:
            out[(link.name, "v", key)] = _ident(f"{link.name}__v_{key}")
        for i, g in enumerate(link.collisions):
            if g.kind == "mesh":
                out[(link.name, "c", i)] = _ident(f"{link.name}__c_{i}")
    return out


def _geometries(robot: Robot, path: Path, names: dict) -> None:
    stage = _stage(path, "geometries")
    for link in robot.links.values():
        for key, mesh in link.visuals.items():
            _mesh(stage, f"/geometries/{names[(link.name, 'v', key)]}", budget_mesh(mesh))
        for i, g in enumerate(link.collisions):
            if g.kind == "mesh":
                _mesh(stage, f"/geometries/{names[(link.name, 'c', i)]}", g.mesh)
    stage.Save()


def _materials(robot: Robot, path: Path) -> dict[str, str]:
    stage = _stage(path)
    stage.SetDefaultPrim(UsdGeom.Scope.Define(stage, "/Looks").GetPrim())
    ids = {}
    for mat, rgba in robot.used_materials().items():
        ids[mat] = _ident(mat)
        r, g, b, a = (float(x) for x in rgba.split())
        m = UsdShade.Material.Define(stage, f"/Looks/{ids[mat]}")
        sh = UsdShade.Shader.Define(stage, f"/Looks/{ids[mat]}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(a)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    stage.Save()
    return ids


def _collision_prim(stage: Usd.Stage, path: str, g, names: dict, link: Link, i: int) -> None:
    if g.kind == "mesh":
        prim = stage.DefinePrim(path, "Mesh")
        prim.GetReferences().AddReference("./geometries.usda", Sdf.Path(f"/geometries/{names[(link.name, 'c', i)]}"))
        xf = UsdGeom.Xformable(prim)
    elif g.kind == "box":
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = cube
    elif g.kind == "cylinder":
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.CreateRadiusAttr(float(g.size[0]))
        cyl.CreateHeightAttr(float(g.size[1]))
        cyl.CreateAxisAttr(UsdGeom.Tokens.z)
        xf = cyl
    else:
        sph = UsdGeom.Sphere.Define(stage, path)
        sph.CreateRadiusAttr(float(g.size[0]))
        xf = sph
    _set_pose(xf, g.transform[:3, 3], g.transform[:3, :3])
    if g.kind == "box":
        xf.AddScaleOp().Set(Gf.Vec3f(*[float(s) for s in g.size]))
    UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreatePurposeAttr(UsdGeom.Tokens.guide)


def _instances(robot: Robot, path: Path, names: dict) -> None:
    """Per link: the visual meshes and collision shapes, referencing the mesh data in geometries.usda."""
    stage = _stage(path, "instances")
    for link in robot.links.values():
        lp = f"/instances/{_ident(link.name)}"
        UsdGeom.Xform.Define(stage, lp)
        UsdGeom.Xform.Define(stage, f"{lp}/visuals")
        UsdGeom.Xform.Define(stage, f"{lp}/collisions")
        for key in link.visuals:
            gname = names[(link.name, "v", key)]
            prim = stage.DefinePrim(f"{lp}/visuals/{gname}", "Mesh")
            prim.GetReferences().AddReference("./geometries.usda", Sdf.Path(f"/geometries/{gname}"))
        for i, g in enumerate(link.collisions):
            _collision_prim(stage, f"{lp}/collisions/c{i}", g, names, link, i)
    stage.Save()


def _base(robot: Robot, path: Path, names: dict, mats: dict[str, str]) -> None:
    """The kinematic hierarchy: a link Xform at its zero-configuration pose, referencing its instances."""
    n = robot.name
    stage = _stage(path, n)
    looks = stage.DefinePrim(f"/{n}/Looks", "Scope")
    looks.GetReferences().AddReference("./materials.usda")
    for link in robot.links.values():
        lp = f"/{n}/{_ident(link.name)}"
        xf = UsdGeom.Xform.Define(stage, lp)
        xf.GetPrim().GetReferences().AddReference("./instances.usda", Sdf.Path(f"/instances/{_ident(link.name)}"))
        _set_pose(xf, link.origin, link.rotation)
        for key in link.visuals:
            prim = stage.OverridePrim(f"{lp}/visuals/{names[(link.name, 'v', key)]}")
            material = UsdShade.Material(stage.GetPrimAtPath(f"/{n}/Looks/{mats[link.material(key)]}"))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    stage.Save()


def _robot_layer(robot: Robot, path: Path) -> None:
    stage = _stage(path, robot.name)
    prim = stage.GetDefaultPrim()
    _attr(prim, "isaac:namespace", Sdf.ValueTypeNames.String, robot.name)
    stage.Save()


def _principal(link: Link) -> tuple[float, tuple, Gf.Vec3f, Gf.Quatf]:
    mass, I = _mass_inertia(link)
    w, V = np.linalg.eigh((I + I.T) / 2)
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return mass, tuple(float(c) for c in link.com), Gf.Vec3f(*[float(x) for x in w]), _quat(V)


def _joint_frames(robot: Robot, j) -> tuple:
    xyz, R = robot.child_in_parent(j)
    Rj = _frame_with_x(robot.axis_local(j)) if j.type != "fixed" else np.eye(3)
    return xyz, R @ Rj, Rj


def _physics(robot: Robot, path: Path, excludes) -> None:
    n = robot.name
    stage = _stage(path, n)
    P = lambda link: Sdf.Path(f"/{n}/{_ident(link)}")  # noqa: E731
    for link in robot.links.values():
        prim = stage.OverridePrim(P(link.name))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass, com, diag, axes = _principal(link)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(float(mass))
        m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        m.CreateDiagonalInertiaAttr(diag)
        m.CreatePrincipalAxesAttr(axes)
        for i, g in enumerate(link.collisions):
            cp = stage.OverridePrim(P(link.name).AppendPath(f"collisions/c{i}"))
            UsdPhysics.CollisionAPI.Apply(cp)
            if g.kind == "mesh":
                UsdPhysics.MeshCollisionAPI.Apply(cp).CreateApproximationAttr("convexHull")
    friction = float((robot.spec.get("contact", {}).get("friction") or [1.0])[0])
    UsdGeom.Scope.Define(stage, f"/{n}/PhysicsMaterials")
    material = UsdShade.Material.Define(stage, f"/{n}/PhysicsMaterials/default")
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr(friction)
    pm.CreateDynamicFrictionAttr(friction)
    pm.CreateRestitutionAttr(0.0)
    for link in robot.links.values():  # the same friction the MJCF uses, bound to every collider
        for i in range(len(link.collisions)):
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(P(link.name).AppendPath(f"collisions/c{i}"))).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdGeom.Scope.Define(stage, f"/{n}/joints")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(P(robot.root)))
    if not robot.floating_base:  # weld the base to the world
        root = robot.links[robot.root]
        rj = UsdPhysics.FixedJoint.Define(stage, f"/{n}/joints/root_joint")
        rj.CreateBody1Rel().SetTargets([P(robot.root)])
        rj.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in root.origin]))
        rj.CreateLocalRot0Attr(_quat(root.rotation))
        rj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        rj.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    for j in robot.joints.values():
        _joint(stage, robot, j, P)
    for c in robot.closures:  # loops a tree can't hold: a ball constraint outside the articulation
        cj = UsdPhysics.SphericalJoint.Define(stage, f"/{n}/joints/{_ident(c['name'])}")
        cj.CreateBody0Rel().SetTargets([P(c["link1"])])
        cj.CreateBody1Rel().SetTargets([P(c["link2"])])
        cj.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in c["anchor1"]]))
        cj.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in c["anchor2"]]))
        cj.CreateLocalRot0Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
        cj.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
        cj.CreateExcludeFromArticulationAttr(True)
    filtered: dict[str, list[str]] = {}
    for a, b in excludes:
        filtered.setdefault(a, []).append(b)
    for a, others in filtered.items():
        if a in robot.links:
            api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(P(a)))
            for b in others:
                if b in robot.links:
                    api.CreateFilteredPairsRel().AddTarget(P(b))
    stage.Save()


def _joint(stage: Usd.Stage, robot: Robot, j, P) -> None:
    n = robot.name
    path = f"/{n}/joints/{_ident(j.name)}"
    xyz, rot0, rot1 = _joint_frames(robot, j)
    revolute = j.type in ("revolute", "continuous")
    cls = {"fixed": UsdPhysics.FixedJoint, "prismatic": UsdPhysics.PrismaticJoint}.get(j.type, UsdPhysics.RevoluteJoint)
    joint = cls.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([P(j.parent)])
    joint.CreateBody1Rel().SetTargets([P(j.child)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in xyz]))
    joint.CreateLocalRot0Attr(_quat(rot0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(_quat(rot1))
    if j.type == "fixed":
        return
    joint.CreateAxisAttr("X")
    scale = DEG if revolute else 1.0
    if j.type in ("revolute", "prismatic"):
        joint.CreateLowerLimitAttr(float(j.lower * scale))
        joint.CreateUpperLimitAttr(float(j.upper * scale))
    kind = j.actuator.get("kind", "none")
    kp, kv = (j.actuator.get("kp", 0.0), j.actuator.get("kv", 0.0)) if kind != "none" else (0.0, 0.0)
    damping, _ = _servo_dynamics(j, TIMESTEP)
    if kind == "velocity":
        kp = 0.0
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if revolute else "linear")
    drive.CreateTypeAttr("force")
    unit = 1.0 / DEG if revolute else 1.0  # per degree for angular drives
    drive.CreateStiffnessAttr(float(kp * unit))
    drive.CreateDampingAttr(float((kv + damping) * unit))
    drive.CreateMaxForceAttr(float(j.effort) if j.effort else 3.4e38)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)


def _engine_layers(robot: Robot, folder: Path) -> None:
    n = robot.name
    movers = [j for j in robot.joints.values() if j.type != "fixed"]
    for engine in ("physx", "mujoco"):
        stage = _stage(folder / f"{engine}.usda", n)
        stage.GetRootLayer().subLayerPaths.append("./physics.usda")
        if engine == "physx":
            root = stage.OverridePrim(f"/{n}/{_ident(robot.root)}")
            root.AddAppliedSchema("PhysxArticulationAPI")
            _attr(root, "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, True)
        for j in movers:
            prim = stage.OverridePrim(f"/{n}/joints/{_ident(j.name)}")
            _, armature = _servo_dynamics(j, TIMESTEP)
            revolute = j.type in ("revolute", "continuous")
            if engine == "physx":
                prim.AddAppliedSchema("PhysxJointAPI")
                _attr(prim, "physxJoint:armature", Sdf.ValueTypeNames.Float, float(armature))
                _attr(prim, "physxJoint:jointFriction", Sdf.ValueTypeNames.Float, float(j.friction))
                if j.velocity:
                    _attr(prim, "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float,
                          float(j.velocity * (DEG if revolute else 1.0)))
            else:
                prim.AddAppliedSchema("MjcJointAPI")
                _attr(prim, "mjc:armature", Sdf.ValueTypeNames.Float, float(armature))
                _attr(prim, "mjc:frictionloss", Sdf.ValueTypeNames.Float, float(j.friction))
                _attr(prim, "newton:armature", Sdf.ValueTypeNames.Float, float(armature))  # Newton's own namespace
            if j.mimic and j.mimic["joint"] in robot.joints:
                _mimic(prim, engine, j, revolute, f"/{n}/joints/{_ident(j.mimic['joint'])}")
        stage.Save()


def _mimic(prim: Usd.Prim, engine: str, j, revolute: bool, leader: str) -> None:
    mult, offset = float(j.mimic.get("multiplier", 1.0)), float(j.mimic.get("offset", 0.0))
    scale = DEG if revolute else 1.0
    if engine == "physx":  # PhysX: follower + gearing * leader = offset
        axis = "rotX" if revolute else "transX"
        prim.AddAppliedSchema(f"PhysxMimicJointAPI:{axis}")
        _attr(prim, f"physxMimicJoint:{axis}:gearing", Sdf.ValueTypeNames.Float, -mult)
        _attr(prim, f"physxMimicJoint:{axis}:offset", Sdf.ValueTypeNames.Float, offset * scale)
        prim.CreateRelationship(f"physxMimicJoint:{axis}:referenceJoint").SetTargets([Sdf.Path(leader)])
    else:  # Newton: follower = coef0 + coef1 * leader (coef0 in degrees for a revolute follower)
        prim.AddAppliedSchema("NewtonMimicAPI")
        prim.CreateRelationship("newton:mimicJoint").SetTargets([Sdf.Path(leader)])
        _attr(prim, "newton:mimicCoef0", Sdf.ValueTypeNames.Float, offset * scale)
        _attr(prim, "newton:mimicCoef1", Sdf.ValueTypeNames.Float, mult)


def _interface(robot: Robot, path: Path) -> None:
    n = robot.name
    stage = _stage(path, n)
    stage.GetRootLayer().subLayerPaths.append(f"./{LAYERS}/robot.usda")
    root = stage.GetDefaultPrim()
    root.GetReferences().AddReference(f"./{LAYERS}/base.usda")
    vs = root.GetVariantSets().AddVariantSet("Physics")
    for v in VARIANTS:
        vs.AddVariant(v)
        vs.SetVariantSelection(v)
        if v != "none":
            with vs.GetVariantEditContext():
                root.GetPayloads().AddPayload(f"./{LAYERS}/Physics/{v}.usda")
    vs.SetVariantSelection(DEFAULT_VARIANT)
    stage.Save()


def write_usd(robot: Robot, out_dir: Path, excludes=()) -> Path:
    """Write ``out_dir/<robot>.usda`` and its configuration layers; returns the interface layer."""
    layers = out_dir / LAYERS
    names = _names(robot)
    _geometries(robot, layers / "geometries.usda", names)
    mats = _materials(robot, layers / "materials.usda")
    _instances(robot, layers / "instances.usda", names)
    _base(robot, layers / "base.usda", names, mats)
    _robot_layer(robot, layers / "robot.usda")
    _physics(robot, layers / "Physics" / "physics.usda", excludes)
    _engine_layers(robot, layers / "Physics")
    interface = out_dir / f"{robot.name}.usda"
    _interface(robot, interface)
    return interface


def expectations(usda: Path, variant: str = DEFAULT_VARIANT) -> dict:
    """What the file says per joint, in SI units (radians): the values an engine should end up applying."""
    stage = Usd.Stage.Open(str(usda))
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection(variant)
    joints, total = {}, 0.0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAttribute("physics:mass"):
            total += float(prim.GetAttribute("physics:mass").Get())
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        k = DEG if revolute else 1.0
        d, get = {}, lambda n: prim.GetAttribute(n).Get() if prim.HasAttribute(n) else None  # noqa: E731
        drive = "angular" if revolute else "linear"
        for field, attr, scale in (("armature", "physxJoint:armature", 1.0),
                                   ("stiffness", f"drive:{drive}:physics:stiffness", k),
                                   ("damping", f"drive:{drive}:physics:damping", k),
                                   ("velocity", "physxJoint:maxJointVelocity", 1 / k),
                                   ("lower", "physics:lowerLimit", 1 / k), ("upper", "physics:upperLimit", 1 / k)):
            v = get(attr)
            if v is not None:
                d[field] = float(v) * scale
        force = get(f"drive:{drive}:physics:maxForce")
        if force is not None and force < 1e30:
            d["effort"] = float(force)
        joints[prim.GetName()] = d
    return {"joints": joints, "total_mass": total}
