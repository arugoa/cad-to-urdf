"""Onshape front end against a fake API that returns the real response shapes (no network, no keys)."""

import io
import json

import numpy as np
import trimesh

from cad2urdf import frontends

DID, WID, EID = "a" * 24, "b" * 24, "c" * 24
URL = f"https://team.onshape.com/documents/{DID}/w/{WID}/e/{EID}"


def T(xyz=(0, 0, 0)):
    M = np.eye(4)
    M[:3, 3] = xyz
    return M.ravel().tolist()


def cs(origin, z=(0, 0, 1), x=(1, 0, 0)):
    z, x = np.array(z, float), np.array(x, float)
    return {"origin": list(origin), "zAxis": list(z), "xAxis": list(x), "yAxis": list(np.cross(z, x))}


def part(iid, name, pid):
    return {"id": iid, "type": "Part", "name": name, "partId": pid, "documentId": DID, "elementId": EID,
            "documentMicroversion": "d" * 24, "configuration": "default"}


def mate(fid, name, mtype, occ_a, cs_a, occ_b, cs_b):
    return {"id": fid, "featureType": "mate", "suppressed": False, "featureData": {
        "name": name, "mateType": mtype,
        "matedEntities": [{"matedOccurrence": occ_a, "matedCS": cs_a}, {"matedOccurrence": occ_b, "matedCS": cs_b}]}}


ASSEMBLY = {
    "rootAssembly": {
        "instances": [part("i_plate", "Base Plate", "JA"), part("i_post", "Post", "JB"),
                      part("i_arm", "Arm", "JC"), part("i_slide", "Carriage", "JD")],
        "occurrences": [
            {"path": ["i_plate"], "transform": T(), "fixed": True},
            {"path": ["i_post"], "transform": T((0, 0, 0.05))},
            {"path": ["i_arm"], "transform": T((0, 0, 0.30))},
            {"path": ["i_slide"], "transform": T((0.2, 0, 0.30))},
        ],
        "features": [
            mate("f1", "plate to post", "FASTENED", ["i_plate"], cs((0, 0, 0.05)), ["i_post"], cs((0, 0, 0))),
            # hinge about world Y at z=0.30; mate frame given in each part's own frame
            mate("f2", "Shoulder", "REVOLUTE", ["i_post"], cs((0, 0, 0.25), z=(0, 1, 0)),
                 ["i_arm"], cs((0, 0, 0), z=(0, 1, 0))),
            mate("f3", "Carriage slide", "SLIDER", ["i_arm"], cs((0.2, 0, 0), z=(1, 0, 0), x=(0, 1, 0)),
                 ["i_slide"], cs((0, 0, 0), z=(1, 0, 0), x=(0, 1, 0))),
        ],
    },
    "subAssemblies": [],
    "parts": [],
}
FEATURES = {"features": [{"message": {"featureId": "f2", "parameters": [
    {"message": {"parameterId": "limitsEnabled", "value": True}},
    {"message": {"parameterId": "limitAxialZMin", "expression": "-90 deg"}},
    {"message": {"parameterId": "limitAxialZMax", "expression": "90 deg"}}]}},
    # sliders keep translation limits in limitZ* (the limitAxialZ* rotation fields are present but 0)
    {"message": {"featureId": "f3", "parameters": [
        {"message": {"parameterId": "limitsEnabled", "value": True}},
        {"message": {"parameterId": "limitZMin", "expression": "0 in"}},
        {"message": {"parameterId": "limitZMax", "expression": "4.5 in"}},
        {"message": {"parameterId": "limitAxialZMin", "expression": "0 deg"}},
        {"message": {"parameterId": "limitAxialZMax", "expression": "0 deg"}}]}}]}
BOXES = {"JA": (0.2, 0.2, 0.05), "JB": (0.04, 0.04, 0.25), "JC": (0.3, 0.04, 0.04), "JD": (0.05, 0.05, 0.05)}


class FakeClient:
    def get(self, path, params=None, binary=False):
        if path.endswith("/features"):
            return FEATURES
        if "/assemblies/" in path:
            return ASSEMBLY
        pid = path.split("/partid/")[1].split("/")[0]
        ext = BOXES[pid]
        box = trimesh.creation.box(extents=ext)
        box.apply_translation([ext[0] / 2 if pid == "JC" else 0, 0, ext[2] / 2 if pid in ("JA", "JB") else 0])
        if path.endswith("/stl"):
            buf = io.BytesIO()
            box.export(buf, file_type="stl")
            return buf.getvalue()
        box.density = 2700.0
        return {"bodies": {pid: {"hasMass": True, "mass": [box.mass, 0, 0], "volume": [box.volume, 0, 0],
                                 "centroid": list(box.center_mass) + [0] * 6,
                                 "inertia": list(np.array(box.moment_inertia).ravel()) + [0] * 12}}}


def test_onshape_mates_become_links_and_joints():
    r = frontends.build_from_onshape({"source": URL}, None, client=FakeClient())
    assert sorted(r.links) == ["arm", "base_link", "carriage"]  # plate + post fastened into one link
    sh = r.joints["shoulder"]
    assert (sh.type, sh.parent, sh.child) == ("revolute", "base_link", "arm")
    np.testing.assert_allclose(sh.axis, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(sh.origin, [0, 0, 0.30], atol=1e-12)
    np.testing.assert_allclose([sh.lower, sh.upper], [-np.pi / 2, np.pi / 2])
    sl = r.joints["carriage_slide"]
    assert (sl.type, sl.parent, sl.child) == ("prismatic", "arm", "carriage")
    np.testing.assert_allclose(sl.axis, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(sl.origin, [0.2, 0, 0.30], atol=1e-12)
    # Onshape limits describe entity 0 relative to entity 1; the carriage is entity 1, so they flip
    np.testing.assert_allclose([sl.lower, sl.upper], [-4.5 * 0.0254, 0.0])
    base = r.links["base_link"]
    assert abs(base.mass - 2700 * (0.2 * 0.2 * 0.05 + 0.04 * 0.04 * 0.25)) < 1e-6


def test_quantity_parsing():
    assert abs(frontends.parse_quantity("90 deg") - np.pi / 2) < 1e-12
    assert abs(frontends.parse_quantity("-12.5 mm") + 0.0125) < 1e-12
    assert frontends.parse_quantity("1 in") == 0.0254
    assert frontends.parse_url(URL)["host"] == "team.onshape.com"


def test_onshape_robot_compiles_and_loads(tmp_path):
    import mujoco

    from cad2urdf import geometry, writers

    r = frontends.build_from_onshape({"source": URL}, None, client=FakeClient())
    geometry.build_collisions(r, with_metrics=False)
    writers.export_meshes(r, tmp_path / "meshes")
    writers.write_mjcf(r, tmp_path / "mjcf" / "r.xml", meshdir="../meshes")
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "r.xml"))
    d = mujoco.MjData(m)
    d.qpos[0] = np.pi / 2  # shoulder
    mujoco.mj_kinematics(m, d)
    # carriage frame starts 0.2 m along the arm (+X); rotating +90 deg about +Y takes +X to -Z
    np.testing.assert_allclose(d.body("carriage").xpos, [0, 0, 0.10], atol=1e-9)


class DofFakeClient(FakeClient):
    """Same parts, but using the dof_ convention: an alignment mate between links must be ignored,
    a pattern copy must stick to its seed, and a mate to the assembly origin must be skipped."""

    def get(self, path, params=None, binary=False):
        if path.endswith("/features"):
            return {"features": []}
        if "/assemblies/" in path and not path.endswith("/features"):
            a = json.loads(json.dumps(ASSEMBLY))
            ra = a["rootAssembly"]
            ra["instances"].append(part("i_copy", "Post copy", "JB"))
            ra["occurrences"].append({"path": ["i_copy"], "transform": T((0.1, 0, 0.05))})
            ra["patterns"] = [{"id": "P1", "suppressed": False, "seedToPatternInstances": {"i_post": ["i_copy"]}}]
            f = ra["features"]
            f[1]["featureData"]["name"] = "dof_shoulder_inv"
            f[2]["featureData"]["name"] = "Carriage slide"  # not dof_: must NOT become a joint
            f.append(mate("f4", "Parallel 1", "PARALLEL", ["i_arm"], cs((0, 0, 0)), ["i_plate"], cs((0, 0, 0))))
            f.append({"id": "f5", "featureType": "mate", "suppressed": False, "featureData": {
                "name": "Fastened origin", "mateType": "FASTENED",
                "matedEntities": [{"matedOccurrence": [], "matedCS": cs((0, 0, 0))},
                                  {"matedOccurrence": ["i_plate"], "matedCS": cs((0, 0, 0))}]}})
            return a
        return super().get(path, params, binary)


def test_dof_convention_patterns_and_origin_mates():
    r = frontends.build_from_onshape({"source": URL}, None, client=DofFakeClient())
    assert list(r.joints) == ["shoulder"]  # only the dof_ mate; "_inv" stripped from the name
    np.testing.assert_allclose(r.joints["shoulder"].axis, [0, -1, 0], atol=1e-12)  # _inv flips the axis
    assert r.joints["shoulder"].child == "arm"
    base = {p.name.split("/")[1] for p in r.links["base_link"].parts}
    assert {"post", "post_copy"} <= base  # pattern copy rigid with its seed
    assert any("mated to the origin" in x for x in r.review)


class ClosingFakeClient(FakeClient):
    """A closing_ mate (onshape-to-robot's loop-closure convention) between the carriage and the base."""

    def get(self, path, params=None, binary=False):
        if "/assemblies/" in path and not path.endswith("/features"):
            a = json.loads(json.dumps(ASSEMBLY))
            a["rootAssembly"]["features"].append(
                mate("f9", "closing_carriage", "REVOLUTE", ["i_slide"], cs((0, 0, 0), z=(0, 1, 0)),
                     ["i_plate"], cs((0.2, 0, 0.30), z=(0, 1, 0))))
            return a
        return super().get(path, params, binary)


def test_closing_mate_becomes_loop_constraint(tmp_path):
    import mujoco

    from cad2urdf import geometry, writers

    spec = {"source": URL, "actuators": {"default": {"kind": "position", "kp": 50, "kv": 1}}}
    r = frontends.build_from_onshape(spec, None, client=ClosingFakeClient())
    assert sorted(r.joints) == ["carriage_slide", "shoulder"]  # the closing mate is not a tree joint
    (c,) = r.closures
    assert {c["link1"], c["link2"]} == {"carriage", "base_link"}
    assert r.joints["shoulder"].actuator.get("kind") == "position"  # on the base: the motor
    assert r.joints["carriage_slide"].actuator == {"kind": "none"}  # inside the loop: passive
    geometry.build_collisions(r, with_metrics=False)
    writers.export_meshes(r, tmp_path / "meshes")
    writers.write_mjcf(r, tmp_path / "mjcf" / "r.xml", meshdir="../meshes")
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "r.xml"))
    assert m.neq == 1 and m.eq_type[0] == mujoco.mjtEq.mjEQ_CONNECT and m.nu == 1
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b1, b2 = m.eq_obj1id[0], m.eq_obj2id[0]
    p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[0, 0:3]
    p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[0, 3:6]
    np.testing.assert_allclose(p1, p2, atol=1e-9)  # assembled pose satisfies the loop
    np.testing.assert_allclose(p1, [0.2, 0, 0.30], atol=1e-9)


def test_joint_name_conventions_for_actuators():
    spec = {"actuators": {"default": {"kind": "position", "kp": 100, "kv": 5}, "wheel2_passive3": {"kind": "position", "kp": 1}}}
    assert frontends._named_actuator("wheel1_passive7", spec) == {"kind": "none"}  # free roller
    assert frontends._named_actuator("wheel1_speed", spec)["kind"] == "velocity"  # drive wheel
    assert frontends._named_actuator("left_knee", spec)["kind"] == "position"
    assert frontends._named_actuator("wheel2_passive3", spec)["kp"] == 1  # an explicit spec entry wins


class PlanarFakeClient(FakeClient):
    """The carriage rides a PLANAR mate (slide x, slide y, spin z) instead of a slider."""

    def get(self, path, params=None, binary=False):
        if path.endswith("/features"):
            lim = [("limitsEnabled", True), ("limitXMin", "0 in"), ("limitXMax", "2 in"), ("limitYMin", "-1 in"),
                   ("limitYMax", "1 in")]
            return {"features": [{"message": {"featureId": "f3", "parameters": [
                {"message": {"parameterId": k, ("value" if k == "limitsEnabled" else "expression"): v}}
                for k, v in lim]}}]}
        if "/assemblies/" in path:
            a = json.loads(json.dumps(ASSEMBLY))
            a["rootAssembly"]["features"][2] = mate("f3", "Carriage slide", "PLANAR", ["i_arm"], cs((0.2, 0, 0)),
                                                    ["i_slide"], cs((0, 0, 0)))
            return a
        return super().get(path, params, binary)


def test_planar_mate_becomes_two_slides_and_a_spin():
    r = frontends.build_from_onshape({"source": URL}, None, client=PlanarFakeClient())
    x, y, z = (r.joints[f"carriage_slide_{k}"] for k in "xyz")
    assert [(j.type, j.parent, j.child) for j in (x, y, z)] == [
        ("prismatic", "arm", "carriage_slide_x_link"), ("prismatic", "carriage_slide_x_link", "carriage_slide_y_link"),
        ("continuous", "carriage_slide_y_link", "carriage")]
    np.testing.assert_allclose(x.axis, [1, 0, 0])
    np.testing.assert_allclose(y.axis, [0, 1, 0])
    np.testing.assert_allclose(z.axis, [0, 0, 1])
    np.testing.assert_allclose([x.lower, x.upper], [-2 * 0.0254, 0.0], atol=1e-9)  # carriage is entity 1: flipped
    assert not r.links["carriage_slide_x_link"].parts  # massless carrier
    assert all(j.actuator == {"kind": "none"} for j in (x, y, z))


class ScrewFakeClient(FakeClient):
    """A screw on the arm with a REVOLUTE mate: it is fixed to the arm, not a joint."""

    def get(self, path, params=None, binary=False):
        if "/assemblies/" in path and not path.endswith("/features"):
            a = json.loads(json.dumps(ASSEMBLY))
            a["rootAssembly"]["instances"].append(part("i_screw", "M3x8_SHCS", "JE"))
            a["rootAssembly"]["occurrences"].append({"path": ["i_screw"], "transform": T((0.1, 0, 0.32))})
            a["rootAssembly"]["features"].append(
                mate("f8", "screw turn", "REVOLUTE", ["i_arm"], cs((0.1, 0, 0.02)), ["i_screw"], cs((0, 0, 0))))
            return a
        if "/partid/JE/" in path:
            return super().get(path.replace("/JE/", "/JD/"), params, binary)
        return super().get(path, params, binary)


def test_screw_mates_are_fixed_not_joints():
    r = frontends.build_from_onshape({"source": URL}, None, client=ScrewFakeClient())
    assert sorted(r.joints) == ["carriage_slide", "shoulder"]
    assert any("m3x8_shcs" in p.name for p in r.links["arm"].parts)


def test_planar_chain_loads_and_moves_in_mujoco(tmp_path):
    import mujoco

    from cad2urdf import geometry, writers

    r = frontends.build_from_onshape({"source": URL}, None, client=PlanarFakeClient())
    geometry.build_collisions(r, with_metrics=False)
    writers.export_meshes(r, tmp_path / "meshes")
    writers.write_mjcf(r, tmp_path / "mjcf" / "r.xml", meshdir="../meshes")
    writers.write_urdf(r, tmp_path / "r.urdf")
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "r.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    start = d.body("carriage").xpos.copy()
    qadr = {m.joint(i).name: m.jnt_qposadr[i] for i in range(m.njnt)}
    d.qpos[qadr["carriage_slide_x"]] = -0.03
    d.qpos[qadr["carriage_slide_y"]] = -0.02
    mujoco.mj_kinematics(m, d)
    # the planar axes are the mate's x and y; the arm is rotated none at q=0, so the carriage slides in world x, y
    np.testing.assert_allclose(d.body("carriage").xpos - start, [-0.03, -0.02, 0.0], atol=1e-9)


class PlanarPairClient(FakeClient):
    """The arm is attached to the post by planar mates with the given face normals (no shoulder mate)."""

    normals: list = []

    def get(self, path, params=None, binary=False):
        if path.endswith("/features"):
            return {"features": []}
        if "/assemblies/" in path:
            a = json.loads(json.dumps(ASSEMBLY))
            feats = [a["rootAssembly"]["features"][0], a["rootAssembly"]["features"][2]]
            for i, n in enumerate(self.normals):
                x = (0, 1, 0) if abs(n[0]) > 0.9 else (1, 0, 0)
                feats.append(mate(f"p{i}", f"Planar {i + 1}", "PLANAR", ["i_post"], cs((0, 0, 0.25), z=n, x=x),
                                  ["i_arm"], cs((0, 0, 0), z=n, x=x)))
            a["rootAssembly"]["features"] = feats
            return a
        return super().get(path, params, binary)


class ThreePlanar(PlanarPairClient):
    normals = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


class TwoPlanar(PlanarPairClient):
    normals = [(0, 1, 0), (0, 0, 1)]


def test_three_planar_mates_are_a_rigid_joint():
    r = frontends.build_from_onshape({"source": URL}, None, client=ThreePlanar())
    assert sorted(r.joints) == ["carriage_slide"]  # the arm is welded to the base, not 3 free DOF
    assert any("arm" in p.name for p in r.links["base_link"].parts)
    assert any("independent normals" in x for x in r.review)


def test_two_planar_mates_leave_one_slide():
    r = frontends.build_from_onshape({"source": URL}, None, client=TwoPlanar())
    j = r.joints["arm_planar_1"]  # a default mate name is prefixed with the link it moves
    assert j.type == "prismatic" and (j.parent, j.child) == ("base_link", "arm")
    np.testing.assert_allclose(np.abs(j.axis), [1, 0, 0], atol=1e-9)  # the planes' intersection line
    assert not any(n.startswith("arm_planar_1_") for n in r.joints)


def test_default_mate_names_are_made_unique_and_meaningful():
    from cad2urdf.frontends import GENERIC_MATE

    for n in ("Revolute 4", "Slider 1", "Pin slot 2", "planar_3", "Cylindrical 1"):
        assert GENERIC_MATE.match(n), n
    for n in ("Shoulder", "dof_wrist", "Revolute hip", "Carriage slide"):
        assert not GENERIC_MATE.match(n), n


class OriginClient(FakeClient):
    """The arm turns about a world axis through an *origin* mate (second entity is the assembly origin),
    and a FOV cone solid sits on the plate."""

    def get(self, path, params=None, binary=False):
        if "/assemblies/" in path and not path.endswith("/features"):
            a = json.loads(json.dumps(ASSEMBLY))
            ra = a["rootAssembly"]
            ra["instances"].append(part("i_fov", "LidarFov", "JA"))
            ra["occurrences"].append({"path": ["i_fov"], "transform": T((0, 0, 0.5))})
            ra["features"][1] = {"id": "f2", "featureType": "mate", "suppressed": False, "featureData": {
                "name": "Yaw", "mateType": "REVOLUTE", "matedEntities": [
                    {"matedOccurrence": ["i_arm"], "matedCS": cs((0, 0, 0), z=(0, 0, 1))},
                    {"matedOccurrence": [], "matedCS": cs((0, 0, 0.30), z=(0, 0, 1))}]}}
            return a
        return super().get(path, params, binary)


def test_origin_mate_moves_the_body_against_the_ground_and_fov_solids_are_skipped():
    r = frontends.build_from_onshape({"source": URL}, None, client=OriginClient())
    yaw = r.joints["yaw"]
    assert (yaw.type, yaw.parent, yaw.child) == ("revolute", "base_link", "arm")  # limits come from the fake feature list
    np.testing.assert_allclose(yaw.axis, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(yaw.origin, [0, 0, 0.30], atol=1e-12)  # where the origin frame and the arm frame meet
    assert any("stands in for it" in x for x in r.review)
    assert not any("lidarfov" in p.name for l in r.links.values() for p in l.parts)


def test_non_physical_names():
    from cad2urdf.util import is_non_physical

    for n in ("FOVCone <1>", "LidarFov <1>", "camera_frustum", "Vision Cone"):
        assert is_non_physical(n), n
    for n in ("Referee Mount", "Cone Bearing", "Shaft", "Safety Cover"):
        assert not is_non_physical(n), n


def test_joints_only_welds_everything_else_and_keeps_names():
    spec = {"source": URL, "joints_only": ["shoulder"]}
    r = frontends.build_from_onshape(spec, None, client=FakeClient())
    assert list(r.joints) == ["shoulder"]  # the carriage slide is welded
    assert sorted(r.links) == ["arm", "base_link"]
    assert any("carriage" in p.name for p in r.links["arm"].parts)
    assert any("kept 1 of 2" in x for x in r.review)
    both = frontends.build_from_onshape({"source": URL, "joints_only": ["*"]}, None, client=FakeClient())
    assert sorted(both.joints) == ["carriage_slide", "shoulder"]
    none = frontends.build_from_onshape({"source": URL, "joints_only": ["nothing_matches"]}, None,
                                        client=FakeClient())
    assert not none.joints and any("no joint matches" in x for x in none.review)
