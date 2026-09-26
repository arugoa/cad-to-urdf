"""Onshape front end against a fake API that returns the real response shapes (no network, no keys)."""

import io

import numpy as np
import trimesh

from cad2urdf import onshape

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
    {"message": {"parameterId": "limitAxialZMax", "expression": "90 deg"}}]}}]}
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
    r = onshape.build_from_onshape({"source": URL}, None, client=FakeClient())
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
    assert any("slider without limits" in x for x in r.review)
    base = r.links["base_link"]
    assert abs(base.mass - 2700 * (0.2 * 0.2 * 0.05 + 0.04 * 0.04 * 0.25)) < 1e-6


def test_quantity_parsing():
    assert abs(onshape.parse_quantity("90 deg") - np.pi / 2) < 1e-12
    assert abs(onshape.parse_quantity("-12.5 mm") + 0.0125) < 1e-12
    assert onshape.parse_quantity("1 in") == 0.0254
    assert onshape.parse_url(URL)["host"] == "team.onshape.com"


def test_onshape_robot_compiles_and_loads(tmp_path):
    import mujoco

    from cad2urdf import collision, mjcf, urdf

    r = onshape.build_from_onshape({"source": URL}, None, client=FakeClient())
    collision.build_collisions(r, with_metrics=False)
    urdf.export_meshes(r, tmp_path / "meshes")
    mjcf.write_mjcf(r, tmp_path / "mjcf" / "r.xml", meshdir="../meshes")
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "r.xml"))
    d = mujoco.MjData(m)
    d.qpos[0] = np.pi / 2  # shoulder
    mujoco.mj_kinematics(m, d)
    # carriage frame starts 0.2 m along the arm (+X); rotating +90 deg about +Y takes +X to -Z
    np.testing.assert_allclose(d.body("carriage").xpos, [0, 0, 0.10], atol=1e-9)
