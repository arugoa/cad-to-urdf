from pathlib import Path

import pytest

from robot_model_ci.checks import validate_path
from robot_model_ci.compare import compare_dirs

FIXTURES = Path(__file__).resolve().parents[3] / "examples" / "robot-model-ci-fixtures"


def test_valid_fixture_passes():
    report = validate_path(FIXTURES / "baseline")
    assert report["status"] == "passed"
    assert report["summary"]["models"] == 2
    assert not [issue for issue in report["issues"] if issue["severity"] == "ERROR"]
    assert not [issue for issue in report["issues"] if issue["code"].startswith("cross_format.")]


def test_broken_fixture_reports_missing_mesh_and_limit():
    report = validate_path(FIXTURES / "broken")
    codes = {issue["code"] for issue in report["issues"]}
    assert report["status"] == "failed"
    assert "asset.missing" in codes
    assert "urdf.limit_order" in codes


def test_compare_detects_removed_joint():
    report = compare_dirs(FIXTURES / "baseline", FIXTURES / "candidate")
    assert report["status"] == "failed"
    assert report["summary"]["breaking_changes"] >= 1
    assert any(c["code"] == "joint.removed" for c in report["changes"])


def test_cross_format_mismatch_is_reported():
    report = validate_path(FIXTURES / "candidate")
    assert any(issue["code"].startswith("cross_format.joint_missing") for issue in report["issues"])


# --- regressions found by running the checker on real cad2urdf output -----------------------------

REPO = Path(__file__).resolve().parents[3]


def _codes(report):
    return {(i["severity"], i["code"]) for i in report["issues"]}


def _urdf(tmp_path, body, inertia='ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"'):
    link = f'<inertial><mass value="1"/><inertia {inertia}/></inertial>'
    (tmp_path / "r.urdf").write_text(
        f'<robot name="r"><link name="a">{link}</link><link name="b">{link}</link>{body}</robot>')
    return tmp_path


def test_mjcf_compiler_meshdir_is_honoured(tmp_path):
    # cad2urdf writes <compiler meshdir="../meshes"> and file="visual/x.stl"
    (tmp_path / "meshes" / "visual").mkdir(parents=True)
    (tmp_path / "meshes" / "visual" / "a.stl").write_bytes(b"solid a\nendsolid a\n")
    (tmp_path / "mjcf").mkdir()
    (tmp_path / "mjcf" / "m.xml").write_text(
        '<mujoco model="m"><compiler meshdir="../meshes"/><asset><mesh name="a" file="visual/a.stl"/></asset>'
        '<worldbody/></mujoco>')
    assert validate_path(tmp_path)["status"] == "passed"
    (tmp_path / "meshes" / "visual" / "a.stl").unlink()
    assert ("ERROR", "asset.missing") in _codes(validate_path(tmp_path))  # still catches a really-missing mesh


def test_mjcf_inline_mesh_needs_no_file(tmp_path):
    (tmp_path / "m.xml").write_text(
        '<mujoco model="m"><asset><mesh name="p" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/></asset><worldbody/></mujoco>')
    assert validate_path(tmp_path)["status"] == "passed"


def test_continuous_joint_may_omit_limit(tmp_path):
    # cad2urdf omits <limit> for continuous joints with no effort/velocity; URDF allows it
    d = _urdf(tmp_path, '<joint name="j" type="continuous"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/></joint>')
    assert validate_path(d)["status"] == "passed"
    d = _urdf(tmp_path, '<joint name="j" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/></joint>')
    assert ("ERROR", "urdf.limit_missing") in _codes(validate_path(d))  # revolute still needs one


def test_inertia_triangle_inequality(tmp_path):
    fixed = '<joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>'
    bad = _urdf(tmp_path, fixed, inertia='ixx="1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"')
    assert ("ERROR", "urdf.inertia_triangle") in _codes(validate_path(bad))
    # a thin plate sits exactly on the boundary (A + B == C) and is physical
    plate = _urdf(tmp_path, fixed, inertia='ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2"')
    assert validate_path(plate)["status"] == "passed"
    # off-diagonal terms: principal moments are 1, 2, 3 (rotated about z); still valid
    rot = _urdf(tmp_path, fixed, inertia='ixx="1.5" ixy="0.5" ixz="0" iyy="1.5" iyz="0" izz="3"')
    assert validate_path(rot)["status"] == "passed"


def test_principal_moments_match_numpy():
    np = pytest.importorskip("numpy")
    from robot_model_ci.checks import _principal_moments

    rng = np.random.default_rng(0)
    for _ in range(500):
        a = rng.normal(size=(3, 3))
        i = a @ a.T
        got = _principal_moments(i[0, 0], i[0, 1], i[0, 2], i[1, 1], i[1, 2], i[2, 2])
        np.testing.assert_allclose(got, np.linalg.eigvalsh(i), rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not (REPO / "examples" / "arm4" / "output").is_dir(), reason="arm4 sample output not present")
def test_committed_arm4_output_validates():
    report = validate_path(REPO / "examples" / "arm4" / "output")
    assert report["status"] == "passed", [i for i in report["issues"] if i["severity"] == "ERROR"][:3]
    assert report["summary"]["models"] >= 3  # URDF, Gazebo URDF, MJCF
