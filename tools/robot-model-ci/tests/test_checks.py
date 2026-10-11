from pathlib import Path

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
