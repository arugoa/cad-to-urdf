"""The asset tester must pass a good asset and catch the mistakes it exists for."""

import re
import shutil
from pathlib import Path

import pytest

from cad2urdf import asset_test


def _run(folder: Path, *extra) -> dict:
    code = asset_test.main([str(folder), "--seconds", "1", *extra])
    import json

    result = json.loads((folder / "asset_test.json").read_text())
    assert (code == 0) == result["ok"]
    return result


def _errors(result) -> list[str]:
    return [f"{f['where']}: {f['what']}" for f in result["flags"] if f["severity"] == "error"]


@pytest.fixture
def copy(arm4_out, tmp_path):
    robot, out = arm4_out
    shutil.copytree(out, tmp_path / "asset")
    return robot, tmp_path / "asset"


def test_a_good_asset_passes(copy):
    robot, folder = copy
    result = _run(folder)
    assert result["ok"], _errors(result)
    assert result["mujoco"]["bang_bang"]["exploded"] is False
    assert result["audit"]["joints"] == len(robot.moving_joints())


def test_zero_newton_armature_is_caught(copy):
    robot, folder = copy
    layer = folder / "usd" / "configuration" / "Physics" / "mujoco.usda"
    layer.write_text(re.sub(r"newton:armature = [0-9.e-]+", "newton:armature = 0", layer.read_text()))
    assert any("newton:armature" in e for e in _errors(_run(folder)))


def test_wrong_angular_gain_units_are_caught(copy):
    robot, folder = copy
    layer = folder / "usd" / "configuration" / "Physics" / "physics.usda"
    text = layer.read_text()
    assert "drive:angular:physics:stiffness = 3.4906" in text
    layer.write_text(text.replace("drive:angular:physics:stiffness = 3.4906", "drive:angular:physics:stiffness = 200.0"))
    assert any("stiffness" in e for e in _errors(_run(folder)))


def test_armature_below_the_stability_floor_is_caught(copy):
    robot, folder = copy
    xml = next((folder / "mjcf").glob("*.xml"))
    xml.write_text(re.sub(r'armature="[0-9.e-]+"', 'armature="1e-05"', xml.read_text()))
    errors = _errors(_run(folder))
    assert any("stability floor" in e for e in errors) and any("armature" in e for e in errors)


def test_wrong_limits_are_caught(copy):
    robot, folder = copy
    layer = folder / "usd" / "configuration" / "Physics" / "physics.usda"
    text = layer.read_text()
    layer.write_text(re.sub(r"physics:upperLimit = [0-9.]+", "physics:upperLimit = 1.0", text, count=1))
    assert any("limits" in e for e in _errors(_run(folder)))


def test_a_missing_usd_is_a_warning_not_a_crash(copy):
    robot, folder = copy
    shutil.rmtree(folder / "usd")
    result = _run(folder)
    assert any(f["where"] == "usd" and f["severity"] == "warning" for f in result["flags"])
