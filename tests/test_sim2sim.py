"""sim2sim: the same scenario in several simulators, matched by joint name, scored against a reference."""

import json
import shutil

import numpy as np
import pytest

from cad2urdf import sim2sim


@pytest.fixture
def asset(arm4_out, tmp_path):
    robot, out = arm4_out
    shutil.copytree(out, tmp_path / "asset")
    return robot, tmp_path / "asset"


def _run(folder, sims):
    code = sim2sim.main([str(folder), "--sims", sims])
    return code, json.loads((folder / "sim2sim.json").read_text())


def test_contract_is_by_name_and_matches_the_mjcf(asset):
    robot, folder = asset
    c = sim2sim.Contract(folder)
    assert sorted(c.names) == sorted(j.name for j in robot.moving_joints() if not j.mimic and j.actuator.get("kind") == "position")
    assert np.all(c.kp > 0) and np.all(c.kv >= 0) and len(c.target(0.0)) == len(c.names)
    # the scenario: hold, a step to half range, a bang-bang square wave, a return
    assert np.allclose(c.target(0.5), 0) and not np.allclose(c.target(1.5), 0)
    wave = [c.target(sim2sim.HOLD + sim2sim.STEP + 0.05), c.target(sim2sim.HOLD + sim2sim.STEP + 0.25)]
    assert np.allclose(wave[0], -wave[1]) and np.allclose(c.target(5.9), 0)


def test_mujoco_is_the_reference_and_pybullet_gap_is_explained_by_armature(asset):
    robot, folder = asset
    code, result = _run(folder, "mujoco,pybullet")
    assert code == 0 and result["reference"] == "mujoco"
    assert result["sims"]["mujoco"]["finite"] and result["sims"]["pybullet"]["finite"]
    pb = result["sims"]["pybullet"]["vs_reference"]
    assert set(pb["rmse_per_joint"]) == set(result["joints"])
    # explicit damping needs kv*dt/I < 2: arm4's light joints break that without the armature, and the tool says so
    assert pb["unstable_without_armature"]
    assert any("not a bug" in f["what"] for f in result["flags"] if f["sim"] == "pybullet")


def test_newton_agrees_with_mujoco_on_the_same_asset(asset):
    pytest.importorskip("newton")
    robot, folder = asset
    code, result = _run(folder, "mujoco,newton")
    assert code == 0, result["flags"]
    cmp_ = result["sims"]["newton"]["vs_reference"]
    assert cmp_["rmse_over_range_max"] < sim2sim.RMSE_WARN  # the same solver family on the same dynamics
    assert cmp_["phase_rmse_over_range"]["step"] < 0.02


def test_a_simulator_that_cannot_run_is_an_error_flag(asset):
    robot, folder = asset
    shutil.rmtree(folder / "usd")  # newton needs the USD
    pytest.importorskip("newton")
    code, result = _run(folder, "mujoco,newton")
    assert code == 1 and any(f["severity"] == "error" and f["sim"] == "newton" for f in result["flags"])
