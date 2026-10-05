"""Shared fixtures. The code classifies parts by name only through the spec's ``part_classes``; the tests that
need classes load the skill's pattern library, the same file the agent hands to ``--part-classes``."""

from pathlib import Path

import pytest
import yaml

from cad2urdf import util

LIBRARY = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "cad2sim" / "part_classes.yaml"


@pytest.fixture
def part_classes():
    util.set_part_classes(yaml.safe_load(LIBRARY.read_text()))
    yield
    util.set_part_classes(None)  # nothing leaks into the next test


@pytest.fixture(scope="session")
def arm4_out(tmp_path_factory):
    """(robot, folder) with the URDF, MJCF and USD of the arm4 sample, written the way the compiler writes them."""
    from cad2urdf import geometry, model, usd_asset, writers

    robot = model.build(Path(__file__).parents[1] / "examples" / "arm4" / "robot_spec.yaml")
    geometry.build_collisions(robot, with_metrics=False)
    out = tmp_path_factory.mktemp("arm4_out")
    writers.export_meshes(robot, out / "meshes")
    writers.write_urdf(robot, out / f"{robot.name}.urdf")
    writers.write_mjcf(robot, out / "mjcf" / f"{robot.name}.xml", meshdir="../meshes")
    usd_asset.write_usd(robot, out / "usd")
    return robot, out
