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
