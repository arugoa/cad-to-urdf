"""Router, deterministic STEP draft, and URDF ingest (round trip)."""

from pathlib import Path

import numpy as np
import pytest
import yourdfpy

from cad2urdf import model, routes, urdf
from cad2urdf.draft import draft_spec

ARM4 = Path(__file__).parents[1] / "examples" / "arm4"


def test_every_cad_format_sim_has_a_plan():
    for (cad, fmt) in routes.FRONT_ENDS:
        for sim in routes.SIMS:
            text = routes.plan(cad, fmt, sim)
            assert "Front end" in text and "Finish" in text


def test_unknown_route_is_rejected():
    with pytest.raises(KeyError):
        routes.front_end("creo", "urdf-export")


def test_draft_recovers_arm4_structure():
    spec, review = draft_spec(ARM4 / "cad" / "arm4.step")
    groups = sorted(sorted(v) for v in spec["links"].values())
    assert ["finger_left"] in groups and ["finger_right"] in groups
    assert sorted(["shoulder_pin", "upper_arm_beam"]) in groups
    assert sorted(f"base_bolt_{i}" for i in range(1, 5)) == [p for g in groups for p in g if "bolt" in p]
    types = sorted(j["type"] for j in spec["joints"].values())
    assert types == ["prismatic"] * 2 + ["revolute"] * 4
    assert any("cylindrical" in r for r in review)


def test_urdf_round_trip_preserves_kinematics(tmp_path):
    src = model.build(ARM4 / "robot_spec.yaml")
    urdf.export_meshes(src, tmp_path / "a" / "meshes")
    urdf.write_urdf(src, tmp_path / "a" / "arm4.urdf")
    spec = tmp_path / "rt.yaml"
    spec.write_text(f"robot: rt\nsource: {tmp_path / 'a' / 'arm4.urdf'}\n")
    rt = model.build(spec)
    urdf.export_meshes(rt, tmp_path / "b" / "meshes")
    urdf.write_urdf(rt, tmp_path / "b" / "rt.urdf")
    a, b = yourdfpy.URDF.load(str(tmp_path / "a" / "arm4.urdf")), yourdfpy.URDF.load(str(tmp_path / "b" / "rt.urdf"))
    q = {n: 0.3 if "finger" not in n else 0.004 for n in a.actuated_joint_names}
    a.update_cfg(q)
    b.update_cfg(q)
    for link in a.link_map:
        np.testing.assert_allclose(a.get_transform(link), b.get_transform(link), atol=1e-9)
    assert rt.joints["finger_right"].mimic["joint"] == "finger_left"


def test_rotated_frames_survive_ingest(tmp_path):
    """A Y-up export with rotated joint frames: axes and FK must be preserved."""
    (tmp_path / "r.urdf").write_text("""<robot name="r">
      <link name="a"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="b"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
        <visual><geometry><box size="0.1 0.2 0.3"/></geometry></visual></link>
      <joint name="j" type="revolute"><parent link="a"/><child link="b"/>
        <origin xyz="0.1 0.2 0.3" rpy="0.3 -0.7 1.1"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>""")
    (tmp_path / "s.yaml").write_text(f"source: {tmp_path / 'r.urdf'}\n")
    r = model.build(tmp_path / "s.yaml")
    urdf.write_urdf(r, tmp_path / "out.urdf")
    x, y = yourdfpy.URDF.load(str(tmp_path / "r.urdf")), yourdfpy.URDF.load(str(tmp_path / "out.urdf"), load_meshes=False)
    for q in (-0.8, 0.0, 0.6):
        x.update_cfg({"j": q})
        y.update_cfg({"j": q})
        np.testing.assert_allclose(x.get_transform("b"), y.get_transform("b"), atol=1e-9)
