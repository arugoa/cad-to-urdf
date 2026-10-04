"""Fast regression tests on the arm4 sample (no CoACD, ~20 s)."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from cad2urdf import geometry, model, step, writers
from cad2urdf.model import CollisionGeom

SPEC = Path(__file__).parents[1] / "examples" / "arm4" / "robot_spec.yaml"


@pytest.fixture(scope="module")
def robot():
    r = model.build(SPEC)
    for link in r.links.values():  # primitives everywhere: fast and deterministic
        link.collisions = geometry.link_collisions(link, {"mode": "primitives", "min_part_fraction": 0.02}, 64)
    return r


def test_every_joint_found_geometrically(robot):
    by_pair = {(c.link_a, c.link_b): c for c in robot.candidates}
    assert len(by_pair) == 6
    yaw = by_pair[("base_link", "turret")]
    assert yaw.type_hint == "revolute"
    np.testing.assert_allclose(yaw.direction, [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(yaw.origin, [0, 0, 0.072], atol=1e-6)
    pitch = by_pair[("turret", "upper_arm")]
    assert pitch.type_hint == "revolute" and len(pitch.evidence) == 2  # pin through both clevis plates
    np.testing.assert_allclose(pitch.origin, [0, 0, 0.150], atol=1e-6)
    # a rail can slide or spin: geometry must flag it as ambiguous
    assert by_pair[("finger_left", "gripper_base")].type_hint == "cylindrical"


def test_spec_resolves_joint_frames(robot):
    j = robot.joints["finger_right"]
    assert j.type == "prismatic" and j.mimic["joint"] == "finger_left"
    np.testing.assert_allclose(j.axis, [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(robot.links["forearm"].origin, [0, 0, 0.4], atol=1e-6)


def test_mass_properties_are_physical(robot):
    for link in robot.links.values():
        assert model.check_inertia(link.name, link.inertia, link.mass) == []
    assert 5.0 < sum(l.mass for l in robot.links.values()) < 5.5


def test_parallel_axis_matches_analytic_box():
    """Two unit-density 1x1x1 m cubes side by side == one 2x1x1 box."""
    def cube(x):
        p = step.Part(name=f"c{x}", shape=None, mass=1.0, com=np.array([x, 0.0, 0.0]))
        p.inertia = np.eye(3) * (1.0 / 6.0)
        return p
    mass, com, inertia = model.combine_inertia([cube(-0.5), cube(0.5)], np.zeros(3))
    expected = np.diag([2 * (1 + 1) / 12, 2 * (4 + 1) / 12, 2 * (4 + 1) / 12])
    assert mass == 2.0
    np.testing.assert_allclose(com, 0, atol=1e-12)
    np.testing.assert_allclose(inertia, expected, atol=1e-12)


def test_primitives_fit_machined_links(robot):
    base = robot.links["base_link"]
    assert [g.kind for g in base.collisions] == ["box", "cylinder"]  # 4 bolts culled
    assert geometry.metrics(base, n=20000)["iou"] > 0.95


def test_hull_vertex_cap():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    capped = geometry._cap_vertices(sphere.convex_hull, 64)
    assert len(capped.vertices) <= 64
    assert abs(capped.volume - sphere.convex_hull.volume) / sphere.convex_hull.volume < 1e-6


def test_axis_aligned_boxes_have_zero_rpy():
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    g = geometry._axis_align_box(CollisionGeom("box", T, (1.0, 2.0, 3.0)))
    np.testing.assert_allclose(g.transform[:3, :3], np.eye(3))
    np.testing.assert_allclose(g.size, (3.0, 1.0, 2.0))


def test_urdf_loads_in_yourdfpy_and_mujoco(robot, tmp_path):
    import mujoco
    import yourdfpy

    writers.export_meshes(robot, tmp_path / "meshes")
    writers.write_urdf(robot, tmp_path / "arm4.urdf")
    r = yourdfpy.URDF.load(str(tmp_path / "arm4.urdf"))
    assert len(r.actuated_joint_names) == 5
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "arm4.urdf"))
    assert m.njnt == 6


def test_self_collision_matrix(robot, tmp_path):
    writers.export_meshes(robot, tmp_path / "meshes")
    disabled, _ = writers.collision_matrix(robot, tmp_path / "meshes", samples=300)
    adjacent = {p for p, r in disabled.items() if r == "Adjacent"}
    assert adjacent == {tuple(sorted((j.parent, j.child))) for j in robot.joints.values()}
    assert ("finger_left", "finger_right") in disabled  # limits keep the jaws apart


def test_bad_part_inertia_does_not_crash_the_link():
    from types import SimpleNamespace as NS

    good = NS(mass=1.0, com=np.zeros(3), inertia=np.eye(3) * 0.01)
    nan = NS(mass=2.0, com=np.zeros(3), inertia=np.full((3, 3), np.nan))
    mass, com, inertia = model.combine_inertia([good, nan], np.zeros(3))
    assert mass == 1.0 and np.isfinite(inertia).all()  # the unusable part is ignored
    assert model.combine_inertia([nan], np.zeros(3))[0] == 0.0
    assert model.check_inertia("x", np.full((3, 3), np.nan), 1.0)  # reported, not raised


def test_collision_metrics_repeat_exactly(robot):
    link = next(iter(robot.links.values()))
    geometry.build_collisions(robot, with_metrics=False)
    first = geometry.metrics(link)
    np.random.seed(123)  # whatever the global RNG state is, the result must not change
    assert geometry.metrics(link) == first
