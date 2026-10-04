"""Fastener detection and the sphere ("bubble") collision fit."""

import numpy as np
import trimesh

from cad2urdf.geometry import fit_spheres
from cad2urdf.util import is_fastener


def test_fastener_names(part_classes):
    for n in ["M3x8_SHCS", "Hex Nut M4", "washer_5mm", "ISO 4762 M3 x 10 Socket Head Cap Screw", "heat_set_insert",
              "base/BHCS_M2", "Standoff_20mm"]:
        assert is_fastener(n), n
    for n in ["Base_08q_0", "STS3215_03a#1", "Rotation_Pitch_08i", "Motor_Mount", "nutrition_plate", "Chassis",
              "Lead_Screw_8mm_Threaded_Rod_Trapezoidal_Z_Axis", "Fitting_brass_Z_Axis_Acme_Screw", "Ball_Screw_SFU1204"]:
        assert not is_fastener(n), n


def test_spheres_cover_a_box():
    box = trimesh.creation.box(extents=(0.2, 0.05, 0.05))
    spheres = fit_spheres(box, max_spheres=12)
    assert 1 < len(spheres) <= 12
    pts = trimesh.sample.sample_surface(box, 2000, seed=0)[0] * 0.98  # just inside the surface
    d = np.min([np.linalg.norm(pts - c, axis=1) - r for c, r in spheres], axis=0)
    assert (d <= 0).mean() > 0.9  # the bubbles cover the part
    for c, r in spheres:  # and don't balloon far past it
        assert r < 0.05


def test_thin_plates_keep_a_collision_shape():
    from types import SimpleNamespace as NS

    from cad2urdf.geometry import _significant

    def part(m):
        return NS(mesh=m, volume=m.volume)

    block = part(trimesh.creation.box(extents=(0.05, 0.05, 0.05)))
    plate = part(trimesh.creation.box(extents=(0.12, 0.08, 0.001)))  # big but 0.1% of the volume
    bolt = part(trimesh.creation.box(extents=(0.004, 0.004, 0.01)))
    link = NS(parts=[block, plate, bolt], mesh=lambda: trimesh.util.concatenate([p.mesh for p in (block, plate, bolt)]))
    kept = _significant(link, link.parts, 0.02)
    assert plate in kept and block in kept and bolt not in kept


def test_servo_dynamics_keep_light_links_stable_and_speed_limited():
    from types import SimpleNamespace as NS

    from cad2urdf.writers import TIMESTEP, _servo_dynamics

    j = NS(actuator={"kind": "position", "kp": 100.0}, damping=0.1, armature=0.001, effort=10.0, velocity=5.0)
    damping, armature = _servo_dynamics(j, TIMESTEP)
    assert armature >= 16 * 100 * TIMESTEP**2 and np.sqrt(100 / armature) * TIMESTEP < 0.3
    assert damping == 2.0  # effort / velocity: terminal speed is the velocity limit
    passive = NS(actuator={"kind": "none"}, damping=0.1, armature=0.001, effort=10.0, velocity=5.0)
    assert _servo_dynamics(passive, TIMESTEP) == (0.1, 0.001)


def test_fastener_removal_with_several_distinct_parts(part_classes):
    from cad2urdf.geometry import simplify_robot
    from cad2urdf.model import Link, Robot
    from cad2urdf.step import Part

    def part(name, size):
        m = trimesh.creation.box(extents=size)
        return Part(name=name, shape=None, mesh=m, volume=m.volume, mass=1.0, com=np.zeros(3), inertia=np.eye(3))

    link = Link("base")
    link.parts = [part("plate", (0.2, 0.1, 0.01)), part("M3x8_SHCS", (0.003, 0.003, 0.008)),
                  part("M3_nut", (0.005, 0.005, 0.002)), part("arm", (0.1, 0.02, 0.02))]
    link.visuals = {"mesh": trimesh.util.concatenate([p.mesh for p in link.parts])}
    link.visual_materials = {"mesh": "grey"}
    robot = Robot("r", {}, {"base": link}, {}, [], "base")
    rep = simplify_robot(robot)  # used to raise "truth value of an array is ambiguous"
    assert rep["base"]["dropped"] == 2 and [p.name for p in link.parts] == ["plate", "arm"]


def test_many_small_parts_still_get_collision_candidates():
    from types import SimpleNamespace as NS

    from cad2urdf.geometry import _significant

    rng = np.random.default_rng(0)
    parts = []
    for i in range(200):  # none of them is 2% of the link on its own
        m = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
        m.apply_translation(rng.uniform(-0.5, 0.5, 3))
        parts.append(NS(mesh=m, volume=m.volume))
    link = NS(parts=parts, mesh=lambda: trimesh.util.concatenate([p.mesh for p in parts]))
    kept = _significant(link, parts, 0.02)
    assert 20 < len(kept) <= 60


def test_flat_plate_gets_a_box_not_a_degenerate_hull():
    from cad2urdf.geometry import _hull_geom

    flat = trimesh.Trimesh([[0, 0, 0], [0.1, 0, 0], [0.1, 0.05, 0], [0, 0.05, 0]], [[0, 1, 2], [0, 2, 3]])
    g = _hull_geom(flat, 64, "plate")
    assert g.kind == "box" and min(g.size) >= 0.002
    solid = _hull_geom(trimesh.creation.icosphere(radius=0.02), 64, "ball")
    assert solid.kind == "mesh"


def test_nothing_is_classified_by_name_without_part_classes():
    from cad2urdf import util

    util.set_part_classes(None)
    assert not util.is_fastener("M3x8_SHCS") and not util.is_non_physical("LidarFov") and not util.part_is("servo", "STS3215")
    util.set_part_classes({"fastener": ["^zork"]})  # only what the spec says
    try:
        assert util.is_fastener("Zork 7") and not util.is_fastener("M3x8_SHCS")
        import pytest

        with pytest.raises(ValueError, match="unknown part_classes"):
            util.set_part_classes({"fasteners": ["x"]})  # a typo is an error, not silently ignored
    finally:
        util.set_part_classes(None)
