"""Fastener detection and the sphere ("bubble") collision fit."""

import numpy as np
import trimesh

from cad2urdf.geometry import fit_spheres
from cad2urdf.util import is_fastener


def test_fastener_names():
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
