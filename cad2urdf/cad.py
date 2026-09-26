"""STEP loading, B-rep mass properties and tessellation (OpenCascade via build123d/OCP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from build123d import Shape, import_step
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

UNIT_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}


@dataclass
class Part:
    name: str
    shape: Shape  # exact B-rep, CAD units
    material: str = ""
    density: float = 0.0  # kg/m^3
    link: str = ""
    # SI mass properties, world frame (zero configuration)
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about COM
    volume: float = 0.0  # m^3
    mesh: trimesh.Trimesh | None = None  # world frame, metres


def load_parts(step_path: Path) -> list[Part]:
    """Flatten a STEP assembly into named solids.

    Nested sub-assemblies are walked depth-first; a leaf that contains several
    solids is split and suffixed ``_0, _1, ...`` so every solid has a unique name.
    """
    root = import_step(str(step_path))
    parts: list[Part] = []

    def walk(shape: Shape, prefix: str):
        if shape.children:
            for child in shape.children:
                walk(child, prefix)
            return
        solids = shape.solids()
        base = shape.label or f"part_{len(parts)}"
        for i, solid in enumerate(solids):
            name = base if len(solids) == 1 else f"{base}_{i}"
            parts.append(Part(name=name, shape=solid))

    walk(root, "")
    names = [p.name for p in parts]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate part names in STEP: {sorted(dupes)}")
    return parts


def mass_properties(part: Part, unit_scale: float) -> None:
    """Exact volume/COM/inertia from the B-rep, scaled to SI with the part density."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(part.shape.wrapped, props)
    vol_cad = props.Mass()  # with density 1 this is the volume, in CAD units^3
    c = props.CentreOfMass()
    m = props.MatrixOfInertia()  # about the centre of mass, CAD units^5
    inertia_cad = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    part.volume = vol_cad * unit_scale**3
    part.mass = part.density * part.volume
    part.com = np.array([c.X(), c.Y(), c.Z()]) * unit_scale
    part.inertia = inertia_cad * part.density * unit_scale**5


def tessellate(part: Part, unit_scale: float, linear_mm: float, angular_deg: float) -> None:
    lin = linear_mm * 1e-3 / unit_scale  # tolerance expressed in CAD units
    verts, tris = part.shape.tessellate(lin, np.radians(angular_deg))
    v = np.array([[p.X, p.Y, p.Z] for p in verts]) * unit_scale
    mesh = trimesh.Trimesh(v, np.array(tris), process=True)
    mesh.fix_normals()
    part.mesh = mesh
