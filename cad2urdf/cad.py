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
    skipped_faces: int = 0  # faces that failed to triangulate


def load_parts(step_path: Path) -> list[Part]:
    """Flatten a STEP assembly into uniquely named solids (multi-solid leaves get ``_0, _1, ...``)."""
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
    # exports repeat names (every screw): suffix duplicates
    seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    for p in parts:
        counts[p.name] = counts.get(p.name, 0) + 1
    for p in parts:
        if counts[p.name] > 1:
            seen[p.name] = seen.get(p.name, 0) + 1
            p.name = f"{p.name}#{seen[p.name]}"
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
    """Mesh the part face by face; faces that fail to triangulate are counted in ``skipped_faces``."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    lin = linear_mm * 1e-3 / unit_scale  # tolerance expressed in CAD units
    BRepMesh_IncrementalMesh(part.shape.wrapped, lin, False, np.radians(angular_deg), True)
    verts, tris, skipped, offset = [], [], 0, 0
    exp = TopExp_Explorer(part.shape.wrapped, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        exp.Next()
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        if poly is None or poly.NbTriangles() == 0:
            skipped += 1
            continue
        trsf = loc.Transformation()
        pts = [poly.Node(i).Transformed(trsf) for i in range(1, poly.NbNodes() + 1)]
        verts += [(p.X(), p.Y(), p.Z()) for p in pts]
        rev = face.Orientation() == TopAbs_REVERSED
        for i in range(1, poly.NbTriangles() + 1):
            a, b, c = poly.Triangle(i).Get()
            tris.append((a - 1 + offset, c - 1 + offset, b - 1 + offset) if rev else (a - 1 + offset, b - 1 + offset, c - 1 + offset))
        offset += poly.NbNodes()
    mesh = trimesh.Trimesh(np.array(verts) * unit_scale, np.array(tris), process=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    part.mesh = mesh
    part.skipped_faces = skipped
