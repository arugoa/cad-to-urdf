"""Joint-limit sweep: move each child link to its limits and measure how much solid volume it shares with
its parent. A jump means the limit drives the part through material (wrong sign, offset or stroke);
simulators can't catch this because parent/child contacts are filtered.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .model import Robot


def _manifold(mesh: trimesh.Trimesh):
    import manifold3d as mf

    if mesh is None or not mesh.is_watertight:
        return None
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    return None if m.is_empty() else m


def _union(parts):
    import manifold3d as mf

    solids = [s for s in (_manifold(p.mesh) for p in parts) if s is not None]
    return mf.Manifold.batch_boolean(solids, mf.OpType.Add) if solids else None, len(solids)


def _motion(joint, q: float) -> np.ndarray:
    """World transform applied to the child's zero-pose geometry when the joint is at q."""
    T = np.eye(4)
    a = joint.axis / np.linalg.norm(joint.axis)
    if joint.type == "prismatic":
        T[:3, 3] = a * q
        return T
    R = trimesh.transformations.rotation_matrix(q, a, joint.origin)
    return R


def limit_sweep(robot: Robot, tol_cm3: float = 1.0) -> dict:
    out, flagged = {}, []
    solids = {name: _union(link.parts) for name, link in robot.links.items()}
    for j in robot.joints.values():
        if j.type not in ("revolute", "prismatic") or j.mimic:
            continue
        (child, _), (parent, _) = solids[j.child], solids[j.parent]
        if child is None or parent is None:
            out[j.name] = {"skipped": "no closed part geometry"}
            continue
        vols = {}
        for tag, q in (("zero", 0.0), ("lower", j.lower), ("upper", j.upper)):
            moved = child.transform(_motion(j, q)[:3, :4])
            vols[tag] = float((moved ^ parent).volume()) * 1e6  # cm^3
        out[j.name] = {f"overlap_cm3_at_{k}": round(v, 2) for k, v in vols.items()}
        if max(vols["lower"], vols["upper"]) > vols["zero"] + tol_cm3:
            flagged.append(j.name)
    return {"joints": out, "limits_driving_into_parent": flagged}
