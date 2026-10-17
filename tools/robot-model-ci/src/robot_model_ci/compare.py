from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .checks import validate_path


def _index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f"{model.get('format')}:{model.get('path')}" : model for model in report.get("models", [])}


def _approx(a: Any, b: Any, tol: float) -> bool:
    if a is None or b is None:
        return a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return a == b


def compare_dirs(baseline_path: str | Path, candidate_path: str | Path, tolerance: float = 1e-6) -> dict[str, Any]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    baseline = validate_path(baseline_path)
    candidate = validate_path(candidate_path)
    old_index, new_index = _index(baseline), _index(candidate)
    changes: list[dict[str, Any]] = []
    breaking = 0

    for key in sorted(old_index.keys() - new_index.keys()):
        changes.append({"severity": "BREAKING", "code": "model.removed", "model": key, "message": "Model artifact removed"})
        breaking += 1
    for key in sorted(new_index.keys() - old_index.keys()):
        changes.append({"severity": "INFO", "code": "model.added", "model": key, "message": "New model artifact added"})

    for key in sorted(old_index.keys() & new_index.keys()):
        old, new = old_index[key], new_index[key]
        fmt = old.get("format")
        for collection, label in (("links", "link"), ("bodies", "body"), ("joints", "joint"), ("actuators", "actuator")):
            before = old.get(collection, {})
            after = new.get(collection, {})
            for name in sorted(before.keys() - after.keys()):
                severe = label in {"link", "joint"}
                changes.append({"severity": "BREAKING" if severe else "WARNING", "code": f"{label}.removed", "model": key, "name": name,
                                "message": f"{label.capitalize()} '{name}' removed"})
                breaking += int(severe)
            for name in sorted(after.keys() - before.keys()):
                changes.append({"severity": "INFO", "code": f"{label}.added", "model": key, "name": name,
                                "message": f"{label.capitalize()} '{name}' added"})
            for name in sorted(before.keys() & after.keys()):
                a, b = before[name], after[name]
                if label == "joint":
                    for field in ("type", "parent", "child"):
                        if a.get(field) != b.get(field):
                            changes.append({"severity": "BREAKING", "code": f"joint.{field}_changed", "model": key, "name": name,
                                            "before": a.get(field), "after": b.get(field),
                                            "message": f"Joint '{name}' {field} changed from {a.get(field)!r} to {b.get(field)!r}"})
                            breaking += 1
                    a_axis, b_axis = a.get("axis"), b.get("axis")
                    if isinstance(a_axis, list) and isinstance(b_axis, list) and any(not _approx(x, y, tolerance) for x, y in zip(a_axis, b_axis)):
                        changes.append({"severity": "WARNING", "code": "joint.axis_changed", "model": key, "name": name,
                                        "before": a_axis, "after": b_axis, "message": f"Joint '{name}' axis changed"})
                    for field in ("lower", "upper", "effort", "velocity", "range"):
                        av = a.get("limits", {}).get(field) if isinstance(a.get("limits"), dict) else a.get("range")
                        bv = b.get("limits", {}).get(field) if isinstance(b.get("limits"), dict) else b.get("range")
                        if isinstance(av, list) or isinstance(bv, list):
                            if av != bv:
                                changes.append({"severity": "WARNING", "code": f"joint.{field}_changed", "model": key, "name": name,
                                                "before": av, "after": bv, "message": f"Joint '{name}' {field} changed; review mechanically"})
                        elif av is not None and bv is not None and not _approx(av, bv, tolerance):
                            changes.append({"severity": "WARNING", "code": f"joint.{field}_changed", "model": key, "name": name,
                                            "before": av, "after": bv, "message": f"Joint '{name}' {field} changed; review mechanically"})
                elif label == "link" and a.get("inertial") != b.get("inertial"):
                    changes.append({"severity": "WARNING", "code": "link.inertial_changed", "model": key, "name": name,
                                    "before": a.get("inertial"), "after": b.get("inertial"),
                                    "message": f"Link '{name}' inertial properties changed; review mass and inertia"})
        old_hashes = old.get("mesh_hashes", {})
        new_hashes = new.get("mesh_hashes", {})
        if fmt == "urdf":
            for asset in sorted(old_hashes.keys() | new_hashes.keys()):
                if old_hashes.get(asset) != new_hashes.get(asset):
                    changes.append({"severity": "WARNING", "code": "mesh.content_changed", "model": key, "name": asset,
                                    "message": f"Mesh content changed or could not be resolved consistently: {asset}"})
        if fmt == "mjcf":
            old_assets, new_assets = old.get("assets", {}), new.get("assets", {})
            for name in sorted(old_assets.keys() | new_assets.keys()):
                if old_assets.get(name) != new_assets.get(name):
                    changes.append({"severity": "WARNING", "code": "mesh.content_changed", "model": key, "name": name,
                                    "message": f"MJCF asset '{name}' changed; review geometry"})

    return {
        "schema_version": 1,
        "baseline": str(Path(baseline_path).resolve()),
        "candidate": str(Path(candidate_path).resolve()),
        "status": "failed" if breaking or baseline.get("status") != "passed" or candidate.get("status") != "passed" else "passed",
        "summary": {"breaking_changes": breaking,
                    "warnings": sum(c["severity"] == "WARNING" for c in changes),
                    "info": sum(c["severity"] == "INFO" for c in changes)},
        "baseline_validation": baseline.get("summary"),
        "candidate_validation": candidate.get("summary"),
        "changes": changes,
    }
