from __future__ import annotations

import hashlib
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Issue:
    severity: str
    code: str
    file: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _issue(items: list[Issue], severity: str, code: str, path: Path, root: Path, message: str) -> None:
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = str(path)
    items.append(Issue(severity, code, name, message))


def _numbers(value: str | None, count: int, label: str) -> list[float] | None:
    if value is None:
        return None
    try:
        vals = [float(x) for x in value.replace(",", " ").split()]
    except ValueError:
        raise ValueError(f"{label} must contain numbers")
    if len(vals) != count:
        raise ValueError(f"{label} must contain {count} numbers; got {len(vals)}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{label} contains NaN or infinity")
    return vals


def _resolve_asset(ref: str, model_file: Path, root: Path) -> Path | None:
    """Resolve local assets conservatively; return None for remote/unhandled URIs."""
    raw = ref.strip()
    if not raw or raw.startswith(("http://", "https://")):
        return None
    if raw.startswith("file://"):
        raw = raw[len("file://"):]
    if raw.startswith("package://"):
        rest = raw[len("package://"):]
        _, _, rel = rest.partition("/")
        candidates = [root / rel, model_file.parent / rel]
        # Some output layouts keep the package directory inside the output root.
        pkg, _, tail = rest.partition("/")
        if tail:
            candidates.extend([root / pkg / tail, root.parent / pkg / tail])
    else:
        p = Path(raw)
        if p.is_absolute():
            return p
        candidates = [model_file.parent / p, root / p, root.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _check_asset(ref: str, path: Path, root: Path, issues: list[Issue], label: str) -> None:
    resolved = _resolve_asset(ref, path, root)
    if resolved is None:
        return
    # Absolute paths may be intentional on a developer's machine, but make portability visible.
    if Path(ref).is_absolute() or ref.startswith("file://"):
        if not resolved.exists():
            _issue(issues, "ERROR", "asset.missing", path, root, f"{label} asset does not exist: {ref}")
        else:
            _issue(issues, "WARNING", "asset.absolute_path", path, root,
                   f"{label} uses an absolute/file URI path, which may not work on another machine: {ref}")
        return
    if not resolved.exists():
        _issue(issues, "ERROR", "asset.missing", path, root, f"{label} asset could not be resolved: {ref}")


def _check_urdf(path: Path, root: Path, issues: list[Issue]) -> dict[str, Any] | None:
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        _issue(issues, "ERROR", "xml.parse", path, root, f"Unable to parse URDF XML: {exc}")
        return None
    robot = tree.getroot()
    if robot.tag != "robot":
        _issue(issues, "ERROR", "urdf.root", path, root, f"Expected <robot> root, found <{robot.tag}>")
        return None

    links = [e.get("name", "") for e in robot.findall("link")]
    joints = robot.findall("joint")
    link_set = set(links)
    if not robot.get("name"):
        _issue(issues, "WARNING", "urdf.robot_name", path, root, "Robot has no name attribute")
    if len(links) != len(link_set):
        _issue(issues, "ERROR", "urdf.duplicate_link", path, root, "Duplicate link names found")
    if any(not x for x in links):
        _issue(issues, "ERROR", "urdf.link_name", path, root, "One or more links have no name")
    joint_names = [j.get("name", "") for j in joints]
    if len(joint_names) != len(set(joint_names)):
        _issue(issues, "ERROR", "urdf.duplicate_joint", path, root, "Duplicate joint names found")

    child_parent: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    joint_summary: dict[str, Any] = {}
    link_summary: dict[str, Any] = {}
    mesh_hashes: dict[str, str] = {}
    for link in robot.findall("link"):
        name = link.get("name", "")
        inertial = link.find("inertial")
        inertial_data: dict[str, Any] | None = None
        if inertial is not None:
            mass_el, inertia_el = inertial.find("mass"), inertial.find("inertia")
            try:
                mass = float(mass_el.get("value", "nan")) if mass_el is not None else float("nan")
            except ValueError:
                mass = float("nan")
            if not math.isfinite(mass) or mass <= 0:
                _issue(issues, "ERROR", "urdf.mass", path, root, f"Link '{name}' has non-positive or invalid mass")
            if inertia_el is None:
                _issue(issues, "ERROR", "urdf.inertia_missing", path, root, f"Link '{name}' has <inertial> but no <inertia>")
            else:
                vals: dict[str, float] = {}
                for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    try:
                        vals[key] = float(inertia_el.get(key, "nan"))
                    except ValueError:
                        vals[key] = float("nan")
                if not all(math.isfinite(v) for v in vals.values()):
                    _issue(issues, "ERROR", "urdf.inertia_values", path, root, f"Link '{name}' has missing/non-finite inertia entries")
                else:
                    ixx, ixy, ixz = vals["ixx"], vals["ixy"], vals["ixz"]
                    iyy, iyz, izz = vals["iyy"], vals["iyz"], vals["izz"]
                    det = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz) + ixz * (ixy * iyz - iyy * ixz)
                    if min(ixx, iyy, izz) < -1e-9 or (ixx * iyy - ixy * ixy) < -1e-9 or det < -1e-9:
                        _issue(issues, "ERROR", "urdf.inertia_not_psd", path, root,
                               f"Link '{name}' inertia tensor is not positive semidefinite")
                inertial_data = {"mass": mass, **vals}
        else:
            inertial_data = None
        for mesh in link.findall(".//mesh"):
            ref = mesh.get("filename")
            if not ref:
                _issue(issues, "ERROR", "asset.mesh_filename", path, root, f"Link '{name}' has a mesh without filename")
                continue
            _check_asset(ref, path, root, issues, f"Link '{name}' mesh")
            resolved = _resolve_asset(ref, path, root)
            if resolved and resolved.exists() and resolved.is_file():
                try:
                    mesh_hashes[ref] = hashlib.sha256(resolved.read_bytes()).hexdigest()
                except OSError:
                    pass
        link_summary[name] = {"inertial": inertial_data}

    for joint in joints:
        name = joint.get("name", "")
        jtype = joint.get("type", "")
        parent_el, child_el = joint.find("parent"), joint.find("child")
        parent = parent_el.get("link", "") if parent_el is not None else ""
        child = child_el.get("link", "") if child_el is not None else ""
        if not name or not jtype:
            _issue(issues, "ERROR", "urdf.joint_attributes", path, root, "Joint missing name or type")
        if jtype not in {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}:
            _issue(issues, "ERROR", "urdf.joint_type", path, root, f"Joint '{name}' has unsupported type '{jtype}'")
        if parent not in link_set:
            _issue(issues, "ERROR", "urdf.parent_missing", path, root, f"Joint '{name}' references missing parent link '{parent}'")
        if child not in link_set:
            _issue(issues, "ERROR", "urdf.child_missing", path, root, f"Joint '{name}' references missing child link '{child}'")
        if parent and child:
            if parent == child:
                _issue(issues, "ERROR", "urdf.self_joint", path, root, f"Joint '{name}' connects link '{parent}' to itself")
            if child in child_parent and child_parent[child] != parent:
                _issue(issues, "ERROR", "urdf.multiple_parents", path, root,
                       f"Link '{child}' has more than one parent joint")
            child_parent[child] = parent
            edges.append((parent, child))
        for origin in joint.findall("origin"):
            for key in ("xyz", "rpy"):
                try:
                    _numbers(origin.get(key), 3, f"joint '{name}' origin {key}")
                except ValueError as exc:
                    _issue(issues, "ERROR", "urdf.origin_vector", path, root, str(exc))
        axis_el = joint.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_el is not None and axis_el.get("xyz") is not None:
            try:
                axis = _numbers(axis_el.get("xyz"), 3, f"joint '{name}' axis") or axis
                norm = math.sqrt(sum(x * x for x in axis))
                if norm <= 1e-10:
                    _issue(issues, "ERROR", "urdf.axis_zero", path, root, f"Joint '{name}' has a zero-length axis")
                elif abs(norm - 1.0) > 1e-3:
                    _issue(issues, "WARNING", "urdf.axis_not_unit", path, root,
                           f"Joint '{name}' axis norm is {norm:.4g}, not 1; confirm intended normalization")
            except ValueError as exc:
                _issue(issues, "ERROR", "urdf.axis_vector", path, root, str(exc))
        limit_el = joint.find("limit")
        limits: dict[str, float | None] = {}
        if jtype in {"revolute", "prismatic", "continuous"} and limit_el is None:
            _issue(issues, "ERROR", "urdf.limit_missing", path, root, f"Movable joint '{name}' has no <limit> element")
        if limit_el is not None:
            for key in ("lower", "upper", "effort", "velocity"):
                raw = limit_el.get(key)
                if raw is None:
                    limits[key] = None
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    val = float("nan")
                limits[key] = val
                if not math.isfinite(val):
                    _issue(issues, "ERROR", "urdf.limit_number", path, root, f"Joint '{name}' has invalid {key} limit")
                if key in {"effort", "velocity"} and math.isfinite(val) and val <= 0:
                    _issue(issues, "ERROR", "urdf.limit_positive", path, root, f"Joint '{name}' {key} must be > 0")
            if jtype in {"revolute", "prismatic"}:
                lo, hi = limits.get("lower"), limits.get("upper")
                if lo is None or hi is None:
                    _issue(issues, "ERROR", "urdf.position_limits", path, root,
                           f"Joint '{name}' ({jtype}) needs both lower and upper limits")
                elif math.isfinite(lo) and math.isfinite(hi) and lo >= hi:
                    _issue(issues, "ERROR", "urdf.limit_order", path, root, f"Joint '{name}' lower limit must be < upper")
        joint_summary[name] = {"type": jtype, "parent": parent, "child": child, "axis": axis, "limits": limits}

    if links:
        roots = link_set - set(child_parent)
        if len(roots) == 0:
            _issue(issues, "ERROR", "urdf.no_root", path, root, "No root link found; joint graph may contain a cycle")
        elif len(roots) > 1:
            _issue(issues, "WARNING", "urdf.multiple_roots", path, root,
                   f"Found {len(roots)} root links; URDF usually expects one connected tree")
        if len(roots) == 1:
            reachable: set[str] = set()
            frontier = list(roots)
            while frontier:
                current = frontier.pop()
                if current in reachable:
                    continue
                reachable.add(current)
                frontier.extend(child for parent, child in edges if parent == current)
            unreachable = link_set - reachable
            if unreachable:
                _issue(issues, "WARNING", "urdf.disconnected", path, root,
                       "Links not reachable from root: " + ", ".join(sorted(unreachable)))
    return {"format": "urdf", "name": robot.get("name", path.stem), "links": link_summary,
            "joints": joint_summary, "mesh_hashes": mesh_hashes}


def _check_mjcf(path: Path, root: Path, issues: list[Issue]) -> dict[str, Any] | None:
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        _issue(issues, "ERROR", "xml.parse", path, root, f"Unable to parse MJCF XML: {exc}")
        return None
    model = tree.getroot()
    if model.tag != "mujoco":
        _issue(issues, "ERROR", "mjcf.root", path, root, f"Expected <mujoco> root, found <{model.tag}>")
        return None
    inventory: dict[str, Any] = {"format": "mjcf", "name": model.get("model", path.stem), "bodies": {}, "joints": {}, "actuators": {}, "assets": {}}
    for include in model.findall(".//include"):
        ref = include.get("file")
        if ref:
            _check_asset(ref, path, root, issues, "MJCF include")
    for mesh in model.findall(".//asset/mesh"):
        name, ref = mesh.get("name", ""), mesh.get("file", "")
        if not ref:
            _issue(issues, "ERROR", "mjcf.mesh_file", path, root, f"MJCF mesh '{name}' has no file")
            continue
        _check_asset(ref, path, root, issues, f"MJCF mesh '{name}'")
        resolved = _resolve_asset(ref, path, root)
        digest = None
        if resolved and resolved.exists() and resolved.is_file():
            try:
                digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            except OSError:
                pass
        inventory["assets"][name] = {"file": ref, "sha256": digest}
    for body in model.findall(".//worldbody//body"):
        name = body.get("name", "")
        if not name:
            _issue(issues, "WARNING", "mjcf.body_name", path, root, "Body without a name found")
        elif name in inventory["bodies"]:
            _issue(issues, "ERROR", "mjcf.duplicate_body", path, root, f"Duplicate body name '{name}'")
        inventory["bodies"][name] = {"pos": body.get("pos"), "quat": body.get("quat"), "euler": body.get("euler")}
    for joint in model.findall(".//worldbody//joint"):
        name, jtype = joint.get("name", ""), joint.get("type", "hinge")
        if not name:
            _issue(issues, "WARNING", "mjcf.joint_name", path, root, "Joint without a name found")
            continue
        if name in inventory["joints"]:
            _issue(issues, "ERROR", "mjcf.duplicate_joint", path, root, f"Duplicate joint name '{name}'")
        record: dict[str, Any] = {"type": jtype, "range": None, "limited": joint.get("limited")}
        raw_range = joint.get("range")
        if raw_range:
            try:
                vals = _numbers(raw_range, 2, f"MJCF joint '{name}' range")
                record["range"] = vals
                if vals and vals[0] >= vals[1]:
                    _issue(issues, "ERROR", "mjcf.joint_range", path, root, f"Joint '{name}' range lower bound must be < upper")
            except ValueError as exc:
                _issue(issues, "ERROR", "mjcf.joint_range", path, root, str(exc))
        inventory["joints"][name] = record
    for actuator in model.findall(".//actuator/*"):
        name = actuator.get("name", "")
        if not name:
            _issue(issues, "WARNING", "mjcf.actuator_name", path, root, f"<{actuator.tag}> actuator without a name")
        else:
            inventory["actuators"][name] = {"type": actuator.tag, "joint": actuator.get("joint"), "target": actuator.get("target")}
    return inventory


def discover_models(path: Path) -> list[tuple[Path, str]]:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".urdf":
            return [(path, "urdf")]
        if suffix == ".mjcf":
            return [(path, "mjcf")]
        if suffix == ".xml":
            try:
                if ET.parse(path).getroot().tag == "mujoco":
                    return [(path, "mjcf")]
            except (ET.ParseError, OSError):
                pass
        return []
    found: list[tuple[Path, str]] = []
    for candidate in sorted(path.rglob("*.urdf")):
        found.append((candidate, "urdf"))
    for candidate in sorted(path.rglob("*.mjcf")):
        found.append((candidate, "mjcf"))
    for candidate in sorted(path.rglob("*.xml")):
        if any(parent.name.lower() in {"meshes", ".git", "node_modules"} for parent in candidate.parents):
            continue
        try:
            if ET.parse(candidate).getroot().tag == "mujoco":
                found.append((candidate, "mjcf"))
        except (ET.ParseError, OSError):
            continue
    # Avoid duplicates if a .mjcf file is also parsed as XML (it is not in current loop, but dedupe anyway).
    unique: dict[Path, str] = {p.resolve(): fmt for p, fmt in found}
    return [(Path(p), fmt) for p, fmt in sorted(unique.items(), key=lambda item: str(item[0]))]


def validate_path(target: str | Path) -> dict[str, Any]:
    path = Path(target).expanduser().resolve()
    root = path if path.is_dir() else path.parent
    issues: list[Issue] = []
    if not path.exists():
        issues.append(Issue("ERROR", "path.missing", str(path), "Input path does not exist"))
        return {"schema_version": 1, "target": str(path), "status": "failed", "models": [], "issues": [x.to_dict() for x in issues], "summary": {"errors": 1, "warnings": 0, "models": 0}}
    models = discover_models(path)
    if not models:
        issues.append(Issue("ERROR", "model.none_found", str(path), "No URDF or MJCF model files found"))
    inventory: list[dict[str, Any]] = []
    for model_path, fmt in models:
        parsed = _check_urdf(model_path, root, issues) if fmt == "urdf" else _check_mjcf(model_path, root, issues)
        if parsed is not None:
            try:
                relative = model_path.relative_to(root).as_posix()
            except ValueError:
                relative = model_path.name
            parsed["path"] = relative
            inventory.append(parsed)
    # Cross-format consistency is a warning gate because simulator exporters may legitimately
    # encode fixed joints, mimic joints, or unsupported constraints differently.
    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    source_by_name: dict[str, Path] = {}
    for model_path, _fmt in models:
        for item in inventory:
            if item.get("path") == (model_path.relative_to(root).as_posix() if model_path.is_relative_to(root) else model_path.name):
                by_name.setdefault(str(item.get("name", "")), {})[str(item.get("format"))] = item
                source_by_name[str(item.get("name", ""))] = model_path
    for model_name, formats in by_name.items():
        if "urdf" not in formats or "mjcf" not in formats:
            continue
        urdf_joints = {name: data for name, data in formats["urdf"].get("joints", {}).items()
                       if data.get("type") != "fixed"}
        mjcf_joints = formats["mjcf"].get("joints", {})
        for joint_name in sorted(urdf_joints.keys() - mjcf_joints.keys()):
            _issue(issues, "WARNING", "cross_format.joint_missing_mjcf", source_by_name[model_name], root,
                   f"Movable URDF joint '{joint_name}' is not present in MJCF model '{model_name}'")
        for joint_name in sorted(mjcf_joints.keys() - urdf_joints.keys()):
            _issue(issues, "WARNING", "cross_format.joint_missing_urdf", source_by_name[model_name], root,
                   f"MJCF joint '{joint_name}' is not present as a movable joint in URDF model '{model_name}'")
        mjcf_type_map = {"hinge": {"revolute", "continuous"}, "slide": {"prismatic"}, "ball": {"floating"}, "free": {"floating"}}
        for joint_name in sorted(urdf_joints.keys() & mjcf_joints.keys()):
            urdf_type = urdf_joints[joint_name].get("type")
            mjcf_type = mjcf_joints[joint_name].get("type", "hinge")
            if urdf_type not in mjcf_type_map.get(mjcf_type, set()):
                _issue(issues, "WARNING", "cross_format.joint_type_mismatch", source_by_name[model_name], root,
                       f"Joint '{joint_name}' type differs across formats (URDF={urdf_type}, MJCF={mjcf_type})")
    errors = sum(i.severity == "ERROR" for i in issues)
    warnings = sum(i.severity == "WARNING" for i in issues)
    return {
        "schema_version": 1,
        "target": str(path),
        "status": "passed" if errors == 0 else "failed",
        "summary": {"errors": errors, "warnings": warnings, "models": len(inventory)},
        "issues": [i.to_dict() for i in issues],
        "models": inventory,
    }
