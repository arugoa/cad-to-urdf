"""Front ends that read a robot description instead of a bare STEP.

* URDF (an exporter's: Onshape export, onshape-to-robot, sw2robot, ACDC4Robot, creo2urdf): keeps frames,
  joints, limits and inertia; each <visual> becomes a part. Spec keys: ``source``, ``package_dirs``,
  ``base``, ``root_rpy``, joint/dynamics/actuator/collision/srdf overrides.
* Onshape: a live assembly through the REST API.

    FASTENED / rigid sub-assembly -> same link      REVOLUTE -> revolute (continuous without limits)
    SLIDER -> prismatic                              CYLINDRICAL / PIN_SLOT -> revolute (REVIEW)
    gear / rack / screw relations -> mimic           PLANAR -> 2 slides + a rotation (massless links between)
    mates on fasteners (screws, nuts) -> fastened    BALL / ... -> ignored (REVIEW)

  Naming conventions (as in onshape-to-robot): if any mate is named ``dof_*``, only those are joints and
  ``_inv`` flips the axis; ``closing_*`` mates close loops; joints named ``*passive*`` get no actuator and
  ``*_speed`` a velocity one; ``frame_*`` markers are skipped. Masses come from Onshape's materials, meshes
  are per-part STL, responses are cached in ``~/.cache/cad2urdf/onshape``. Keys: ONSHAPE_ACCESS_KEY /
  ONSHAPE_SECRET_KEY (docs/ONSHAPE_API_KEYS.md). Spec keys: ``source`` (assembly URL),
  ``subassemblies: flexible|rigid``, ``rigid_subassemblies: [regex]``, ``default_density``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlparse

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .model import CollisionGeom, Joint, Link, Robot, _per, check_inertia, combine_inertia
from .step import Part
from .util import UnionFind, is_fastener, slug


def _pose(el: ET.Element | None) -> np.ndarray:
    T = np.eye(4)
    if el is None:
        return T
    T[:3, 3] = [float(x) for x in el.get("xyz", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", [float(x) for x in el.get("rpy", "0 0 0").split()]).as_matrix()
    return T


def _floats(s: str | None, n: int, default=0.0) -> list[float]:
    return [float(x) for x in s.split()] if s else [default] * n


class _Resolver:
    """package:// and relative mesh paths, the usual source of broken exports."""

    def __init__(self, urdf: Path, package_dirs: dict[str, str]):
        self.urdf_dir = urdf.parent.resolve()
        self.package_dirs = {k: Path(v).expanduser() for k, v in package_dirs.items()}

    def __call__(self, filename: str) -> Path:
        if filename.startswith("file://"):
            return Path(filename[7:])
        if not filename.startswith("package://"):
            p = Path(filename)
            return p if p.is_absolute() else self.urdf_dir / p
        pkg, _, rest = filename[len("package://"):].partition("/")
        candidates = []
        if pkg in self.package_dirs:
            candidates.append(self.package_dirs[pkg] / rest)
        for d in [self.urdf_dir, *self.urdf_dir.parents][:4]:
            candidates += [d / rest, d / pkg / rest]
            if d.name == pkg:
                candidates.append(d / rest)
        for c in candidates:
            if c.exists():
                return c
        hits = list(self.urdf_dir.rglob(Path(rest).name))
        if len(hits) == 1:
            return hits[0]
        raise FileNotFoundError(f"cannot resolve {filename}; add it to package_dirs in the spec")


def _geometry(geo: ET.Element, resolve: _Resolver) -> tuple[str, tuple, trimesh.Trimesh | None]:
    """(kind, size, mesh-in-element-frame)."""
    child = geo[0]
    if child.tag == "box":
        size = tuple(_floats(child.get("size"), 3))
        return "box", size, trimesh.creation.box(extents=size)
    if child.tag == "cylinder":
        r, l = float(child.get("radius")), float(child.get("length"))
        return "cylinder", (r, l), trimesh.creation.cylinder(radius=r, height=l, sections=32)
    if child.tag == "sphere":
        r = float(child.get("radius"))
        return "sphere", (r,), trimesh.creation.icosphere(subdivisions=3, radius=r)
    if child.tag == "mesh":
        m = trimesh.load(resolve(child.get("filename")), force="mesh", process=True)
        if child.get("scale"):
            m.apply_scale(_floats(child.get("scale"), 3))
        return "mesh", (), m
    raise ValueError(f"unsupported geometry <{child.tag}>")



def build_from_urdf(spec: dict, base: Path) -> Robot:
    urdf_path = (base / spec["source"]).resolve()
    tree = ET.parse(urdf_path).getroot()
    resolve = _Resolver(urdf_path, spec.get("package_dirs", {}))

    named_colors = {}
    for m in tree.findall("material"):
        c = m.find("color")
        if c is not None:
            named_colors[m.get("name")] = c.get("rgba")

    # --- joints / tree, world poses at q = 0
    joint_els = tree.findall("joint")
    children = {j.find("child").get("link") for j in joint_els}
    link_names = [l.get("name") for l in tree.findall("link")]
    roots = [n for n in link_names if n not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, got {roots}")
    root = roots[0]
    T_root = np.eye(4)
    T_root[:3, :3] = Rotation.from_euler("xyz", spec.get("root_rpy", [0, 0, 0])).as_matrix()
    T_root[:3, 3] = spec.get("root_xyz", [0, 0, 0])
    world = {root: T_root}
    pending = list(joint_els)
    while pending:
        progressed = False
        for j in list(pending):
            p, c = j.find("parent").get("link"), j.find("child").get("link")
            if p in world:
                world[c] = world[p] @ _pose(j.find("origin"))
                pending.remove(j)
                progressed = True
        if not progressed:
            raise ValueError("URDF joints do not form a tree")

    # --- links
    robot_materials = {}
    links: dict[str, Link] = {}
    for lel in tree.findall("link"):
        name = lel.get("name")
        T = world[name]
        link = Link(name, origin=T[:3, 3].copy(), rotation=T[:3, :3].copy())
        inert = lel.find("inertial")
        if inert is not None and inert.find("mass") is not None:
            Ti = _pose(inert.find("origin"))
            ie = inert.find("inertia")
            g = lambda k: float(ie.get(k, 0.0)) if ie is not None else 0.0
            I = np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]])
            link.mass = float(inert.find("mass").get("value"))
            link.com = Ti[:3, 3]
            link.inertia = Ti[:3, :3] @ I @ Ti[:3, :3].T
        for i, v in enumerate(lel.findall("visual")):
            kind, size, m = _geometry(v.find("geometry"), resolve)
            m.apply_transform(_pose(v.find("origin")))
            key = f"{i}_{slug(v.get('name', ''))}" if v.get("name") else f"{i}"
            link.visuals[key] = m
            mat = v.find("material")
            rgba = None
            mat_name = f"{name}_{key}"
            if mat is not None:
                c = mat.find("color")
                rgba = c.get("rgba") if c is not None else named_colors.get(mat.get("name"))
                mat_name = slug(mat.get("name") or mat_name)
            robot_materials.setdefault(mat_name, rgba or "0.7 0.7 0.7 1")
            link.visual_materials[key] = mat_name
            world_mesh = m.copy()
            world_mesh.apply_transform(T)
            part = Part(name=f"{name}/{key}", shape=None, link=name, mesh=world_mesh)
            part.volume = float(world_mesh.volume if world_mesh.is_watertight else world_mesh.convex_hull.volume)
            link.parts.append(part)
        for c in lel.findall("collision"):
            kind, size, m = _geometry(c.find("geometry"), resolve)
            To = _pose(c.find("origin"))
            if kind == "mesh":
                m.apply_transform(To)
                link.source_collisions.append(CollisionGeom("mesh", np.eye(4), mesh=m, source="input"))
            else:
                link.source_collisions.append(CollisionGeom(kind, To, size, source="input"))
        for issue in check_inertia(name, link.inertia, link.mass):
            print("  note (input URDF):", issue)
        links[name] = link

    # --- joints
    overrides = spec.get("joints", {}) or {}
    joints: dict[str, Joint] = {}
    for jel in joint_els:
        jname, jtype = jel.get("name"), jel.get("type")
        child = jel.find("child").get("link")
        ax = np.array(_floats(jel.find("axis").get("xyz") if jel.find("axis") is not None else None, 3, 0.0))
        if not ax.any():
            ax = np.array([1.0, 0, 0])
        axis_world = world[child][:3, :3] @ (ax / np.linalg.norm(ax))
        lim = jel.find("limit")
        la = (lambda k, d=0.0: float(lim.get(k, d)) if lim is not None else d)
        dyn_el = jel.find("dynamics")
        mim = jel.find("mimic")
        ov = overrides.get(jname, {})
        dyn = _per(spec.get("dynamics", {}), jname)
        lower, upper = ov.get("limits", [la("lower"), la("upper")])
        joints[jname] = Joint(
            name=jname, type=ov.get("type", jtype), parent=jel.find("parent").get("link"), child=child,
            origin=world[child][:3, 3].copy(), axis=axis_world * ov.get("axis_sign", 1),
            lower=lower, upper=upper,
            effort=ov.get("effort", la("effort")), velocity=ov.get("velocity", la("velocity")),
            damping=dyn.get("damping", float(dyn_el.get("damping", 0)) if dyn_el is not None else 0.0),
            friction=dyn.get("friction", float(dyn_el.get("friction", 0)) if dyn_el is not None else 0.0),
            armature=dyn.get("armature", 0.0),
            mimic=ov.get("mimic") or ({"joint": mim.get("joint"), "multiplier": float(mim.get("multiplier", 1)),
                                       "offset": float(mim.get("offset", 0))} if mim is not None else None),
            actuator=_per(spec.get("actuators", {}), jname),
        )

    r = Robot(spec.get("robot", tree.get("name")), spec, links, joints, [], root,
              floating_base=spec.get("base", "fixed") == "floating")
    r.materials.update(robot_materials)
    return r


URL_RE = re.compile(r"/documents/(?P<did>[0-9a-f]{24})/(?P<wvm>[wvm])/(?P<wvmid>[0-9a-f]{24})/e/(?P<eid>[0-9a-f]{24})")
UNIT = {"m": 1.0, "meter": 1.0, "mm": 1e-3, "millimeter": 1e-3, "cm": 1e-2, "centimeter": 1e-2, "in": 0.0254,
        "inch": 0.0254, "ft": 0.3048, "foot": 0.3048, "rad": 1.0, "radian": 1.0, "deg": np.pi / 180, "degree": np.pi / 180}


def parse_url(url: str) -> dict:
    m = URL_RE.search(url)
    if not m:
        raise ValueError(f"not an Onshape element URL: {url}")
    return {**m.groupdict(), "host": urlparse(url).netloc}


def parse_quantity(expr: str | None, default: float | None = None) -> float | None:
    """'90 deg', '-12.5 mm', '0.3 rad', '1 in' -> SI float."""
    if not expr:
        return default
    m = re.fullmatch(r"\s*([-+]?[\d.eE+-]+)\s*\*?\s*([a-zA-Z]*)\s*", str(expr))
    if not m:
        return default
    return float(m.group(1)) * UNIT.get(m.group(2).lower(), 1.0) if m.group(2) else float(m.group(1))


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from the repo's untracked .env into os.environ (existing vars win)."""
    path = path or Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Client:
    """Minimal Onshape REST client with an on-disk response cache."""

    def __init__(self, host: str, cache_dir: Path | None = None):
        load_dotenv()
        self.base = os.environ.get("ONSHAPE_API", f"https://{host}").rstrip("/")
        self.cache = cache_dir or Path.home() / ".cache" / "cad2urdf" / "onshape"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _auth(self) -> dict:
        ak, sk = os.environ.get("ONSHAPE_ACCESS_KEY"), os.environ.get("ONSHAPE_SECRET_KEY")
        if not (ak and sk):
            raise SystemExit("ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY are not set: export them, or put them in the "
                             "repo's untracked .env file (cp .env.example .env; see docs/ONSHAPE_API_KEYS.md)")
        return {"Authorization": "Basic " + base64.b64encode(f"{ak}:{sk}".encode()).decode()}

    def get(self, path: str, params: dict | None = None, binary: bool = False):
        import requests

        key = hashlib.sha1((self.base + path + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        f = self.cache / (key + (".bin" if binary else ".json"))
        if f.exists():
            return f.read_bytes() if binary else json.loads(f.read_text())
        headers = {**self._auth(), "Accept": "application/octet-stream" if binary else "application/json"}
        r = requests.get(self.base + path, params=params, headers=headers, timeout=120, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):  # STL/translation downloads redirect
            r = requests.get(r.headers["Location"], headers=headers, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"Onshape {r.status_code} for {path}: {r.text[:300]}")
        f.write_bytes(r.content)
        return r.content if binary else r.json()


@dataclass
class Occ:
    """One leaf part occurrence in the flattened assembly."""

    path: tuple[str, ...]
    name: str
    T: np.ndarray  # world transform (m)
    part: dict  # instance record (documentId, elementId, partId, documentMicroversion, configuration)
    fixed: bool = False
    rigid_group: tuple[str, ...] | None = None  # path of the enclosing rigid sub-assembly, if any


@dataclass
class Mate:
    name: str
    feature_id: str
    type: str
    occ: list[tuple[str, ...]]  # two occurrence paths (may point at sub-assemblies)
    cs: list[np.ndarray]  # two 4x4 mate frames in their occurrence's frame
    limits: tuple[float, float] | None = None
    planar_limits: dict | None = None  # PLANAR: {"x": (lo, hi), "y": ..., "z": ...} (z is the rotation)
    axis_world: np.ndarray | None = None  # overrides the mate frame's z as the joint axis (planar pairs -> slide)
    relations: list = field(default_factory=list)


def _cs(m: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = m["xAxis"], m["yAxis"], m["zAxis"], m["origin"]
    return T


def _limits(client: Client, ref: dict, mates: dict[str, Mate]) -> None:
    """Mate limits live in the feature parameters, not in the assembly definition."""
    try:
        feats = client.get(f"/api/v10/assemblies/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/e/{ref['eid']}/features")
    except Exception as e:  # noqa: BLE001  (limits are optional)
        print(f"  note: could not read mate limits ({e})")
        return
    for f in feats.get("features", []):
        msg = f.get("message", f)
        fid = msg.get("featureId")
        if fid not in mates:
            continue
        params = {}
        for p in msg.get("parameters", []):
            pm = p.get("message", p)
            params[pm.get("parameterId")] = pm.get("expression", pm.get("value"))
        if str(params.get("limitsEnabled")).lower() != "true":
            continue
        if mates[fid].type == "PLANAR":
            lim = {k: (parse_quantity(params.get(f"limit{p}Min")), parse_quantity(params.get(f"limit{p}Max")))
                   for k, p in (("x", "X"), ("y", "Y"), ("z", "AxialZ"))}
            mates[fid].planar_limits = {k: v for k, v in lim.items() if None not in v}
            continue
        # sliders use limitZ*, rotations limitAxialZ* (both are always present)
        pre = "limitZ" if mates[fid].type == "SLIDER" else "limitAxialZ"
        lo, hi = parse_quantity(params.get(pre + "Min")), parse_quantity(params.get(pre + "Max"))
        if lo is not None and hi is not None:
            mates[fid].limits = (lo, hi)


def read_assembly(client: Client, ref: dict, flexible: bool = True, rigid_patterns: tuple = ()):
    """Flatten the assembly: leaf part occurrences (world frames) + mates (global paths) + relations."""
    try:
        asm = client.get(f"/api/v10/assemblies/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/e/{ref['eid']}",
                         {"includeMateFeatures": "true", "includeMateConnectors": "true", "includeNonSolids": "true"})
    except RuntimeError as e:
        if "must be an assembly" not in str(e):
            raise
        els = client.get(f"/api/v10/documents/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/elements")
        base = f"https://{ref['host']}/documents/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/e/"
        lines = [f"  {e['name']}: {base}{e['id']}" for e in els if e["elementType"] == "ASSEMBLY"]
        raise SystemExit("That link is not an assembly tab (it's a Part Studio or another element). "
                         "Assemblies in this document:\n" + "\n".join(lines)) from None
    root = asm["rootAssembly"]
    subs = {(s["documentId"], s["elementId"], s.get("configuration", "")): s for s in asm.get("subAssemblies", [])}
    occ_T = {tuple(o["path"]): np.array(o["transform"], float).reshape(4, 4) for o in root["occurrences"]}
    # parts from linked documents must be fetched at the linked version
    versions = {(q["documentId"], q["elementId"], q["partId"]): q.get("documentVersion")
                for q in asm.get("parts", []) if q.get("documentVersion")}
    # keep solid and composite parts; skip sheets, wires and frame_* marker parts
    body_type = {(q["documentId"], q["elementId"], q["partId"]): q.get("bodyType", "solid")
                 for q in asm.get("parts", [])}
    in_frame, in_other = set(), set()
    for d in [root, *asm.get("subAssemblies", [])]:
        for f in d["features"]:
            if f["featureType"] != "mate":
                continue
            ids = {e["matedOccurrence"][-1] for e in f["featureData"].get("matedEntities", [])
                   if e.get("matedOccurrence")}  # empty path = mated to the assembly origin
            (in_frame if f["featureData"].get("name", "").lower().startswith("frame_") else in_other).update(ids)
    frame_ids = in_frame - in_other  # parts attached ONLY through frame_* mates are markers
    occ_fixed = {tuple(o["path"]) for o in root["occurrences"] if o.get("fixed")}
    # dof_ convention: top-level instances are links, sub-assemblies are rigid
    uses_dof = any(f["featureType"] == "mate" and f["featureData"].get("name", "").lower().startswith("dof_")
                   for d in [root, *asm.get("subAssemblies", [])] for f in d["features"])
    if uses_dof:
        flexible = False
    occs: dict[tuple, Occ] = {}
    mates: list[Mate] = []
    relations: list[dict] = []
    groups: list[list[tuple]] = []  # Group mates: everything listed moves as one rigid body

    def walk(defn: dict, prefix: tuple, rigid: tuple | None):
        for inst in defn["instances"]:
            if inst.get("suppressed"):
                continue
            path = prefix + (inst["id"],)
            if inst["type"] == "Assembly":
                sub = subs[(inst["documentId"], inst["elementId"], inst.get("configuration", ""))]
                # sub-assemblies are flexible by default; the API doesn't expose "Make rigid"
                named_rigid = any(re.search(p, inst["name"], re.I) for p in rigid_patterns)
                walk(sub, path, rigid if rigid else (path if (named_rigid or not flexible) else None))
            elif inst["type"] == "Part":
                if not inst.get("partId"):  # e.g. a surface or deleted part: no mesh to fetch
                    continue
                bt = body_type.get((inst["documentId"], inst["elementId"], inst["partId"]), "solid")
                if bt not in ("solid", "composite") or inst["id"] in frame_ids:
                    continue
                part = dict(inst)
                v = versions.get((inst["documentId"], inst["elementId"], inst["partId"]))
                if v and not part.get("documentVersion"):
                    part["documentVersion"] = v
                occs[path] = Occ(path, inst["name"], occ_T[path], part, path in occ_fixed, rigid)
        for f in defn.get("features", []):
            if f.get("suppressed"):
                continue
            d = f["featureData"]
            if f["featureType"] == "mate":
                ents = d["matedEntities"]
                ents = [e for e in ents if e.get("matedOccurrence")]  # drop "mated to origin" entities
                mates.append(Mate(d.get("name", f["id"]), f["id"], d["mateType"],
                                  [prefix + tuple(e["matedOccurrence"]) for e in ents], [_cs(e["matedCS"]) for e in ents]))
            elif f["featureType"] == "mateRelation":
                relations.append({**d, "prefix": prefix})
            elif f["featureType"] == "mateGroup":
                groups.append([prefix + tuple(o["occurrence"]) for o in d.get("occurrences", [])])
        # pattern copies are rigid with their seed
        for pat in defn.get("patterns", []):
            if pat.get("suppressed"):
                continue
            for seed, copies in pat.get("seedToPatternInstances", {}).items():
                for cp in copies:
                    groups.append([prefix + (seed,), prefix + (cp,)])

    walk(root, (), None)
    by_id = {m.feature_id: m for m in mates}
    _limits(client, ref, by_id)
    return occs, occ_T, mates, relations, groups


def _mesh_and_mass(client: Client, occ: Occ, density: float, top_did: str | None = None):
    p = occ.part
    linked = bool(top_did and p["documentId"] != top_did)
    if linked and p.get("documentVersion"):
        wvm, wvmid = "v", p["documentVersion"]  # linked documents are only readable at the linked version
    elif p.get("documentMicroversion"):
        wvm, wvmid = "m", p["documentMicroversion"]
    else:
        raise ValueError(f"part {occ.name} has no document version/microversion")
    base = f"/api/v10/parts/d/{p['documentId']}/{wvm}/{wvmid}/e/{p['elementId']}/partid/{quote(p['partId'], safe='')}"
    cfg = {"configuration": p.get("configuration", "")}
    if linked:
        cfg["linkDocumentId"] = top_did
    stl = client.get(base + "/stl", {**cfg, "mode": "binary", "units": "meter", "grouping": "true"}, binary=True)
    mesh = trimesh.load(io.BytesIO(stl), file_type="stl", force="mesh")
    # Onshape's STL doesn't share vertices: weld them so parts are closed solids
    mesh.merge_vertices(digits_vertex=6)
    mesh.remove_unreferenced_vertices()
    mesh.apply_transform(occ.T)
    mp = client.get(base + "/massproperties", {**cfg, "useMassPropertyOverrides": "true"})
    body = next(iter(mp.get("bodies", {}).values()), {})
    R = occ.T[:3, :3]
    if body.get("hasMass", False) and body.get("mass", [0])[0] > 0:
        mass = float(body["mass"][0])
        com = (occ.T @ np.r_[np.array(body["centroid"][:3], float), 1.0])[:3]
        I = np.array(body["inertia"][:9], float).reshape(3, 3)  # about the centroid, part frame
        if np.isfinite([mass, *com, *I.ravel()]).all():  # else: fall through to the mesh estimate
            return mesh, mass, com, R @ I @ R.T, float(body.get("volume", [mesh.volume])[0])
    # no material assigned in Onshape: uniform density on the mesh
    m = mesh.copy()
    if not m.is_watertight:
        m = m.convex_hull
    m.density = density
    return mesh, float(m.mass), np.array(m.center_mass), np.array(m.moment_inertia), float(m.volume)




def _resolve_planar(planar: list, uf: UnionFind, rep, occ_T: dict, review: list) -> list:
    """Planar mates between the same two bodies, taken together.

    One planar mate leaves 3 DOF (two slides and a spin). Each extra mate with a different plane normal removes
    more: two leave a slide along the planes' intersection, three independent ones are a rigid joint (a common
    way to fix a part). Returns the mates to treat as joints; rigid pairs are welded into ``uf``.
    """
    from collections import defaultdict
    from dataclasses import replace

    out: list = []
    pending = list(planar)
    while pending:
        by_pair: dict = defaultdict(list)
        for m in pending:
            a, b = uf.find(rep(m.occ[0])), uf.find(rep(m.occ[1]))
            if a != b:
                by_pair[tuple(sorted((a, b), key=str))].append(m)
        pending, welded = [], False
        for ms in by_pair.values():
            normals = np.array([(occ_T[m.occ[0]] @ m.cs[0])[:3, 2] for m in ms])
            rank = int((np.linalg.svd(normals / np.linalg.norm(normals, axis=1, keepdims=True),
                                      compute_uv=False) > 0.05).sum())
            first = ms[0]
            if rank >= 3:
                uf.union(rep(first.occ[0]), rep(first.occ[1]))
                welded = True
                review.append(f"{len(ms)} planar mates ({', '.join(m.name for m in ms[:3])}) with independent "
                              f"normals fix {first.name}'s two bodies together: welded")
            elif rank == 2:
                n = np.cross(*[v for v in normals[[0, int(np.argmax(np.abs(normals @ normals[0]) < 0.95))]]])
                out.append(replace(first, type="SLIDER", axis_world=n / np.linalg.norm(n), limits=None))
                review.append(f"{len(ms)} planar mates ({', '.join(m.name for m in ms[:3])}) leave one slide along "
                              f"their planes' intersection: prismatic")
            else:
                out.append(first)  # parallel planes: one planar mate's worth of freedom
        if welded:  # welding changes which bodies are the same: re-evaluate the rest
            pending = [m for ms in by_pair.values() for m in ms]
            out = []
    return out


def _planar_joints(m: Mate, F: list, c: int, jname: str, parent: str, child: str, spec: dict,
                   review: list) -> tuple[dict, dict]:
    """A PLANAR mate (slide x, slide y, spin z in the mate frame) as a chain: parent -x-> dummy -y-> dummy -z-> child.

    URDF has no 3-DOF joint that every simulator reads, so two massless links carry the slides. The joints are
    passive unless the spec names an actuator.
    """
    ov = spec.get("joints", {}) or {}
    x_w, y_w, z_w = (F[0][:3, k] for k in range(3))
    d = F[0][:3, 3] - F[1][:3, 3]
    sign = 1.0 if c == 0 else -1.0
    lim = m.planar_limits or {}
    if not lim:
        review.append(f"joints.{jname}: planar mate without limits; slides get a placeholder +/-0.1 m")
    origin = F[c][:3, 3].copy()
    links, joints = {}, {}
    prev = parent
    for tag, axis, q_now, kind in (("x", x_w, float(d @ x_w), "prismatic"), ("y", y_w, float(d @ y_w), "prismatic"),
                                   ("z", z_w, 0.0, "continuous")):
        name = f"{jname}_{tag}"
        end = child if tag == "z" else f"{jname}_{tag}_link"
        lo = hi = 0.0
        if kind == "prismatic":
            lo, hi = sorted(sign * (v - q_now) for v in lim.get(tag, (-0.1 + q_now, 0.1 + q_now)))
        elif tag in lim:
            kind = "revolute"
            lo, hi = sorted(sign * v for v in lim[tag])
        if end != child:
            links[end] = Link(end)
            links[end].origin = origin.copy()
        j_ov = ov.get(name, {})
        joints[name] = Joint(name=name, type=kind, parent=prev, child=end, origin=origin.copy(), axis=axis,
                             lower=j_ov.get("limits", [lo, hi])[0], upper=j_ov.get("limits", [lo, hi])[1],
                             effort=j_ov.get("effort", 10.0), velocity=j_ov.get("velocity", 5.0),
                             actuator=dict((spec.get("actuators", {}) or {}).get(name, {"kind": "none"})))
        prev = end
    return links, joints


def _named_actuator(jname: str, spec: dict) -> dict:
    """The spec's actuator; unless the spec names the joint, ``*passive*`` -> none, ``*_speed`` -> velocity."""
    acts = spec.get("actuators", {}) or {}
    a = dict(_per(acts, jname))
    if jname in acts:
        return a
    if "passive" in jname.lower():
        return {"kind": "none"}
    if jname.lower().endswith("_speed"):
        return {"kind": "velocity", "kv": a.get("kv", 1.0)}
    return a


def _add_closures(robot: Robot, closing: list, gname: dict, group_of, occ_T: dict, explicit_act: set,
                  review: list) -> None:
    """``closing_*`` mates become point constraints at the mate origin. Loop joints not on the base get no
    actuator (in a parallel mechanism the motors are at the base; servos elsewhere would fight the loop)."""
    tree_parent = {j.child: j for j in robot.joints.values()}

    def path_to_root(link):
        out = []
        while link in tree_parent:
            out.append(tree_parent[link])
            link = tree_parent[link].parent
        return out

    for m in closing:
        ga, gb = group_of(m.occ[0]), group_of(m.occ[1])
        la, lb = gname.get(ga), gname.get(gb)
        if la is None or lb is None or la == lb:
            review.append(f"mate {m.name}: closing mate between parts of one link (or off the tree); ignored")
            continue
        world = (occ_T[m.occ[0]] @ m.cs[0])[:3, 3]
        anchors = [np.linalg.inv(robot.links[n].pose()) @ np.r_[world, 1.0] for n in (la, lb)]
        robot.closures.append({"name": slug(m.name, lower=True), "link1": la, "link2": lb,
                               "anchor1": anchors[0][:3], "anchor2": anchors[1][:3]})
        pa, pb = path_to_root(la), path_to_root(lb)
        loop = {j.name for j in pa} ^ {j.name for j in pb}
        passive = sorted(n for n in loop if robot.joints[n].parent != robot.root and n not in explicit_act)
        for n in passive:
            robot.joints[n].actuator = {"kind": "none"}
        review.append(f"mate {m.name}: loop closure {la} <-> {lb} (MJCF equality; URDF-only simulators need a "
                      f"constraint added by hand); passive loop joints: {passive}")


def build_from_onshape(spec: dict, base: Path, client: Client | None = None) -> Robot:
    ref = parse_url(spec["source"])
    client = client or Client(ref["host"])
    flexible = spec.get("subassemblies", "flexible") != "rigid"
    occs, occ_T, mates, relations, rigid_groups = read_assembly(
        client, ref, flexible, tuple(spec.get("rigid_subassemblies", [])))
    review: list[str] = []

    def leaves_under(path):  # a mate may reference a sub-assembly occurrence
        return [p for p in occs if p[: len(path)] == path]

    def rep(path):
        ls = leaves_under(path)
        if not ls:
            raise ValueError(f"mate references unknown occurrence {path}")
        return ls[0]

    # mates to instances the API didn't return (markers, sketches) aren't joints
    kept = []
    for m in mates:
        if len(m.occ) < 2:
            review.append(f"mate {m.name}: has a single entity (mated to the origin/assembly); ignored")
        elif all(leaves_under(p) for p in m.occ):
            kept.append(m)
        else:
            review.append(f"mate {m.name}: references a non-solid or missing instance; ignored")
    mates = kept

    use_dof = any(m.name.lower().startswith("dof_") for m in mates)
    if use_dof:
        n_other = sum(1 for m in mates if not m.name.lower().startswith("dof_")
                      and m.type in ("REVOLUTE", "SLIDER", "CYLINDRICAL", "PIN_SLOT"))
        review.append(f"'dof_' naming found: top-level instances are links (sub-assemblies rigid), only dof_* "
                      f"mates are joints, fastened mates join, other mates ignored ({n_other} revolute/slider-type)")

    uf = UnionFind()
    for p, o in occs.items():
        uf.find(p)
        if o.rigid_group:
            uf.union(p, rep(o.rigid_group))
    for grp in rigid_groups:  # Group mates
        leaves = [rep(p) for p in grp if leaves_under(p)]
        for a in leaves[1:]:
            uf.union(leaves[0], a)
    moving, closing, planar = [], [], []
    for m in mates:
        a, b = rep(m.occ[0]), rep(m.occ[1])
        if m.name.lower().startswith("closing_") and m.type in ("REVOLUTE", "SLIDER", "CYLINDRICAL", "PIN_SLOT",
                                                                  "BALL", "FASTENED"):
            closing.append(m)
            continue
        if not m.name.lower().startswith("dof_") and m.type != "FASTENED" and any(
                all(is_fastener(occs[l].name) for l in leaves_under(o)) for o in m.occ):
            uf.union(a, b)  # a screw or nut mated with a slot/cylindrical mate is still just fixed to its part
            continue
        if use_dof and not m.name.lower().startswith("dof_"):
            # dof_ mode: fastened mates still join, other mates only align parts
            if m.type == "FASTENED":
                uf.union(a, b)
            continue
        if m.type == "FASTENED":
            uf.union(a, b)
        elif m.type == "PLANAR":
            planar.append(m)
        elif m.type in ("REVOLUTE", "SLIDER", "CYLINDRICAL", "PIN_SLOT"):
            moving.append(m)
            if m.type in ("CYLINDRICAL", "PIN_SLOT"):
                review.append(f"mate {m.name}: {m.type} (rotates AND slides) exported as revolute")
        else:
            review.append(f"mate {m.name}: {m.type} not supported as a joint; ignored")

    moving += _resolve_planar(planar, uf, rep, occ_T, review)

    groups: dict = {}
    for p in occs:
        groups.setdefault(uf.find(p), []).append(p)

    # --- tree over groups: BFS from the fixed / heaviest group
    density = spec.get("default_density", 1200.0)
    part_data = {p: _mesh_and_mass(client, o, density, ref["did"]) for p, o in occs.items()}
    gmass = {g: sum(part_data[p][1] for p in ps) for g, ps in groups.items()}
    fixed_groups = [g for g, ps in groups.items() if any(occs[p].fixed for p in ps)]
    root = max(fixed_groups or groups, key=lambda g: gmass[g])
    adj: dict = {}
    for m in moving:
        ga, gb = uf.find(rep(m.occ[0])), uf.find(rep(m.occ[1]))
        if ga == gb:
            review.append(f"mate {m.name}: both sides are fastened together elsewhere; ignored")
            continue
        adj.setdefault(ga, []).append((gb, m, 0))
        adj.setdefault(gb, []).append((ga, m, 1))
    parent_of, seen, order, tree_mates = {}, {root}, [root], set()
    for g in order:
        for nb, m, side in sorted(adj.get(g, []), key=lambda e: e[1].name):
            if nb in seen:
                continue
            seen.add(nb)
            order.append(nb)
            parent_of[nb] = (g, m, side)
            tree_mates.add(m.feature_id)
    for m in moving:
        ga, gb = uf.find(rep(m.occ[0])), uf.find(rep(m.occ[1]))
        if ga != gb and m.feature_id not in tree_mates and ga in seen and gb in seen:
            review.append(f"mate {m.name}: closes a kinematic loop; left out of the tree")
    orphans = [g for g in groups if g not in seen]
    if orphans:  # no moving mate connects them: weld to the base
        n = sum(len(groups[g]) for g in orphans)
        for g in orphans:
            groups[root] += groups.pop(g)
        review.append(f"{n} part(s) in {len(orphans)} group(s) have no mate path to the base; merged into it")

    # --- names
    taken: set[str] = set()

    def link_name(ps):
        big = max(ps, key=lambda p: part_data[p][1])
        n = slug(occs[big].name, lower=True)
        k, out = 2, n
        while out in taken:
            out, k = f"{n}_{k}", k + 1
        taken.add(out)
        return out

    gname = {g: link_name(groups[g]) for g in order}
    if "base_link" not in gname.values():
        gname[root] = "base_link"

    # --- IR
    links: dict[str, Link] = {}
    joints: dict[str, Joint] = {}
    joint_of_mate: dict[str, str] = {}
    overrides = spec.get("joints", {}) or {}
    for g in order:
        name = gname[g]
        link = Link(name)
        for p in groups[g]:
            mesh, mass, com, I, vol = part_data[p]
            part = Part(name=f"{name}/{slug(occs[p].name, lower=True)}", shape=None, link=name, mesh=mesh,
                            mass=mass, com=com, inertia=I, volume=vol)
            link.parts.append(part)
        if g in parent_of:
            pg, m, side = parent_of[g]
            c = 1 - side  # index of this (child) side in the mate
            F = [occ_T[m.occ[k]] @ m.cs[k] for k in (0, 1)]  # both mate frames in world
            axis_w = F[0][:3, 2] if m.axis_world is None else m.axis_world
            raw = m.name[4:] if m.name.lower().startswith("dof_") else m.name
            flip = raw.lower().endswith("_inv")
            jname = slug(raw[:-4] if flip else raw, lower=True)
            while jname in joints:
                jname += "_"
            if m.type == "PLANAR":
                pl, pj = _planar_joints(m, F, c, jname, gname[pg], name, spec, review)
                links.update(pl)
                joints.update(pj)
                joint_of_mate[m.feature_id] = jname + "_z"
                link.origin = F[c][:3, 3].copy()
                links[name] = link
                continue
            jtype = "prismatic" if m.type == "SLIDER" else "revolute"
            # Onshape limits are entity 0's motion relative to entity 1 from the mate's zero; our zero is the
            # current pose and the child moves along +axis_w: shift by the current value, flip if child is 1
            if jtype == "prismatic":
                q_now = float(np.dot(F[0][:3, 3] - F[1][:3, 3], axis_w))
            else:
                x0, x1 = F[0][:3, 0], F[1][:3, 0]
                q_now = float(np.arctan2(np.dot(np.cross(x1, x0), axis_w), np.dot(x0, x1)))
            sign = 1.0 if c == 0 else -1.0
            lo = hi = None
            if m.limits is not None:
                a, b = sorted(sign * (v - q_now) for v in m.limits)
                lo, hi = a, b
            elif jtype == "revolute":
                jtype = "continuous"
            else:
                lo, hi = -0.1, 0.1
                review.append(f"joints.{jname}: slider without limits in Onshape; placeholder +/-0.1 m")
            Tj = F[c]
            if flip:  # "_inv": same motion, measured the other way round
                axis_w = -axis_w
                lo, hi = (-hi if hi is not None else None), (-lo if lo is not None else None)
            ov = overrides.get(jname, {})
            lo, hi = ov.get("limits", [lo if lo is not None else 0.0, hi if hi is not None else 0.0])
            joints[jname] = Joint(
                name=jname, type=ov.get("type", jtype), parent=gname[pg], child=name,
                origin=Tj[:3, 3].copy(), axis=axis_w * ov.get("axis_sign", 1), lower=lo, upper=hi,
                effort=ov.get("effort", 10.0), velocity=ov.get("velocity", 5.0),
                damping=_per(spec.get("dynamics", {}), jname).get("damping", 0.0),
                friction=_per(spec.get("dynamics", {}), jname).get("friction", 0.0),
                armature=_per(spec.get("dynamics", {}), jname).get("armature", 0.0),
                mimic=ov.get("mimic"), actuator=_named_actuator(jname, spec))
            joint_of_mate[m.feature_id] = jname
            link.origin = Tj[:3, 3].copy()
        links[name] = link

    # --- mate relations -> mimic
    for rel in relations:
        ids = [x.get("featureId") for x in rel.get("mates", [])]
        js = [joint_of_mate.get(i) for i in ids]
        if len(js) == 2 and all(js) and joints[js[1]].mimic is None:
            ratio = float(rel.get("relationRatio", 1.0) or 1.0) * (-1 if rel.get("reverseDirection") else 1)
            joints[js[1]].mimic = {"joint": js[0], "multiplier": ratio, "offset": 0.0}
            joints[js[1]].actuator = {"kind": "none"}
            review.append(f"joints.{js[1]}: mimics {js[0]} x {ratio} from {rel.get('relationType')} relation")

    for link in links.values():
        if not link.parts:  # a massless carrier between planar-joint axes
            continue
        link.mass, link.com, link.inertia = combine_inertia(link.parts, link.origin)
        for issue in check_inertia(link.name, link.inertia, link.mass):
            print("  WARNING", issue)
        link.visuals = {"mesh": trimesh.util.concatenate([link.to_link(p.mesh) for p in link.parts])}
        link.visual_materials = {"mesh": "onshape_grey"}

    robot = Robot(spec.get("robot", "onshape_robot"), spec, links, joints, [], gname[root],
                  floating_base=spec.get("base", "fixed") == "floating")
    _add_closures(robot, closing, {g: gname[g] for g in order}, lambda o: uf.find(rep(o)), occ_T,
                  set((spec.get("actuators") or {}).keys()), review)
    robot.materials["onshape_grey"] = "0.72 0.74 0.78 1"
    robot.review = review  # type: ignore[attr-defined]
    for r in review:
        print("  REVIEW", r)
    return robot
