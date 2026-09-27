"""Front end #3: a live Onshape assembly, read directly through the Onshape REST API.

Reads the assembly's own mates. No naming convention is required:

* FASTENED mates and parts of one rigid sub-assembly  -> same link
* REVOLUTE                                            -> revolute (or continuous if the mate has no limits)
* SLIDER                                              -> prismatic
* CYLINDRICAL / PIN_SLOT                              -> revolute, flagged REVIEW (they also slide)
* BALL / PLANAR / PARALLEL / ...                      -> ignored, flagged REVIEW
* GEAR / RACK_AND_PINION / SCREW / LINEAR relations   -> mimic on the follower joint
* mate limits (limitsEnabled, limit*Min/Max)          -> joint limits

Mass properties come from Onshape (the materials you assigned); meshes are
fetched per part as binary STL in metres. Every response is cached on disk
(``~/.cache/cad2urdf/onshape``) so re-runs are offline and repeatable.

Auth: API keys (Onshape settings -> Developer -> API keys; see docs/ONSHAPE_API_KEYS.md), exported as
ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY (HTTP Basic auth). ONSHAPE_API
defaults to the document URL's domain, so Enterprise domains work.

Spec keys used here (all optional besides ``source``):

    source: https://<domain>/documents/<did>/w/<wid>/e/<eid>
    subassemblies: flexible | rigid   # flexible (default, Onshape's behaviour): sub-assembly mates cascade up
    rigid_subassemblies: [regex, ...] # sub-assembly instance names to treat as one rigid body ("Make rigid"
                                      # in Onshape is not exposed by the API)
    default_density: 1200             # for parts without an Onshape material (kg/m^3)
    joints / dynamics / actuators / collision / srdf    # as elsewhere
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlparse

import numpy as np
import trimesh

from . import cad
from .model import Joint, Link, Robot, _per, check_inertia, combine_inertia

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
        # sliders store translation limits in limitZ*, revolute/cylindrical rotation limits in limitAxialZ*
        # (seen on the live API; both families are always present, only the matching one is meaningful)
        pre = "limitZ" if mates[fid].type == "SLIDER" else "limitAxialZ"
        lo, hi = parse_quantity(params.get(pre + "Min")), parse_quantity(params.get(pre + "Max"))
        if lo is not None and hi is not None:
            mates[fid].limits = (lo, hi)


def read_assembly(client: Client, ref: dict, flexible: bool = True, rigid_patterns: tuple = ()):
    """Flatten the assembly: leaf part occurrences (world frames) + mates (global paths) + relations."""
    try:
        asm = client.get(f"/api/v10/assemblies/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/e/{ref['eid']}",
                         {"includeMateFeatures": "true", "includeMateConnectors": "true", "includeNonSolids": "false"})
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
    # linked-document parts must be fetched through the linked *version*; parts[] carries it
    versions = {(q["documentId"], q["elementId"], q["partId"]): q.get("documentVersion")
                for q in asm.get("parts", []) if q.get("documentVersion")}
    occ_fixed = {tuple(o["path"]) for o in root["occurrences"] if o.get("fixed")}
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
                # Onshape sub-assemblies are flexible by default (their mates' DOF cascade up to the parent).
                # The API does not expose "Make rigid", so rigid ones are named in the spec.
                named_rigid = any(re.search(p, inst["name"], re.I) for p in rigid_patterns)
                walk(sub, path, rigid if rigid else (path if (named_rigid or not flexible) else None))
            elif inst["type"] == "Part":
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
                mates.append(Mate(d.get("name", f["id"]), f["id"], d["mateType"],
                                  [prefix + tuple(e["matedOccurrence"]) for e in ents], [_cs(e["matedCS"]) for e in ents]))
            elif f["featureType"] == "mateRelation":
                relations.append({**d, "prefix": prefix})
            elif f["featureType"] == "mateGroup":
                groups.append([prefix + tuple(o["occurrence"]) for o in d.get("occurrences", [])])

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
        # part lives in a linked document (library / purchased part): Onshape grants access only
        # through the document that links to it
        cfg["linkDocumentId"] = top_did
    stl = client.get(base + "/stl", {**cfg, "mode": "binary", "units": "meter", "grouping": "true"}, binary=True)
    mesh = trimesh.load(io.BytesIO(stl), file_type="stl", force="mesh")
    # Onshape's STL does not share vertices between triangles: weld coincident vertices (1 um) so parts
    # come out as closed solids (needed for volumes, containment and exact-geometry checks)
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
        return mesh, mass, com, R @ I @ R.T, float(body.get("volume", [mesh.volume])[0])
    # no material assigned in Onshape: uniform density on the mesh
    m = mesh.copy()
    if not m.is_watertight:
        m = m.convex_hull
    m.density = density
    return mesh, float(m.mass), np.array(m.center_mass), np.array(m.moment_inertia), float(m.volume)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower() or "x"


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

    uf = _UF()
    for p, o in occs.items():
        uf.find(p)
        if o.rigid_group:
            uf.union(p, rep(o.rigid_group))
    for grp in rigid_groups:  # Group mates
        leaves = [rep(p) for p in grp if leaves_under(p)]
        for a in leaves[1:]:
            uf.union(leaves[0], a)
    moving = []
    for m in mates:
        a, b = rep(m.occ[0]), rep(m.occ[1])
        if m.type == "FASTENED":
            uf.union(a, b)
        elif m.type in ("REVOLUTE", "SLIDER", "CYLINDRICAL", "PIN_SLOT"):
            moving.append(m)
            if m.type in ("CYLINDRICAL", "PIN_SLOT"):
                review.append(f"mate {m.name}: {m.type} (rotates AND slides) exported as revolute")
        else:
            review.append(f"mate {m.name}: {m.type} not supported as a joint; ignored")

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
        n = _key(occs[big].name)
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
            part = cad.Part(name=f"{name}/{_key(occs[p].name)}", shape=None, link=name, mesh=mesh,
                            mass=mass, com=com, inertia=I, volume=vol)
            link.parts.append(part)
        if g in parent_of:
            pg, m, side = parent_of[g]
            c = 1 - side  # index of this (child) side in the mate
            F = [occ_T[m.occ[k]] @ m.cs[k] for k in (0, 1)]  # both mate frames in world
            axis_w = F[0][:3, 2]
            jname = _key(m.name)
            while jname in joints:
                jname += "_"
            jtype = "prismatic" if m.type == "SLIDER" else "revolute"
            # Onshape's mate value is the motion of entity 0 relative to entity 1 along/about the mate Z
            # axis, from the mate's own zero. (Established on a real pneumatic cylinder: SLIDER limits
            # [-4.5 in, 0] with the piston as entity 0 and mate Z pointing down only fit the geometry if
            # the piston moves UP into the barrel.) Our joint zero is the assembly's current pose and the
            # joint moves the child along +axis_w, so shift by the current value and flip the sign when
            # the child is entity 1.
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
            ov = overrides.get(jname, {})
            lo, hi = ov.get("limits", [lo if lo is not None else 0.0, hi if hi is not None else 0.0])
            joints[jname] = Joint(
                name=jname, type=ov.get("type", jtype), parent=gname[pg], child=name,
                origin=Tj[:3, 3].copy(), axis=axis_w * ov.get("axis_sign", 1), lower=lo, upper=hi,
                effort=ov.get("effort", 10.0), velocity=ov.get("velocity", 5.0),
                damping=_per(spec.get("dynamics", {}), jname).get("damping", 0.0),
                friction=_per(spec.get("dynamics", {}), jname).get("friction", 0.0),
                armature=_per(spec.get("dynamics", {}), jname).get("armature", 0.0),
                mimic=ov.get("mimic"), actuator=_per(spec.get("actuators", {}), jname))
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
        link.mass, link.com, link.inertia = combine_inertia(link.parts, link.origin)
        for issue in check_inertia(link.name, link.inertia, link.mass):
            print("  WARNING", issue)
        link.visuals = {"mesh": trimesh.util.concatenate([link.to_link(p.mesh) for p in link.parts])}
        link.visual_materials = {"mesh": "onshape_grey"}

    robot = Robot(spec.get("robot", "onshape_robot"), spec, links, joints, [], gname[root],
                  floating_base=spec.get("base", "fixed") == "floating")
    robot.materials["onshape_grey"] = "0.72 0.74 0.78 1"
    robot.review = review  # type: ignore[attr-defined]
    for r in review:
        print("  REVIEW", r)
    return robot
