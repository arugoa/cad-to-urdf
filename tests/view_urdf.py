"""Interactive browser viewer for any URDF this repo produces (robots and static scenes).

    python tests/view_urdf.py examples/arm4/output            # an output dir (picks <robot>.urdf)
    python tests/view_urdf.py build/hero/hero_2027.urdf --port 8081

Open http://localhost:8080. Controls:
  * a slider per actuated joint (mimic joints follow their leader);
  * toggles for visual meshes, collision geometry, and link frames;
  * a dropdown of the SRDF named poses (group_state), if an SRDF sits next to the URDF;
  * "sweep" animates every joint through its range to spot bad axes/limits quickly.

Not collected by pytest (the file name doesn't start with test_). Needs `pip install viser`.
If ROS is sourced, run with `env -u PYTHONPATH`.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf


def resolve(path: Path) -> Path:
    if path.is_dir():
        cands = [p for p in path.glob("*.urdf") if not p.name.startswith("_")]
        if not cands:
            raise SystemExit(f"no .urdf in {path}")
        return cands[0]
    return path


def named_poses(urdf: Path) -> dict[str, dict[str, float]]:
    srdf = urdf.with_suffix(".srdf")
    if not srdf.exists():
        return {}
    out = {}
    for gs in ET.parse(srdf).getroot().findall("group_state"):
        out[gs.get("name")] = {j.get("name"): float(j.get("value")) for j in gs.findall("joint")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="URDF file or an output directory containing one")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    urdf_path = resolve(args.path).resolve()

    robot = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True,
                               build_collision_scene_graph=True, load_collision_meshes=True)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=4, height=4, cell_size=0.1)
    vu = ViserUrdf(server, robot, root_node_name="/robot", load_meshes=True, load_collision_meshes=True,
                   collision_mesh_color_override=(0.2, 0.6, 1.0, 0.45))
    vu.show_collision = False

    names = vu.get_actuated_joint_names()
    limits = vu.get_actuated_joint_limits()
    n_links, n_joints = len(robot.link_map), len(robot.joint_map)
    server.gui.add_markdown(f"**{urdf_path.name}**  \n{n_links} links, {n_joints} joints, {len(names)} actuated")

    with server.gui.add_folder("Display"):
        show_vis = server.gui.add_checkbox("visual meshes", True)
        show_col = server.gui.add_checkbox("collision geometry", False)
        show_frames = server.gui.add_checkbox("link frames", False)

    frames = {}

    def update_frames():
        for name in robot.link_map:
            T = robot.get_transform(name)
            if name not in frames:
                frames[name] = server.scene.add_frame(f"/frames/{name}", axes_length=0.05, axes_radius=0.002)
            frames[name].wxyz = viser.transforms.SO3.from_matrix(T[:3, :3]).wxyz
            frames[name].position = T[:3, 3]
            frames[name].visible = show_frames.value

    sliders = []
    with server.gui.add_folder("Joints"):
        for name, (lo, hi) in zip(names, limits.values() if isinstance(limits, dict) else limits):
            lo = -np.pi if lo is None else lo
            hi = np.pi if hi is None else hi
            if hi <= lo:
                lo, hi = lo - 1.0, lo + 1.0
            init = float(np.clip(0.0, lo, hi))
            sliders.append(server.gui.add_slider(name, min=float(lo), max=float(hi),
                                                 step=float((hi - lo) / 500), initial_value=init))

    def apply():
        q = np.array([s.value for s in sliders])
        vu.update_cfg(q)
        robot.update_cfg(q)
        update_frames()

    for s in sliders:
        s.on_update(lambda _: apply())
    show_vis.on_update(lambda _: setattr(vu, "show_visual", show_vis.value))
    show_col.on_update(lambda _: setattr(vu, "show_collision", show_col.value))
    show_frames.on_update(lambda _: update_frames())

    poses = named_poses(urdf_path)
    with server.gui.add_folder("Poses"):
        if poses:
            dd = server.gui.add_dropdown("SRDF pose", options=["(none)", *poses])

            @dd.on_update
            def _(_):
                q = poses.get(dd.value, {})
                for s, n in zip(sliders, names):
                    if n in q:
                        s.value = float(np.clip(q[n], s.min, s.max))
        zero = server.gui.add_button("zero")
        sweep = server.gui.add_button("sweep all joints")

    @zero.on_click
    def _(_):
        for s in sliders:
            s.value = float(np.clip(0.0, s.min, s.max))

    sweeping = {"on": False}

    @sweep.on_click
    def _(_):
        sweeping["on"] = not sweeping["on"]

    apply()
    print(f"viewing {urdf_path}  ->  http://localhost:{args.port}")
    t0 = time.time()
    while True:
        if sweeping["on"] and sliders:
            ph = (time.time() - t0) * 0.5
            for k, s in enumerate(sliders):
                s.value = float(s.min + (s.max - s.min) * 0.5 * (1 + np.sin(ph + k)))
        time.sleep(0.05)


if __name__ == "__main__":
    main()
