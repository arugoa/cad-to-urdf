"""Render figures for the docs (matplotlib, no GPU needed).

    python examples/arm4/render.py [robot] [modes]

* docs/img/arm4_visual_vs_collision.png  - the generated URDF, visual vs collision geometry, two poses
* docs/img/collision_modes.png           - every collision mode on three representative links
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yourdfpy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
IMG = ROOT / "docs" / "img"
sys.path.insert(0, str(ROOT))
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def draw(ax, mesh: trimesh.Trimesh, color, alpha=1.0, edge=False):
    tris = mesh.triangles
    shade = 0.45 + 0.55 * np.clip(mesh.face_normals @ LIGHT, 0, 1)
    fc = np.clip(np.outer(shade, np.array(color[:3])), 0, 1)
    pc = Poly3DCollection(tris, facecolors=np.c_[fc, np.full(len(fc), alpha)],
                          edgecolors=(0, 0, 0, 0.25) if edge else "none", linewidths=0.2)
    ax.add_collection3d(pc)


def frame(ax, meshes, elev=18, azim=-55, zoom=1.0):
    b = np.array([m.bounds for m in meshes])
    lo, hi = b[:, 0].min(0), b[:, 1].max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def scene_meshes(robot: yourdfpy.URDF, collision: bool):
    scene = robot.collision_scene if collision else robot.scene
    out = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        color = m.visual.face_colors[0] / 255.0 if hasattr(m.visual, "face_colors") and not collision else None
        out.append((m, color))
    return out


def fig_robot():
    urdf = HERE / "output" / "arm4.urdf"
    robot = yourdfpy.URDF.load(str(urdf), build_scene_graph=True, build_collision_scene_graph=True,
                               load_meshes=True, load_collision_meshes=True)
    poses = {"home": {}, "ready": {"shoulder_pitch": 0.6, "elbow": 1.2, "shoulder_yaw": 0.5, "finger_left": 0.008,
                                   "finger_right": 0.008}}
    fig = plt.figure(figsize=(13, 5.6), dpi=130)
    k = 1
    for pname, q in poses.items():
        robot.update_cfg({n: q.get(n, 0.0) for n in robot.actuated_joint_names})
        for collision in (False, True):
            ax = fig.add_subplot(1, 4, k, projection="3d")
            k += 1
            ms = scene_meshes(robot, collision)
            palette = plt.cm.tab10(np.linspace(0, 1, 10))
            for i, (m, c) in enumerate(ms):
                col = palette[i % 10] if collision else (c if c is not None else (0.7, 0.7, 0.75, 1))
                draw(ax, m, col, alpha=0.95 if collision else 1.0, edge=collision)
            frame(ax, [m for m, _ in ms], zoom=1.3)
            ax.set_title(f"{pname}: {'collision' if collision else 'visual'}\n"
                         f"{len(ms)} geoms, {sum(len(m.faces) for m, _ in ms):,} tris", fontsize=9)
    fig.suptitle("arm4: URDF generated from arm4.step + robot_spec.yaml", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "arm4_visual_vs_collision.png", bbox_inches="tight")
    plt.close(fig)


def fig_modes():
    from cad2urdf import collision, model

    robot = model.build(HERE / "robot_spec.yaml")
    modes = [("box", {"mode": "box"}), ("primitives", {"mode": "primitives", "min_part_fraction": 0.02}),
             ("hull", {"mode": "hull"}), ("CoACD 0.05", {"mode": "decompose", "threshold": 0.05, "max_hulls": 8}),
             ("CoACD 0.02", {"mode": "decompose", "threshold": 0.02, "max_hulls": 32})]
    links = ["forearm", "upper_arm", "finger_left"]
    fig = plt.figure(figsize=(3.0 * (len(modes) + 1), 3.2 * len(links)), dpi=110)
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    for r, lname in enumerate(links):
        link = robot.links[lname]
        cad_meshes = list(link.visuals.values())
        ax = fig.add_subplot(len(links), len(modes) + 1, r * (len(modes) + 1) + 1, projection="3d")
        for m in cad_meshes:
            draw(ax, m, (0.72, 0.74, 0.78, 1))
        frame(ax, cad_meshes, elev=20, azim=-35, zoom=1.3)
        ax.set_title(f"{lname}: CAD\n{sum(len(m.faces) for m in cad_meshes):,} tris", fontsize=9)
        for c, (label, cfg) in enumerate(modes):
            link.collisions = collision.link_collisions(link, cfg, 64)
            met = collision.metrics(link, n=30000)
            ax = fig.add_subplot(len(links), len(modes) + 1, r * (len(modes) + 1) + c + 2, projection="3d")
            meshes = [collision.geom_to_mesh(g) for g in link.collisions]
            for i, m in enumerate(meshes):
                draw(ax, m, palette[i % 20], alpha=0.9, edge=True)
            frame(ax, cad_meshes, elev=20, azim=-35, zoom=1.3)
            ax.set_title(f"{label}\n{met['geoms']} geoms  IoU {met['iou']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "collision_modes.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["robot", "modes"]
    if "robot" in which:
        fig_robot()
    if "modes" in which:
        fig_modes()
    print("wrote", IMG)
