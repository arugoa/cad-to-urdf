"""View generated robots and scenes: in the simulator they were built for, or in the browser.

    # a robot output dir (from `python -m cad2urdf.route ...` or `python -m cad2urdf ...`)
    python tests/view_urdf.py build/arm4 --sim maniskill       # SAPIEN viewer window, PD-held at the home pose
    python tests/view_urdf.py build/arm4 --sim mujoco          # MuJoCo viewer (Control panel = joint targets)
    python tests/view_urdf.py build/arm4                       # browser: sliders, collision toggle, SRDF poses

    # a static scene dir (from `python -m cad2urdf.scene ...`)
    python tests/view_urdf.py build/field --sim maniskill
    python tests/view_urdf.py build/field --sim mujoco

    # a robot standing on a scene, in ManiSkill
    python tests/view_urdf.py build/arm4 --sim maniskill --scene build/field --at 0 0 0.02

    # no window: save a picture instead (works headless with a GPU)
    python tests/view_urdf.py build/field --sim maniskill --screenshot field.png

MuJoCo viewer keys: 2/3 toggle visual/collision geom groups, Space pauses, Ctrl+drag applies forces.
SAPIEN viewer: right-drag rotates, middle-drag pans, scroll zooms; the left panel shows collision shapes.

Not collected by pytest (the name doesn't start with test_). If ROS is sourced, run with `env -u PYTHONPATH`.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere


# ------------------------------------------------------------------ what's in the dir
def inspect_dir(path: Path) -> dict:
    path = path.resolve()
    if path.is_file():
        return {"kind": "urdf", "urdf": path, "dir": path.parent}
    agents = list((path / "maniskill").glob("*_agent.py"))
    scenes = list((path / "maniskill").glob("*_scene.py"))
    mjcf = [p for p in (path / "mjcf").glob("*.xml") if not p.name.startswith("_")]
    urdf = [p for p in path.glob("*.urdf") if not p.name.startswith("_")]
    kind = "scene" if scenes else "robot"
    return {"kind": kind, "dir": path, "agent": agents[0] if agents else None, "scene": scenes[0] if scenes else None,
            "mjcf": mjcf[0] if mjcf else None, "urdf": urdf[0] if urdf else None}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ MuJoCo
def view_mujoco(info: dict, screenshot: str | None) -> None:
    import mujoco

    if info["mjcf"] is None:
        raise SystemExit(f"no mjcf/*.xml in {info['dir']}")
    m = mujoco.MjModel.from_xml_path(str(info["mjcf"]))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    if screenshot:
        # frame the whole model from above at 45 degrees
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        r = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        # frame the model's own geoms (ignore the floor plane, which would dominate the extent)
        idx = [i for i in range(m.ngeom) if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE]
        pts = d.geom_xpos[idx]
        rad = m.geom_rbound[idx]
        lo, hi = (pts - rad[:, None]).min(0), (pts + rad[:, None]).max(0)
        cam.lookat[:] = (lo + hi) / 2
        cam.distance = float(np.linalg.norm(hi - lo)) * 1.2
        cam.elevation, cam.azimuth = -40, 135
        r.update_scene(d, cam)
        _save(r.render(), screenshot)
        return
    import mujoco.viewer

    print(f"MuJoCo viewer: {info['mjcf']}")
    mujoco.viewer.launch(m, d)


# ------------------------------------------------------------------ ManiSkill
def view_maniskill(info: dict, scene_info: dict | None, at: list[float], screenshot: str | None,
                   size: float = 1.0) -> None:
    import gymnasium as gym
    import sapien
    import torch
    from mani_skill.envs.tasks.empty_env import EmptyEnv
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env
    from mani_skill.sensors.camera import CameraConfig

    robot_uid = None
    if info["kind"] == "robot":
        if info["agent"] is None:
            raise SystemExit(f"no maniskill/*_agent.py in {info['dir']}")
        agent_mod = _load_module(info["agent"], "cad2sim_agent")
        agent_cls = next(v for v in vars(agent_mod).values()
                         if isinstance(v, type) and getattr(v, "__module__", "") == "cad2sim_agent" and hasattr(v, "uid"))
        robot_uid = agent_cls.uid
    static = scene_info if scene_info else (info if info["kind"] == "scene" else None)
    build_scene = None
    extent = 2.0
    if static:
        smod = _load_module(static["scene"], "cad2sim_scene")
        build_scene = next(v for k, v in vars(smod).items() if k.startswith("build_") and callable(v))
        extent = 8.0

    @register_env("Cad2SimView-v1", max_episode_steps=10**9, override=True)
    class ViewEnv(EmptyEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, robot_uids=robot_uid, **kw)

        def _load_agent(self, options):
            if robot_uid is not None:
                super(EmptyEnv, self)._load_agent(options, sapien.Pose(p=at))

        # ManiSkill's state save/restore assumes a robot; a scene-only view has none
        def get_state_dict(self):
            return self.scene.get_sim_state() if robot_uid is None else super().get_state_dict()

        def set_state_dict(self, state, env_idx=None):
            if robot_uid is None:
                self.scene.set_sim_state(state, env_idx)
            else:
                super().set_state_dict(state, env_idx)

        def _load_scene(self, options):
            super()._load_scene(options)
            if build_scene is not None:
                build_scene(self.scene)

        @property
        def _default_human_render_camera_configs(self):
            k = max(size, 0.2)
            eye = [extent * 0.6, -extent * 0.6, extent * 0.5] if static else [at[0] + 1.2 * k, at[1] - 1.2 * k, at[2] + 1.0 * k]
            look = [0, 0, 0] if static else [at[0], at[1], at[2] + 0.4 * k]
            return CameraConfig("render_camera", sapien_utils.look_at(eye, look), 1280, 720, 1.0, 0.01, 100)

    env = gym.make("Cad2SimView-v1", render_mode="rgb_array" if screenshot else "human", obs_mode="none",
                   control_mode="pd_joint_pos" if robot_uid else None, sim_backend="physx_cpu")
    env.reset(seed=0)
    hold = None
    if robot_uid is not None:
        agent = env.unwrapped.agent
        hold = torch.zeros(env.action_space.shape)
        if agent.keyframes:  # start from the first SRDF pose / keyframe and hold it
            kf = next(iter(agent.keyframes.values()))
            agent.robot.set_qpos(torch.tensor(kf.qpos, dtype=torch.float32))
            i = 0
            for c in agent.controller.controllers.values():
                mimic = getattr(c.config, "mimic", {}) or {}
                for j in c.config.joint_names:
                    if j in mimic:
                        continue
                    idx = [jj.name for jj in agent.robot.active_joints].index(j)
                    hold[..., i] = float(kf.qpos[idx])
                    i += 1
    if screenshot:
        for _ in range(5):
            env.step(hold) if hold is not None else env.unwrapped.scene.step()
        img = env.render()
        _save(img[0].cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)[0], screenshot)
        env.close()
        return
    print("SAPIEN viewer open; close the window (or Ctrl+C) to quit")
    viewer = env.render()
    while True:
        if hold is not None:
            env.step(hold)
        else:
            env.unwrapped.scene.step()
        env.render()
        if getattr(viewer, "closed", False):
            break


def _save(img: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8)).save(path)
    print(f"saved {path}")


# ------------------------------------------------------------------ browser (viser)
def view_browser(info: dict, port: int) -> None:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    urdf_path = info["urdf"]
    if urdf_path is None:
        raise SystemExit(f"no .urdf in {info['dir']}")
    robot = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True,
                               build_collision_scene_graph=True, load_collision_meshes=True)
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=4, height=4, cell_size=0.1)
    vu = ViserUrdf(server, robot, root_node_name="/robot", load_meshes=True, load_collision_meshes=True,
                   collision_mesh_color_override=(0.2, 0.6, 1.0, 0.45))
    vu.show_collision = False
    names = vu.get_actuated_joint_names()
    limits = vu.get_actuated_joint_limits()
    server.gui.add_markdown(f"**{urdf_path.name}**  \n{len(robot.link_map)} links, {len(robot.joint_map)} joints, "
                            f"{len(names)} actuated")
    with server.gui.add_folder("Display"):
        show_vis = server.gui.add_checkbox("visual meshes", True)
        show_col = server.gui.add_checkbox("collision geometry", False)
    sliders = []
    with server.gui.add_folder("Joints"):
        for name, (lo, hi) in zip(names, limits.values() if isinstance(limits, dict) else limits):
            lo = -np.pi if lo is None else lo
            hi = np.pi if hi is None else hi
            if hi <= lo:
                lo, hi = lo - 1.0, lo + 1.0
            sliders.append(server.gui.add_slider(name, min=float(lo), max=float(hi), step=float((hi - lo) / 500),
                                                 initial_value=float(np.clip(0.0, lo, hi))))

    def apply():
        vu.update_cfg(np.array([s.value for s in sliders]))

    for s in sliders:
        s.on_update(lambda _: apply())
    show_vis.on_update(lambda _: setattr(vu, "show_visual", show_vis.value))
    show_col.on_update(lambda _: setattr(vu, "show_collision", show_col.value))
    poses = {}
    srdf = urdf_path.with_suffix(".srdf")
    if srdf.exists():
        for gs in ET.parse(srdf).getroot().findall("group_state"):
            poses[gs.get("name")] = {j.get("name"): float(j.get("value")) for j in gs.findall("joint")}
    if poses:
        dd = server.gui.add_dropdown("SRDF pose", options=["(none)", *poses])

        @dd.on_update
        def _(_):
            for s, n in zip(sliders, names):
                if n in poses.get(dd.value, {}):
                    s.value = float(np.clip(poses[dd.value][n], s.min, s.max))

    apply()
    print(f"viewing {urdf_path} -> http://localhost:{port}")
    while True:
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="output dir (robot or scene) or a .urdf file")
    ap.add_argument("--sim", choices=["browser", "mujoco", "maniskill"], default="browser")
    ap.add_argument("--scene", type=Path, help="(maniskill) static scene output dir to place the robot on")
    ap.add_argument("--at", type=float, nargs=3, default=None,
                    help="(maniskill) robot base position; default lifts the robot just above the floor")
    ap.add_argument("--screenshot", help="save an image instead of opening a window (maniskill/mujoco)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    info = inspect_dir(args.path)
    if args.sim == "mujoco":
        view_mujoco(info, args.screenshot)
    elif args.sim == "maniskill":
        at = args.at
        if at is None:
            at = [0.0, 0.0, 0.0]
            if info["kind"] == "robot" and info["urdf"] is not None:
                from cad2urdf.validate import ground_clearance

                at[2] = ground_clearance(info["urdf"])
        size = 1.0
        if info["kind"] == "robot" and info["urdf"] is not None:
            import yourdfpy

            b = yourdfpy.URDF.load(str(info["urdf"]), load_meshes=True).scene.bounds
            size = float(np.linalg.norm(b[1] - b[0]))
        view_maniskill(info, inspect_dir(args.scene) if args.scene else None, at, args.screenshot, size)
    else:
        view_browser(info, args.port)


if __name__ == "__main__":
    from cad2urdf.safety import sandbox

    sandbox()  # re-launch this script inside the shared memory cap
    main()
