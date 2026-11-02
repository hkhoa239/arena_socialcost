"""Short Arena clips: PedVO, the 50k policies, and the 1-second density forecast.

One fresh simulator should already be running. Each clip is a single head-on
or five-pedestrian episode. Frames are the viewport camera plus a side panel
drawn from the live pedestrian poses.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from PIL import Image, ImageDraw
from viewport_control_msgs.srv import ViewportCapture

from arena_mujoco.viewport.camera import pose_of
from matched_eval import _Trace, _config, _reason
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.density import rasterize_density
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.interaction import (
    InteractionResidual,
    forecast_density,
    residual_checkpoint,
)

from arena_training.arena_rosnav_rl.trainer import get_trainer
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import spawn_envs, wait_for_simulation

OUT = Path("/opt/arena_ws/data/mf_eval/videos")
CLIPS = (
    ("01_pedvo_stopped_robot", "reference", "headon", 200, "PedVO, robot held still"),
    ("02_policy_agent5", "AGENT_5", "headon", 140, "AGENT_5 policy, head-on"),
    ("03_policy_online", "AGENT_MF_ONLINE", "headon", 240, "Online forecast policy, head-on"),
    ("04_policy_residual", "AGENT_MF_RESIDUAL", "headon", 280, "Residual forecast policy, head-on"),
    ("05_pedvo_five_pedestrians", "reference", "peds_5", 160, "PedVO, five pedestrians, robot held still"),
)
EVERY = 3
LOOK = (20.0, 8.0, 0.3)


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heat(grid: np.ndarray) -> Image.Image:
    values = np.asarray(grid, dtype=np.float32)
    peak = float(values.max()) if values.size else 0.0
    scale = values / peak if peak > 1e-8 else values
    image = np.zeros((*scale.shape, 3), dtype=np.uint8)
    image[..., 0] = 28 + (210 * scale)
    image[..., 1] = 32 + (70 * scale)
    image[..., 2] = 36
    shown = np.flipud(np.swapaxes(image, 0, 1))
    return Image.fromarray(shown, "RGB").resize((150, 150), Image.Resampling.NEAREST)


def _densities(module, peds: dict[str, tuple[float, float, float, float]], robot, command):
    if not peds or robot is None:
        blank = np.zeros((16, 16), dtype=np.float32)
        return blank, blank, blank
    rx, ry, yaw = robot
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    positions, velocities = [], []
    for x, y, vx, vy in peds.values():
        dx, dy = x - rx, y - ry
        positions.append((cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy))
        velocities.append((cos_yaw * vx + sin_yaw * vy, -sin_yaw * vx + cos_yaw * vy))
    pos = torch.tensor(positions, dtype=torch.float32).view(1, -1, 2)
    vel = torch.tensor(velocities, dtype=torch.float32).view(1, -1, 2)
    valid = torch.ones(1, pos.shape[1], dtype=torch.bool)
    cmd = torch.tensor([command], dtype=torch.float32)
    heading = torch.tensor([[cos_yaw, -sin_yaw]], dtype=torch.float32)
    current = rasterize_density(np.asarray(positions, dtype=np.float32))[0, 0]
    with torch.no_grad():
        online = forecast_density(module, pos, vel, valid, cmd, False, heading)[0]
        residual = forecast_density(module, pos, vel, valid, cmd, True, heading)[0]
    return current, online.numpy(), residual.numpy()


def _topdown(peds, robot, trails, command) -> Image.Image:
    canvas = Image.new("RGB", (400, 520), (24, 26, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "map, 20 m window", fill=(210, 210, 210))
    origin = (200, 250)
    scale = 16.0
    cx, cy = (robot[0], robot[1]) if robot is not None else LOOK[:2]

    def point(x, y):
        return origin[0] + (x - cx) * scale, origin[1] - (y - cy) * scale

    draw.line((20, 250, 380, 250), fill=(50, 52, 56))
    draw.line((200, 40, 200, 480), fill=(50, 52, 56))
    for name, trail in trails.items():
        pts = [point(x, y) for x, y in trail[-40:]]
        if len(pts) > 1:
            draw.line(pts, fill=(120, 170, 210), width=2)
    for x, y, vx, vy in peds.values():
        px, py = point(x, y)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(90, 160, 220))
        draw.line((px, py, px + vx * 18, py - vy * 18), fill=(180, 210, 230))
    if robot is not None:
        rx, ry, yaw = robot
        nose = (rx + math.cos(yaw) * 0.6, ry + math.sin(yaw) * 0.6)
        left = (rx + math.cos(yaw + 2.4) * 0.35, ry + math.sin(yaw + 2.4) * 0.35)
        right = (rx + math.cos(yaw - 2.4) * 0.35, ry + math.sin(yaw - 2.4) * 0.35)
        draw.polygon([point(*nose), point(*left), point(*right)], fill=(230, 150, 60))
    draw.text((12, 490), f"command v {command[0]:+.2f}  w {command[1]:+.2f}", fill=(210, 210, 210))
    return canvas


def _compose(photo: Image.Image | None, peds, robot, trails, command, module, caption, step, reason) -> Image.Image:
    frame = Image.new("RGB", (1280, 720), (16, 17, 18))
    draw = ImageDraw.Draw(frame)
    if photo is not None:
        frame.paste(photo.resize((820, 462), Image.Resampling.BILINEAR), (16, 48))
    else:
        draw.rectangle((16, 48, 836, 510), fill=(32, 32, 34))
        draw.text((40, 250), "viewport frame unavailable", fill=(180, 180, 180))
    frame.paste(_topdown(peds, robot, trails, command), (850, 48))
    current, online, residual = _densities(module, peds, robot, command)
    labels = (("now", current), ("online +1 s", online), ("residual +1 s", residual))
    for index, (label, grid) in enumerate(labels):
        left = 16 + index * 270
        frame.paste(_heat(grid), (left, 530))
        draw.text((left + 160, 590), label, fill=(220, 220, 220))
    draw.text((16, 14), f"{caption}    step {step}    {reason}", fill=(235, 235, 235))
    return frame


class _Camera(_Trace):
    def __init__(self) -> None:
        super().__init__()
        self.latest: dict[str, tuple[float, float, float, float]] = {}
        self.robot = None
        self.capture = self.create_client(ViewportCapture, "/arena/viewport/capture")
        self._odom_ready = False
        self._pose_logged = False
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

    def _on_peds(self, msg) -> None:
        super()._on_peds(msg)
        for ped in msg.pedestrians:
            self.latest[ped.name] = (
                float(ped.pose.position.x),
                float(ped.pose.position.y),
                float(ped.twist.linear.x),
                float(ped.twist.linear.y),
            )

    def attach_odom(self, namespace: str) -> None:
        if self._odom_ready:
            return
        env_name = namespace.split("/")[1] if "/" in namespace else namespace
        names = [name for name, _ in self.get_topic_names_and_types()]
        odoms = [name for name in names if name.endswith("/jackal/odom") and f"/{env_name}/" in name]
        print(f"ODOM {odoms[:4]}", flush=True)
        if odoms:
            self.create_subscription(Odometry, odoms[0], self._odom, 10)
            self._odom_ready = True

    def _odom(self, msg: Odometry) -> None:
        return

    def refresh_robot(self) -> None:
        for child in ("env_0/jackal/base_link", "base_link"):
            try:
                transform = self._tf_buffer.lookup_transform("map", child, Time())
            except Exception:
                continue
            t = transform.transform.translation
            q = transform.transform.rotation
            self.robot = (float(t.x), float(t.y), _yaw((q.w, q.x, q.y, q.z)))
            if not self._pose_logged:
                self._pose_logged = True
                print(f"MAP_POSE {child} x={t.x:.2f} y={t.y:.2f}", flush=True)
            return

    def snap(self) -> Image.Image | None:
        if not self.capture.service_is_ready():
            return None
        self.refresh_robot()
        if self.latest:
            xs = [item[0] for item in self.latest.values()]
            ys = [item[1] for item in self.latest.values()]
            target = (sum(xs) / len(xs), sum(ys) / len(ys), 0.4)
        elif self.robot is not None:
            target = (self.robot[0], self.robot[1], 0.3)
        else:
            target = LOOK
        eye = pose_of(target, 12.0, 32.0, -52.0)
        request = ViewportCapture.Request()
        request.pose = Pose()
        request.pose.position.x, request.pose.position.y, request.pose.position.z = eye.position
        request.pose.orientation.w, request.pose.orientation.x, request.pose.orientation.y, request.pose.orientation.z = eye.orientation
        request.world_orientation = True
        request.fov = 0.9
        future = self.capture.call_async(request)
        deadline = time.time() + 8.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        if not future.done() or future.result() is None or not future.result().success:
            message = "" if not future.done() or future.result() is None else future.result().message
            print(f"CAPTURE_FAIL {message}", flush=True)
            return None
        image = future.result().image
        pixels = np.frombuffer(bytes(image.data), dtype=np.uint8)
        if pixels.size != image.height * image.width * 3:
            return None
        return Image.fromarray(pixels.reshape(image.height, image.width, 3), "RGB")


def _one(agent: str, stage: str, limit: int, caption: str, module, camera: _Camera, folder: Path) -> None:
    spec_steps = limit
    config, folder_name = _config(agent, stage)
    config.arena_cfg.general.max_num_moves_per_eps = spec_steps
    launch = [[*config.arena_cfg.general.env_launch_args[0], config.arena_cfg.robot.launch_arg, "task.episode.spawn.robot_clearance:=1.0"]]
    env_map = spawn_envs(1, launch)
    camera.listen(env_map[0])
    camera.attach_odom(env_map[0])
    ready = time.time() + 20.0
    while not camera.capture.service_is_ready() and time.time() < ready:
        time.sleep(0.2)
    trainer = get_trainer(config, namespace_fn=lambda index, mapping=env_map: mapping[index])
    try:
        model = trainer.agent.model._model
        if agent != "reference":
            checkpoint = Path(f"/opt/arena_ws/data/agents/{folder_name}/last_model.zip")
            model = type(model).load(str(checkpoint), env=model.get_env(), device="cuda")
        env = model.get_env()
        zero = np.zeros((env.num_envs,) + env.action_space.shape)
        env.step(zero)
        previous = camera.episode_id
        obs = env.reset()
        camera.wait_episode(previous)
        time.sleep(0.3)
        camera.reset_episode()
        camera.latest = {}
        trails: dict[str, list[tuple[float, float]]] = {}
        command = [0.0, 0.0]
        done = False
        steps = 0
        reason = "running"
        folder.mkdir(parents=True, exist_ok=True)
        frame_index = 0
        while not done and steps < spec_steps:
            if agent == "reference":
                action = zero
            else:
                action, _ = model.predict(obs, deterministic=True)
            command = [float(action.reshape(-1)[0]), float(action.reshape(-1)[-1])]
            obs, _, dones, infos = env.step(action)
            steps += 1
            done = bool(np.asarray(dones).reshape(-1)[0])
            if done:
                reason, _ = _reason(infos[0] if isinstance(infos, (list, tuple)) else infos)
            for name, (x, y, _, _) in camera.latest.items():
                trails.setdefault(name, []).append((x, y))
            if steps % EVERY == 1 or done:
                photo = camera.snap()
                image = _compose(photo, dict(camera.latest), camera.robot, trails, command, module, caption, steps, reason)
                image.save(folder / f"frame_{frame_index:04d}.png")
                frame_index += 1
                if frame_index == 1:
                    print(f"ROBOT {camera.robot} PEDS {len(camera.latest)} PHOTO {photo is not None}", flush=True)
        print(f"CLIP {folder.name} steps={steps} reason={reason} frames={frame_index}", flush=True)
    finally:
        trainer.close()


def main() -> None:
    if not rclpy.ok():
        rclpy.init()
    wait_for_simulation(timeout=180.0)
    module = InteractionResidual()
    module.load_state_dict(torch.load(residual_checkpoint(), map_location="cpu", weights_only=True))
    module.eval()
    camera = _Camera()
    stop = __import__("threading").Event()
    thread = __import__("threading").Thread(target=lambda: _spin(camera, stop), daemon=True)
    thread.start()
    try:
        selected = sys.argv[1]
        name, agent, stage, limit, caption = next(clip for clip in CLIPS if clip[0] == selected)
        print(f"BEGIN {name}", flush=True)
        _one(agent, stage, limit, caption, module, camera, OUT / name)
    finally:
        stop.set()


def _spin(node, stop) -> None:
    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)


if __name__ == "__main__":
    main()
