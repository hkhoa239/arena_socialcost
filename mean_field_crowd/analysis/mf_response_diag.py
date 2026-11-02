"""Paired pedestrian-response diagnostic and offline CrowdResponseNet fit.

Seed 5 resets every episode. The arenian starts at the same scenario pose.
Two repeats of each robot command separate simulation noise from a reaction.
CrowdResponseNet keeps its 10-step horizon and the existing architecture.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import torch
from arena_people_msgs.msg import Pedestrians
from geometry_msgs.msg import Twist
from rclpy.node import Node
from task_generator_msgs.srv import ResetEpisode
from tf2_ros import Buffer, TransformListener

from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.density import (
    rasterize_density,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.response_model import (
    CrowdResponseNet,
)
from rosnav_rl.observations.utils.semantic import get_relative_pos_to_robot

SEED = 5
HORIZON = 10
SAMPLE_HZ = 10.0
EPISODE_STEPS = 80
MOVE_M = 0.15
EMPTY_SUM = 1e-4
RESET_SERVICE = "/arena/env_0/task_generator_node/lifecycle/reset_episode"
OUT = Path("/opt/arena_ws/data/mf_response_diag.json")
NPZ = Path("/opt/arena_ws/data/mf_response_diag.npz")

# name, repeat, command schedule as a function of step
CONDITIONS = (
    ("block", 0),
    ("block", 1),
    ("leave", 0),
    ("leave", 1),
    ("approach", 0),
    ("approach", 1),
)


def command_at(name: str, step: int) -> np.ndarray:
    if name == "block":
        return np.zeros(2)
    if name == "approach":
        return np.array([0.8, 0.0])
    # Leave the pedestrian's line: turn for 1.6 s, then drive sideways.
    if step < 16:
        return np.array([0.0, 1.0])
    return np.array([0.6, 0.0])


def yaw_of(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) == 0:
        return float("nan")
    return float(np.mean((prediction - target) ** 2))


class Collector(Node):
    def __init__(self) -> None:
        super().__init__("mf_response_diag")
        self.peds: Pedestrians | None = None
        self.command = np.zeros(2)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(Twist, "/arena/env_0/task_generator_node/jackal/cmd_vel", 10)
        self.create_subscription(Pedestrians, "/arena/env_0/arena_peds", self._on_peds, 10)
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE)
        self.create_timer(1.0 / 50.0, self._publish)

    def _on_peds(self, msg: Pedestrians) -> None:
        self.peds = msg

    def _publish(self) -> None:
        msg = Twist()
        msg.linear.x = float(self.command[0])
        msg.angular.z = float(self.command[1])
        self.pub.publish(msg)

    def ped_xy(self) -> np.ndarray | None:
        if self.peds is None or not self.peds.pedestrians:
            return None
        person = self.peds.pedestrians[0]
        return np.array([person.pose.position.x, person.pose.position.y], dtype=np.float64)

    def robot_pose(self) -> dict | None:
        try:
            transform = self.buffer.lookup_transform("map", "env_0/jackal/base_link", rclpy.time.Time())
        except Exception:
            return None
        translation = transform.transform.translation
        return {
            "x": translation.x,
            "y": translation.y,
            "yaw": yaw_of(transform.transform.rotation),
        }

    def wait(self, seconds: float) -> None:
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def reset_and_settle(self) -> np.ndarray:
        if not self.reset_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"reset service unavailable: {RESET_SERVICE}")
        last_error = "pedestrian did not respawn"
        for _attempt in range(3):
            before = self.ped_xy()
            future = self.reset_client.call_async(ResetEpisode.Request(world="", seed=SEED))
            start = self.get_clock().now().nanoseconds
            while rclpy.ok() and not future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.get_clock().now().nanoseconds - start > 15e9:
                    raise RuntimeError("reset service timed out")
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError(f"reset failed: {None if response is None else response.error_msg}")
            history: list[np.ndarray] = []
            start = self.get_clock().now().nanoseconds
            settled = None
            while rclpy.ok() and self.get_clock().now().nanoseconds - start < 12e9:
                rclpy.spin_once(self, timeout_sec=0.02)
                current = self.ped_xy()
                if current is None:
                    continue
                history.append(current.copy())
                xs = np.array([point[0] for point in history])
                peak_i = int(xs.argmax())
                peak = history[peak_i]
                rose = before is None or float(peak[0]) > float(before[0]) + 0.8
                already_there = before is not None and np.linalg.norm(peak - before) < 0.25
                if not rose and not already_there:
                    continue
                if len(history) < peak_i + 3:
                    continue
                after = xs[peak_i:]
                left_spawn = float(after[-1]) < float(peak[0]) - 0.05
                holding = float(np.max(np.abs(after - peak[0]))) < 0.08 and len(after) >= 5
                if not left_spawn and not holding:
                    continue
                if abs(float(current[0]) - float(peak[0])) > 0.2:
                    last_error = f"recording window missed spawn peak {peak.tolist()} at {current.tolist()}"
                    settled = None
                    break
                settled = current.copy()
                break
            if settled is not None:
                return settled
        raise RuntimeError(last_error)

    def record(self, name: str, repeat: int) -> dict:
        spawn = self.reset_and_settle()
        samples = []
        next_tick = self.get_clock().now().nanoseconds
        period = int(1e9 / SAMPLE_HZ)
        while len(samples) < EPISODE_STEPS and rclpy.ok():
            self.command = command_at(name, len(samples))
            while self.get_clock().now().nanoseconds < next_tick:
                rclpy.spin_once(self, timeout_sec=0.01)
            next_tick += period
            pose = self.robot_pose()
            ped = self.ped_xy()
            rclpy.spin_once(self, timeout_sec=0.0)
            if pose is None or ped is None:
                continue
            world = np.array([[ped[0], ped[1], 1.0]])
            relative = np.asarray(get_relative_pos_to_robot(pose, world), dtype=np.float64).reshape(1, 2)
            samples.append(
                {
                    "command": self.command.copy(),
                    "robot_xy": np.array([pose["x"], pose["y"]]),
                    "yaw": pose["yaw"],
                    "world": ped.copy(),
                    "relative": relative,
                }
            )
        self.command = np.zeros(2)
        self.get_logger().info(f"{name}_{repeat}: {len(samples)} samples, spawn {spawn.tolist()}")
        return {"name": name, "repeat": repeat, "spawn": spawn, "samples": samples}


def episode_arrays(episode: dict):
    samples = episode["samples"]
    rho = np.stack([rasterize_density(sample["relative"])[0, 0] for sample in samples])
    world = np.stack([sample["world"] for sample in samples])
    relative = np.stack([sample["relative"].reshape(2) for sample in samples])
    command = np.stack([sample["command"] for sample in samples])
    robot = np.stack([sample["robot_xy"] for sample in samples])
    # An episode completion teleports the pedestrian and the robot. Pairs must
    # not cross that boundary, so the series ends on the last pre-jump sample.
    if len(world) > 1:
        ped_step = np.linalg.norm(np.diff(world, axis=0), axis=1)
        robot_step = np.linalg.norm(np.diff(robot, axis=0), axis=1)
        jumps = np.flatnonzero((ped_step > 0.5) | (robot_step > 0.5))
        if len(jumps):
            cut = int(jumps[0]) + 1
            rho, world, relative, command, robot = (
                rho[:cut],
                world[:cut],
                relative[:cut],
                command[:cut],
                robot[:cut],
            )
    return rho, world, relative, command, robot


def pairs_from_episode(episode: dict) -> dict:
    rho, world, relative, command, robot = episode_arrays(episode)
    last = len(rho) - HORIZON
    index = np.arange(1, last)
    extrapolated = relative[index] + HORIZON * (relative[index] - relative[index - 1])
    return {
        "name": np.array([episode["name"]] * len(index)),
        "repeat": np.full(len(index), episode["repeat"]),
        "rho_t": rho[index],
        "rho_prev": rho[index - 1],
        "target": rho[index + HORIZON],
        "action": command[index],
        "distance": np.linalg.norm(relative[index], axis=1),
        "world_move": np.linalg.norm(world[index + HORIZON] - world[index], axis=1),
        "extrapolated_density": np.stack([rasterize_density(pos.reshape(1, 2))[0, 0] for pos in extrapolated]),
        "robot_xy": robot[index],
        "ped_xy": world[index],
    }


def stack_pairs(parts: list[dict]) -> dict:
    keys = [key for key in parts[0] if key != "name"]
    out = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    out["name"] = np.concatenate([part["name"] for part in parts])
    return out


def path_y_delta(world_a: np.ndarray, world_b: np.ndarray) -> np.ndarray:
    """Absolute lateral difference after matching along-track x, so a start-time offset is not a swerve."""

    def curve(world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(world[:, 0])
        x = world[order, 0]
        y = world[order, 1]
        unique, index = np.unique(x, return_index=True)
        return unique, y[index]

    x_a, y_a = curve(world_a)
    x_b, y_b = curve(world_b)
    lo = max(float(x_a[0]), float(x_b[0]))
    hi = min(float(x_a[-1]), float(x_b[-1]))
    if hi - lo < 0.5:
        return np.zeros(0)
    grid = np.linspace(lo, hi, 40)
    return np.abs(np.interp(grid, x_a, y_a) - np.interp(grid, x_b, y_b))


def pointwise(world_a: np.ndarray, world_b: np.ndarray) -> np.ndarray:
    n = min(len(world_a), len(world_b))
    return np.linalg.norm(world_a[:n] - world_b[:n], axis=1)


def train_net(conditioned: bool, rho_t, rho_prev, action, target, steps: int) -> CrowdResponseNet:
    torch.manual_seed(SEED)
    net = CrowdResponseNet(conditioned=conditioned)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rho_t_t = torch.as_tensor(rho_t[:, None], dtype=torch.float32)
    rho_prev_t = torch.as_tensor(rho_prev[:, None], dtype=torch.float32)
    action_t = torch.as_tensor(action, dtype=torch.float32)
    target_t = torch.as_tensor(target[:, None], dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    net.train()
    for _ in range(steps):
        choice = torch.randint(0, len(rho_t), (64,), generator=generator)
        prediction = net(rho_t_t[choice], rho_prev_t[choice], action_t[choice])
        loss = torch.mean((prediction - target_t[choice]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    return net


def predict(net: CrowdResponseNet, rho_t, rho_prev, action) -> np.ndarray:
    with torch.no_grad():
        prediction = net(
            torch.as_tensor(rho_t[:, None], dtype=torch.float32),
            torch.as_tensor(rho_prev[:, None], dtype=torch.float32),
            torch.as_tensor(action, dtype=torch.float32),
        )
    return prediction[:, 0].numpy()


def episodes_from_raw(path: Path) -> list[dict]:
    blob = np.load(path)
    episodes = []
    for name, repeat in CONDITIONS:
        key = f"{name}_{repeat}"
        world = blob[f"{key}_world"]
        relative = blob[f"{key}_relative"]
        command = blob[f"{key}_command"]
        robot = blob[f"{key}_robot"]
        samples = [
            {
                "command": command[i],
                "robot_xy": robot[i],
                "yaw": 0.0,
                "world": world[i],
                "relative": relative[i].reshape(1, 2),
            }
            for i in range(len(world))
        ]
        episodes.append({"name": name, "repeat": repeat, "spawn": world[0].copy(), "samples": samples})
    return episodes


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--from-raw":
        episodes = episodes_from_raw(NPZ.with_name("mf_response_raw.npz"))
    else:
        rclpy.init()
        node = Collector()
        episodes = []
        try:
            node.wait(1.0)
            for name, repeat in CONDITIONS:
                episodes.append(node.record(name, repeat))
        finally:
            node.command = np.zeros(2)
            node.wait(0.2)
            node.destroy_node()
            rclpy.shutdown()

    raw = {}
    worlds = {}
    for episode in episodes:
        rho, world, relative, command, robot = episode_arrays(episode)
        key = f"{episode['name']}_{episode['repeat']}"
        raw[f"{key}_rho"] = rho
        raw[f"{key}_world"] = world
        raw[f"{key}_relative"] = relative
        raw[f"{key}_command"] = command
        raw[f"{key}_robot"] = robot
        worlds[(episode["name"], episode["repeat"])] = world
    if len(sys.argv) <= 1 or sys.argv[1] != "--from-raw":
        np.savez_compressed(NPZ.with_name("mf_response_raw.npz"), **raw)

    def stats(a, b) -> dict:
        delta = pointwise(worlds[a], worlds[b])
        n = min(len(worlds[a]), len(worlds[b]))
        cross = np.abs(worlds[a][:n, 1] - worlds[b][:n, 1])
        path = path_y_delta(worlds[a], worlds[b])
        return {
            "mean_m": float(delta.mean()),
            "max_m": float(delta.max()),
            "mean_cross_track_m": float(cross.mean()),
            "max_cross_track_m": float(cross.max()),
            "mean_path_y_m": float(path.mean()) if len(path) else float("nan"),
            "max_path_y_m": float(path.max()) if len(path) else float("nan"),
            "samples": int(len(delta)),
        }

    noise_block = stats(("block", 0), ("block", 1))
    noise_leave = stats(("leave", 0), ("leave", 1))
    noise_approach = stats(("approach", 0), ("approach", 1))
    # Approach repeats disagree with each other, so they are not the randomness
    # floor. The floor is the repeat scatter of the two stable commands.
    stable_noise = max(noise_block["mean_m"], noise_leave["mean_m"])
    effect_leave = stats(("block", 0), ("leave", 0))
    effect_approach = stats(("block", 0), ("approach", 0))
    response_threshold = max(0.15, 3.0 * stable_noise)

    parts = [pairs_from_episode(episode) for episode in episodes]
    data = stack_pairs(parts)
    np.savez_compressed(NPZ, **{key: value for key, value in data.items()})

    # A leave-the-path sample is robot-induced when it has left the stationary
    # robot's trajectory by more than the stable repeat noise. Head-on approach
    # repeats are labeled separately: their repeat scatter exceeds that noise,
    # so the deviation cannot be credited to the command.
    block_world = {0: worlds[("block", 0)], 1: worlds[("block", 1)]}
    responsive = np.zeros(len(data["name"]), dtype=bool)
    approach = data["name"] == "approach"
    cursor = 0
    for episode in episodes:
        count = len(pairs_from_episode(episode)["distance"])
        if episode["name"] == "leave":
            reference = block_world[episode["repeat"]]
            _, world, _, _, _ = episode_arrays(episode)
            deviation = pointwise(world, reference)
            pair_deviation = deviation[1 : 1 + count]
            responsive[cursor : cursor + count] = pair_deviation > response_threshold
        cursor += count
    distance = data["distance"]
    bins = {
        "0_to_1.5m": distance < 1.5,
        "1.5_to_3m": (distance >= 1.5) & (distance < 3.0),
        "3_to_10m": (distance >= 3.0) & (distance < 10.0),
        "beyond_10m": distance >= 10.0,
    }

    train = data["repeat"] == 0
    test = data["repeat"] == 1
    steps = 400
    conditioned = train_net(True, data["rho_t"][train], data["rho_prev"][train], data["action"][train], data["target"][train], steps)
    unconditioned = train_net(False, data["rho_t"][train], data["rho_prev"][train], data["action"][train], data["target"][train], steps)
    pred_c = predict(conditioned, data["rho_t"][test], data["rho_prev"][test], data["action"][test])
    pred_u = predict(unconditioned, data["rho_t"][test], data["rho_prev"][test], data["action"][test])
    pred_c_zero = predict(conditioned, data["rho_t"][test], data["rho_prev"][test], np.zeros_like(data["action"][test]))
    pred_persist = data["rho_t"][test]
    pred_extra = data["extrapolated_density"][test]
    target = data["target"][test]
    test_responsive = responsive[test]
    test_bins = {name: mask[test] for name, mask in bins.items()}

    def block(mask: np.ndarray) -> dict:
        return {
            "pairs": int(mask.sum()),
            "persistence_mse": mse(pred_persist[mask], target[mask]),
            "motion_extrapolation_mse": mse(pred_extra[mask], target[mask]),
            "conditioned_mse": mse(pred_c[mask], target[mask]),
            "unconditioned_mse": mse(pred_u[mask], target[mask]),
            "conditioned_with_velocity_zeroed_mse": mse(pred_c_zero[mask], target[mask]),
        }

    alignment = []
    for episode in episodes:
        rho, _, relative, _, _ = episode_arrays(episode)
        t_index = np.arange(1, len(rho) - HORIZON)
        rebuilt = np.stack([rasterize_density(relative[t + HORIZON].reshape(1, 2))[0, 0] for t in t_index])
        alignment.append(float(np.max(np.abs(rebuilt - rho[t_index + HORIZON]))))

    result = {
        "seed": SEED,
        "scenario": "mf_headon",
        "driver": "arena_humansim default local_planner sfm, adult agent type, waypoint goal",
        "horizon_steps": HORIZON,
        "sample_hz": SAMPLE_HZ,
        "episode_seconds": EPISODE_STEPS / SAMPLE_HZ,
        "offline_gradient_steps_each": steps,
        "train_episodes": "repeat 0 of block, leave, approach",
        "test_episodes": "repeat 1 of block, leave, approach",
        "parameter_count": sum(p.numel() for p in conditioned.parameters()),
        "unconditioned_parameter_count": sum(p.numel() for p in unconditioned.parameters()),
        "spawn_xy": {f"{ep['name']}_{ep['repeat']}": ep["spawn"].tolist() for ep in episodes},
        "episode_motion": {
            f"{ep['name']}_{ep['repeat']}": {
                "spawn_xy": ep["spawn"].tolist(),
                "end_xy": episode_arrays(ep)[1][-1].tolist(),
                "max_abs_y_from_spawn_m": float(np.max(np.abs(episode_arrays(ep)[1][:, 1] - ep["spawn"][1]))),
                "min_robot_distance_m": float(np.min(np.linalg.norm(episode_arrays(ep)[2], axis=1))),
                "robot_path_m": float(np.linalg.norm(np.diff(episode_arrays(ep)[4], axis=0), axis=1).sum()),
            }
            for ep in episodes
        },
        "robot_path_m": {
            f"{ep['name']}_{ep['repeat']}": float(np.linalg.norm(np.diff(episode_arrays(ep)[4], axis=0), axis=1).sum())
            for ep in episodes
        },
        "paired_response_m": {
            "block_repeat_noise": noise_block,
            "leave_repeat_noise": noise_leave,
            "approach_repeat_noise": noise_approach,
            "leave_versus_block": effect_leave,
            "approach_versus_block": effect_approach,
            "stable_repeat_noise_m": stable_noise,
            "response_threshold_m": response_threshold,
            "approach_repeat_exceeds_threshold": bool(noise_approach["mean_m"] > response_threshold),
        },
        "target_alignment_max_abs_error": max(alignment),
        "train_mse": {
            "conditioned": mse(predict(conditioned, data["rho_t"][train], data["rho_prev"][train], data["action"][train]), data["target"][train]),
            "unconditioned": mse(predict(unconditioned, data["rho_t"][train], data["rho_prev"][train], data["action"][train]), data["target"][train]),
            "persistence": mse(data["rho_t"][train], data["target"][train]),
        },
        "train_pairs": int(train.sum()),
        "test_pairs": int(test.sum()),
        "test_errors": {
            "all": block(np.ones(test.sum(), dtype=bool)),
            "responsive_leave_path": block(test_responsive),
            "nonresponsive_same_command_or_block": block((~test_responsive) & ~approach[test]),
            "unstable_head_on": block(approach[test]),
            "reference_block_episode": block(data["name"][test] == "block"),
            **{f"distance_{name}": block(mask) for name, mask in test_bins.items()},
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
