"""Seeded density-dynamics diagnostic. Does not train and does not change configs.

Seed 5 shuffles a fixed catalog of (linear.x, angular.z) commands. Each command
is held for 2 seconds. Samples are taken at 10 Hz, the training control rate,
so a 10-step target is the one-second-later density.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rclpy
import torch
from arena_people_msgs.msg import Pedestrians
from geometry_msgs.msg import Twist
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from rosnav_rl.model.stable_baselines3.mean_field_ppo import MeanFieldPPO
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.density import (
    rasterize_density,
)
from rosnav_rl.observations.utils.semantic import get_relative_pos_to_robot

SEED = 5
HORIZON = 10
HOLD_STEPS = 20
REPEATS = 3
SAMPLE_HZ = 10.0
MOVE_M = 0.15
EMPTY_SUM = 1e-4
OUT = Path("/opt/arena_ws/data/mf_dynamics_diag.json")
MF_CKPT = "/opt/arena_ws/data/agents/AGENT_MF_PILOT/last_model.zip"
FC_CKPT = "/opt/arena_ws/data/agents/AGENT_MF_FORECAST_PILOT/last_model.zip"

CATALOG = np.array(
    [
        [0.0, 0.0],
        [0.5, 0.0],
        [0.9, 0.0],
        [0.3, 0.7],
        [0.3, -0.7],
        [0.0, 1.0],
        [-0.4, 0.0],
        [0.6, -0.4],
    ],
    dtype=np.float64,
)


def command_schedule(seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(CATALOG))
    commands = np.repeat(CATALOG[order], HOLD_STEPS, axis=0)
    return np.tile(commands, (REPEATS, 1))


def yaw_of(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) == 0:
        return float("nan")
    return float(np.mean((prediction - target) ** 2))


class Recorder(Node):
    def __init__(self, schedule: np.ndarray) -> None:
        super().__init__("mf_dynamics_diag")
        self.schedule = schedule
        self.step = 0
        self.done = False
        self.command = np.zeros(2, dtype=np.float64)
        self.peds: Pedestrians | None = None
        self.samples: list[dict] = []
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(
            Twist, "/arena/env_0/task_generator_node/jackal/cmd_vel", 10
        )
        self.create_subscription(Pedestrians, "/arena/env_0/arena_peds", self._on_peds, 10)
        self.create_timer(1.0 / 50.0, self._publish)
        self.create_timer(1.0 / SAMPLE_HZ, self._sample)

    def _on_peds(self, msg: Pedestrians) -> None:
        self.peds = msg

    def _publish(self) -> None:
        msg = Twist()
        msg.linear.x = float(self.command[0])
        msg.angular.z = float(self.command[1])
        self.pub.publish(msg)

    def _sample(self) -> None:
        if self.step >= len(self.schedule):
            self.done = True
            return
        try:
            transform = self.buffer.lookup_transform(
                "map", "env_0/jackal/base_link", rclpy.time.Time()
            )
        except Exception:
            return
        if self.peds is None:
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        pose = {"x": translation.x, "y": translation.y, "yaw": yaw_of(rotation)}
        world = np.array(
            [[p.pose.position.x, p.pose.position.y, 1.0] for p in self.peds.pedestrians],
            dtype=np.float64,
        )
        if len(world) == 0:
            relative = np.zeros((0, 2), dtype=np.float64)
        else:
            relative = get_relative_pos_to_robot(pose, world)
        self.samples.append(
            {
                "command": self.command.copy(),
                "robot_xy": np.array([pose["x"], pose["y"]], dtype=np.float64),
                "yaw": pose["yaw"],
                "world": world[:, :2].copy() if len(world) else np.zeros((0, 2)),
                "relative": np.asarray(relative, dtype=np.float64),
                "stamp": self.peds.header.stamp.sec + self.peds.header.stamp.nanosec * 1e-9,
            }
        )
        self.command = self.schedule[self.step].copy()
        self.step += 1
        if self.step % 50 == 0:
            self.get_logger().info(f"samples {self.step}/{len(self.schedule)}")


def densities(samples: list[dict]) -> np.ndarray:
    frames = []
    for sample in samples:
        frames.append(rasterize_density(sample["relative"])[0, 0])
    return np.stack(frames)


def pair_arrays(samples: list[dict], rho: np.ndarray):
    count = len(samples) - HORIZON
    rho_t = rho[1:count]
    rho_prev = rho[0 : count - 1]
    target = rho[1 + HORIZON : count + HORIZON]
    # The command stored with sample t is the command that was already in effect.
    action = np.stack([samples[t]["command"] for t in range(1, count)])
    world_move = []
    for t in range(1, count):
        now = samples[t]["world"]
        future = samples[t + HORIZON]["world"]
        if len(now) == 0 or len(future) == 0:
            world_move.append(0.0)
        else:
            world_move.append(float(np.linalg.norm(future.mean(axis=0) - now.mean(axis=0))))
    return rho_t, rho_prev, target, action, np.asarray(world_move)


def predict(net, rho_t, rho_prev, action) -> np.ndarray:
    with torch.no_grad():
        pred = net(
            torch.as_tensor(rho_t[:, None], dtype=torch.float32),
            torch.as_tensor(rho_prev[:, None], dtype=torch.float32),
            torch.as_tensor(action, dtype=torch.float32),
        )
    return pred[:, 0].numpy()


def main() -> None:
    schedule = command_schedule(SEED)
    rclpy.init()
    node = Recorder(schedule)
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.2)
    samples = node.samples
    node.destroy_node()
    rclpy.shutdown()

    rho = densities(samples)
    rho_t, rho_prev, target, action, world_move = pair_arrays(samples, rho)
    rebuilt = np.stack(
        [
            rasterize_density(samples[t + HORIZON]["relative"])[0, 0]
            for t in range(1, len(samples) - HORIZON)
        ]
    )
    alignment_error = float(np.max(np.abs(rebuilt - target)))
    off_by_one = float(np.mean((rho[2 : len(rho_t) + 2] - target) ** 2))

    mf = MeanFieldPPO.load(MF_CKPT, device="cpu")
    forecast = MeanFieldPPO.load(FC_CKPT, device="cpu")
    mf_net = mf.policy.features_extractor.response_net
    fc_net = forecast.policy.features_extractor.response_net
    pred_mf = predict(mf_net, rho_t, rho_prev, action)
    pred_fc = predict(fc_net, rho_t, rho_prev, np.zeros_like(action))
    pred_persist = rho_t

    current_sum = rho_t.reshape(len(rho_t), -1).sum(axis=1)
    target_sum = target.reshape(len(target), -1).sum(axis=1)
    empty = (current_sum < EMPTY_SUM) & (target_sum < EMPTY_SUM)
    moving = (~empty) & (world_move >= MOVE_M)
    stationary = (~empty) & (world_move < MOVE_M)
    nonempty_frames = rho.reshape(len(rho), -1).sum(axis=1) >= EMPTY_SUM

    def block(mask: np.ndarray) -> dict:
        return {
            "pairs": int(mask.sum()),
            "persistence_mse": mse(pred_persist[mask], target[mask]),
            "agent_mf_mse": mse(pred_mf[mask], target[mask]),
            "agent_mf_forecast_mse": mse(pred_fc[mask], target[mask]),
        }

    probes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-0.5, 0.0], [0.5, -1.0]])
    base = predict(mf_net, rho_t, rho_prev, np.repeat(probes[:1], len(rho_t), axis=0))
    deltas = []
    for probe in probes[1:]:
        other = predict(mf_net, rho_t, rho_prev, np.repeat(probe[None], len(rho_t), axis=0))
        deltas.append(np.linalg.norm((other - base).reshape(len(rho_t), -1), axis=1))
    delta = np.stack(deltas).max(axis=0)
    sensitivity_mask = nonempty_frames[1 : 1 + len(delta)]

    stamps = np.array([sample["stamp"] for sample in samples])
    dt = np.diff(stamps)
    robot = np.stack([sample["robot_xy"] for sample in samples])
    commands = np.stack([sample["command"] for sample in samples])

    result = {
        "seed": SEED,
        "horizon_steps": HORIZON,
        "sample_hz": SAMPLE_HZ,
        "command_catalog": CATALOG.tolist(),
        "command_order": np.random.default_rng(SEED).permutation(len(CATALOG)).tolist(),
        "hold_seconds": HOLD_STEPS / SAMPLE_HZ,
        "repeats": REPEATS,
        "checkpoints": {"AGENT_MF": MF_CKPT, "AGENT_MF_FORECAST": FC_CKPT},
        "samples": len(samples),
        "median_sample_dt_s": float(np.median(dt)) if len(dt) else None,
        "robot_path_length_m": float(np.linalg.norm(np.diff(robot, axis=0), axis=1).sum()),
        "unique_commands": len(np.unique(np.round(commands, 5), axis=0)),
        "nonempty_frame_fraction": float(nonempty_frames.mean()),
        "pair_count": int(len(rho_t)),
        "nonempty_pair_fraction": float((~empty).mean()),
        "moving_pair_fraction": float(moving.mean()),
        "target_matches_density_at_t_plus_10": alignment_error < 1e-5,
        "max_abs_target_rebuild_error": alignment_error,
        "mse_if_target_were_the_next_frame": off_by_one,
        "mean_world_displacement_m": float(world_move.mean()),
        "splits": {
            "all": block(np.ones(len(rho_t), dtype=bool)),
            "empty": block(empty),
            "stationary_crowd": block(stationary),
            "moving_crowd": block(moving),
        },
        "velocity_sensitivity": {
            "network": "AGENT_MF",
            "probes": probes.tolist(),
            "nonempty_pairs": int(sensitivity_mask.sum()),
            "mean_max_l2_vs_zero_velocity": float(delta[sensitivity_mask].mean()) if sensitivity_mask.any() else None,
            "fraction_changed_above_1e-3": float((delta[sensitivity_mask] > 1e-3).mean()) if sensitivity_mask.any() else None,
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
