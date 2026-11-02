"""Interaction residual on top of one-second motion extrapolation.

The arenian either keeps its observed velocity or stops. A logistic model
reads the robot–pedestrian approach and the candidate command, then scales
that velocity before the density is rasterized. Held-out seeds are scored
with the same trajectories for the corrected and uncorrected forecasts.
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
from rosnav_rl.observations.utils.semantic import get_relative_pos_to_robot

HORIZON = 10
DT = HORIZON / 10.0
SAMPLE_HZ = 10.0
EPISODE_STEPS = 80
MOVING_SPEED = 0.40
STOP_SPEED = 0.25
SEEDS = (5, 6, 7, 8, 9, 10)
TRAIN_SEEDS = {5, 6, 7}
TEST_SEEDS = {8, 9, 10}
CONDITIONS = ("block", "leave", "aside")
CANONICAL_SPAWN = np.array([21.3, 7.3])
REPAIR = (
    (5, "block", 0),
    (5, "block", 1),
    (6, "leave", 0),
    (6, "aside", 0),
    (9, "leave", 1),
    (10, "aside", 0),
)
RESET_SERVICE = "/arena/env_0/task_generator_node/lifecycle/reset_episode"
RAW = Path("/opt/arena_ws/data/mf_interaction_raw.npz")
OUT = Path("/opt/arena_ws/data/mf_interaction_diag.json")
FEATURE_NAMES = (
    "rel_x",
    "rel_y",
    "rel_vx",
    "rel_vy",
    "closing_speed",
    "time_to_closest_s",
    "min_separation_m",
    "candidate_v",
    "candidate_w",
)
GEOMETRY = FEATURE_NAMES[:7]


def command_at(name: str, step: int) -> np.ndarray:
    if name == "block":
        return np.zeros(2)
    if name == "leave":
        if step < 16:
            return np.array([0.0, 1.0])
        return np.array([0.6, 0.0])
    # Turn off the waypoint line, drive clear, then sit still.
    if step < 16:
        return np.array([0.0, 1.0])
    if step < 41:
        return np.array([0.6, 0.0])
    return np.zeros(2)


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
        super().__init__("mf_interaction_diag")
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

    def ped_state(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.peds is None or not self.peds.pedestrians:
            return None
        person = self.peds.pedestrians[0]
        position = np.array([person.pose.position.x, person.pose.position.y], dtype=np.float64)
        velocity = np.array([person.twist.linear.x, person.twist.linear.y], dtype=np.float64)
        return position, velocity

    def robot_pose(self) -> dict | None:
        try:
            transform = self.buffer.lookup_transform("map", "env_0/jackal/base_link", rclpy.time.Time())
        except Exception:
            return None
        translation = transform.transform.translation
        return {"x": translation.x, "y": translation.y, "yaw": yaw_of(transform.transform.rotation)}

    def wait(self, seconds: float) -> None:
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def reset_and_settle(self, seed: int) -> np.ndarray:
        if not self.reset_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"reset service unavailable: {RESET_SERVICE}")
        last_error = "pedestrian did not respawn"
        for _attempt in range(3):
            before = None if self.ped_state() is None else self.ped_state()[0]
            future = self.reset_client.call_async(ResetEpisode.Request(world="", seed=int(seed)))
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
                state = self.ped_state()
                if state is None:
                    continue
                current = state[0]
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
                if np.linalg.norm(peak - CANONICAL_SPAWN) > 0.15:
                    continue
                if abs(float(current[0]) - float(peak[0])) > 0.2:
                    last_error = f"missed spawn peak {peak.tolist()} at {current.tolist()}"
                    settled = None
                    break
                settled = current.copy()
                break
            if settled is not None:
                return settled
        raise RuntimeError(last_error)

    def record(self, seed: int, name: str, repeat: int) -> dict:
        spawn = self.reset_and_settle(seed)
        samples = []
        next_tick = self.get_clock().now().nanoseconds
        period = int(1e9 / SAMPLE_HZ)
        while len(samples) < EPISODE_STEPS and rclpy.ok():
            self.command = command_at(name, len(samples))
            while self.get_clock().now().nanoseconds < next_tick:
                rclpy.spin_once(self, timeout_sec=0.01)
            next_tick += period
            pose = self.robot_pose()
            state = self.ped_state()
            rclpy.spin_once(self, timeout_sec=0.0)
            if pose is None or state is None:
                continue
            ped, ped_v = state
            samples.append(
                {
                    "command": self.command.copy(),
                    "robot_xy": np.array([pose["x"], pose["y"]], dtype=np.float64),
                    "yaw": float(pose["yaw"]),
                    "ped_xy": ped.copy(),
                    "ped_v": ped_v.copy(),
                }
            )
        self.command = np.zeros(2)
        self.get_logger().info(f"seed {seed} {name}_{repeat}: {len(samples)} samples, spawn {spawn.tolist()}")
        return {"seed": seed, "name": name, "repeat": repeat, "spawn": spawn, "samples": samples}


def stack_episode(episode: dict) -> dict:
    samples = episode["samples"]
    ped = np.stack([sample["ped_xy"] for sample in samples])
    ped_v = np.stack([sample["ped_v"] for sample in samples])
    robot = np.stack([sample["robot_xy"] for sample in samples])
    yaw = np.array([sample["yaw"] for sample in samples], dtype=np.float64)
    command = np.stack([sample["command"] for sample in samples])
    if len(ped) > 1:
        ped_step = np.linalg.norm(np.diff(ped, axis=0), axis=1)
        robot_step = np.linalg.norm(np.diff(robot, axis=0), axis=1)
        jumps = np.flatnonzero((ped_step > 0.5) | (robot_step > 0.5))
        if len(jumps):
            cut = int(jumps[0]) + 1
            ped, ped_v, robot, yaw, command = ped[:cut], ped_v[:cut], robot[:cut], yaw[:cut], command[:cut]
    return {
        "seed": int(episode["seed"]),
        "name": episode["name"],
        "repeat": int(episode["repeat"]),
        "ped": ped,
        "ped_v": ped_v,
        "robot": robot,
        "yaw": yaw,
        "command": command,
    }


def observed_velocity(tracks: list[dict]) -> tuple[list[dict], str]:
    dots = []
    for track in tracks:
        if len(track["ped"]) < 2:
            continue
        step = np.diff(track["ped"], axis=0) * SAMPLE_HZ
        dots.append(np.sum(track["ped_v"][1:] * step, axis=1))
    agreement = float(np.mean(np.concatenate(dots))) if dots else 0.0
    source = "pedestrian twist" if agreement > 0.0 else "backward position difference"
    for track in tracks:
        finite = np.zeros_like(track["ped"])
        if len(track["ped"]) > 1:
            finite[1:] = np.diff(track["ped"], axis=0) * SAMPLE_HZ
            finite[0] = finite[1]
        track["ped_speed_twist"] = np.linalg.norm(track["ped_v"], axis=1)
        if source != "pedestrian twist":
            track["ped_v"] = finite
        track["robot_v"] = np.zeros_like(track["robot"])
        if len(track["robot"]) > 1:
            track["robot_v"][1:] = np.diff(track["robot"], axis=0) * SAMPLE_HZ
            track["robot_v"][0] = track["robot_v"][1]
    return tracks, source


def world_to_robot(yaw: float, vector: np.ndarray) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [
            cos_yaw * vector[0] + sin_yaw * vector[1],
            -sin_yaw * vector[0] + cos_yaw * vector[1],
        ],
        dtype=np.float64,
    )


def approach_features(rel: np.ndarray, rel_v: np.ndarray) -> tuple[float, float, float]:
    closing = float(-np.dot(rel, rel_v) / max(np.linalg.norm(rel), 1e-6))
    speed_sq = float(np.dot(rel_v, rel_v))
    if speed_sq < 1e-6:
        return closing, 0.0, float(np.linalg.norm(rel))
    tca = float(-np.dot(rel, rel_v) / speed_sq)
    if tca < 0.0:
        return closing, 0.0, float(np.linalg.norm(rel))
    separation = rel + rel_v * tca
    return closing, min(tca, 5.0), float(np.linalg.norm(separation))


def density_of(ped_xy: np.ndarray, robot_xy: np.ndarray, yaw: float) -> np.ndarray:
    pose = {"x": float(robot_xy[0]), "y": float(robot_xy[1]), "yaw": float(yaw)}
    homogeneous = np.array([[ped_xy[0], ped_xy[1], 1.0]], dtype=np.float64)
    relative = np.asarray(get_relative_pos_to_robot(pose, homogeneous), dtype=np.float64).reshape(1, 2)
    return rasterize_density(relative)[0, 0]


def pairs_from_track(track: dict) -> dict:
    ped = track["ped"]
    last = len(ped) - HORIZON
    index = np.arange(1, last)
    rows = []
    extrapolated = []
    corrected_slot = []
    target = []
    distance = []
    labels = []
    moving = []
    future_speed = []
    progress = []
    for t in index:
        rel = world_to_robot(track["yaw"][t], ped[t] - track["robot"][t])
        rel_v = world_to_robot(track["yaw"][t], track["ped_v"][t] - track["robot_v"][t])
        closing, tca, separation = approach_features(rel, rel_v)
        rows.append([*rel, *rel_v, closing, tca, separation, track["command"][t, 0], track["command"][t, 1]])
        coast = ped[t] + track["ped_v"][t] * DT
        extrapolated.append(coast)
        corrected_slot.append(coast)
        target.append(ped[t + HORIZON])
        distance.append(float(np.linalg.norm(rel)))
        speed = float(np.linalg.norm(track["ped_v"][t]))
        ahead = float(np.linalg.norm(track["ped_v"][t + HORIZON]))
        moved = float(np.linalg.norm(ped[t + HORIZON] - ped[t]))
        moving.append(speed >= MOVING_SPEED)
        future_speed.append(ahead)
        progress.append(moved)
        slowed = ahead < STOP_SPEED or moved < 0.5 * speed * DT
        labels.append(float(slowed and speed >= MOVING_SPEED))
    extrap_xy = np.stack(extrapolated)
    target_xy = np.stack(target)
    extrap_density = np.stack(
        [density_of(extrap_xy[k], track["robot"][t + HORIZON], track["yaw"][t + HORIZON]) for k, t in enumerate(index)]
    )
    target_density = np.stack(
        [density_of(target_xy[k], track["robot"][t + HORIZON], track["yaw"][t + HORIZON]) for k, t in enumerate(index)]
    )
    return {
        "seed": np.full(len(index), track["seed"]),
        "name": np.array([track["name"]] * len(index)),
        "repeat": np.full(len(index), track["repeat"]),
        "features": np.stack(rows),
        "extrap_xy": extrap_xy,
        "target_xy": target_xy,
        "ped_v": np.stack([track["ped_v"][t] for t in index]),
        "ped_xy": np.stack([ped[t] for t in index]),
        "robot_future": np.stack([track["robot"][t + HORIZON] for t in index]),
        "yaw_future": np.array([track["yaw"][t + HORIZON] for t in index]),
        "extrap_density": extrap_density,
        "target_density": target_density,
        "distance": np.array(distance),
        "label": np.array(labels),
        "moving": np.array(moving),
        "future_speed": np.array(future_speed),
        "progress": np.array(progress),
    }


def stack_pairs(parts: list[dict]) -> dict:
    out = {}
    for key in parts[0]:
        out[key] = np.concatenate([part[key] for part in parts])
    return out


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> tuple[torch.nn.Linear, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    torch.manual_seed(5)
    layer = torch.nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)
    inputs = torch.as_tensor((features - mean) / std, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.float32)
    positives = target.sum().clamp(min=1.0)
    pos_weight = (len(target) - positives).clamp(min=1.0) / positives
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    layer.train()
    for _ in range(500):
        optimizer.zero_grad()
        loss = loss_fn(layer(inputs).squeeze(1), target)
        loss.backward()
        optimizer.step()
    layer.eval()
    return layer, mean, std


def predict_stop(layer: torch.nn.Linear, features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    inputs = torch.as_tensor((features - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(layer(inputs).squeeze(1)).numpy()


def classification(labels: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict:
    chosen = mask & np.isfinite(probability)
    truth = labels[chosen] > 0.5
    guess = probability[chosen] >= 0.5
    if len(truth) == 0:
        return {"samples": 0, "accuracy": float("nan"), "stop_precision": float("nan"), "stop_recall": float("nan"), "majority_continue_accuracy": float("nan")}
    true_stop = truth.sum()
    pred_stop = guess.sum()
    hit = np.logical_and(guess, truth).sum()
    return {
        "samples": int(len(truth)),
        "stop_rate": float(truth.mean()),
        "accuracy": float((guess == truth).mean()),
        "stop_precision": float(hit / pred_stop) if pred_stop else float("nan"),
        "stop_recall": float(hit / true_stop) if true_stop else float("nan"),
        "majority_continue_accuracy": float((~truth).mean()),
    }


def corrected_density(data: dict, probability: np.ndarray) -> np.ndarray:
    scale = (1.0 - probability)[:, None]
    corrected_xy = data["ped_xy"] + data["ped_v"] * DT * scale
    return np.stack(
        [
            density_of(corrected_xy[i], data["robot_future"][i], data["yaw_future"][i])
            for i in range(len(probability))
        ]
    )


def position_error(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(prediction[mask] - target[mask], axis=1)))


def save_raw(tracks: list[dict]) -> None:
    blob = {}
    for track in tracks:
        key = f"s{track['seed']}_{track['name']}_{track['repeat']}"
        for field in ("ped", "ped_v", "robot", "yaw", "command"):
            blob[f"{key}_{field}"] = track[field]
    np.savez_compressed(RAW, **blob)


def load_raw() -> list[dict]:
    blob = np.load(RAW)
    tracks = []
    for seed in SEEDS:
        for name in CONDITIONS:
            for repeat in (0, 1):
                key = f"s{seed}_{name}_{repeat}"
                if f"{key}_ped" not in blob:
                    continue
                tracks.append(
                    {
                        "seed": seed,
                        "name": name,
                        "repeat": repeat,
                        "ped": blob[f"{key}_ped"],
                        "ped_v": blob[f"{key}_ped_v"],
                        "robot": blob[f"{key}_robot"],
                        "yaw": blob[f"{key}_yaw"],
                        "command": blob[f"{key}_command"],
                    }
                )
    return tracks


def repeat_stats(tracks: list[dict]) -> dict:
    grouped: dict[tuple[int, str], dict[int, dict]] = {}
    for track in tracks:
        grouped.setdefault((track["seed"], track["name"]), {})[track["repeat"]] = track
    rows = []
    agreements = []
    for (seed, name), repeats in sorted(grouped.items()):
        if 0 not in repeats or 1 not in repeats:
            continue
        a, b = repeats[0], repeats[1]
        n = min(len(a["ped"]), len(b["ped"]))
        delta = np.linalg.norm(a["ped"][:n] - b["ped"][:n], axis=1)
        # Decision labels from the same index, using each repeat's own future.
        agree = []
        for t in range(1, n - HORIZON):
            speeds = []
            labels = []
            for track in (a, b):
                speed = float(np.linalg.norm(track["ped_v"][t]))
                ahead = float(np.linalg.norm(track["ped_v"][t + HORIZON]))
                moved = float(np.linalg.norm(track["ped"][t + HORIZON] - track["ped"][t]))
                if speed < MOVING_SPEED:
                    continue
                speeds.append(speed)
                labels.append(ahead < STOP_SPEED or moved < 0.5 * speed * DT)
            if len(labels) == 2:
                agree.append(labels[0] == labels[1])
        agreement = float(np.mean(agree)) if agree else float("nan")
        agreements.append(agreement)
        rows.append(
            {
                "seed": seed,
                "condition": name,
                "mean_position_m": float(delta.mean()),
                "max_position_m": float(delta.max()),
                "stop_label_agreement": agreement,
                "decision_samples": int(len(agree)),
            }
        )
    valid = [value for value in agreements if np.isfinite(value)]
    return {
        "pairs": rows,
        "mean_repeat_position_m": float(np.mean([row["mean_position_m"] for row in rows])),
        "mean_label_agreement": float(np.mean(valid)) if valid else float("nan"),
    }


def slice_errors(data: dict, probability: np.ndarray, density: np.ndarray, mask: np.ndarray) -> dict:
    extrap_xy = data["extrap_xy"]
    corrected_xy = data["ped_xy"] + data["ped_v"] * DT * (1.0 - probability)[:, None]
    return {
        "pairs": int(mask.sum()),
        "extrapolation_density_mse": mse(data["extrap_density"][mask], data["target_density"][mask]),
        "corrected_density_mse": mse(density[mask], data["target_density"][mask]),
        "extrapolation_position_m": position_error(extrap_xy, data["target_xy"], mask),
        "corrected_position_m": position_error(corrected_xy, data["target_xy"], mask),
    }


def analyze(tracks: list[dict], velocity_source: str) -> dict:
    parts = [pairs_from_track(track) for track in tracks if len(track["ped"]) > HORIZON + 2]
    data = stack_pairs(parts)
    train = np.isin(data["seed"], list(TRAIN_SEEDS)) & data["moving"]
    full, full_mean, full_std = fit_logistic(data["features"][train], data["label"][train])
    geo_index = [FEATURE_NAMES.index(name) for name in GEOMETRY]
    geo, geo_mean, geo_std = fit_logistic(data["features"][train][:, geo_index], data["label"][train])
    probability = np.zeros(len(data["label"]))
    probability[train] = predict_stop(full, data["features"][train], full_mean, full_std)
    test = np.isin(data["seed"], list(TEST_SEEDS))
    probability[test] = predict_stop(full, data["features"][test], full_mean, full_std)
    geo_probability = np.zeros(len(data["label"]))
    geo_probability[test] = predict_stop(geo, data["features"][test][:, geo_index], geo_mean, geo_std)
    test_index = np.flatnonzero(test)
    full_density = data["extrap_density"].copy()
    full_density[test_index] = corrected_density(
        {key: data[key][test] for key in ("ped_xy", "ped_v", "robot_future", "yaw_future")},
        probability[test],
    )
    geo_density = data["extrap_density"].copy()
    geo_density[test_index] = corrected_density(
        {key: data[key][test] for key in ("ped_xy", "ped_v", "robot_future", "yaw_future")},
        geo_probability[test],
    )

    def errors(mask: np.ndarray) -> dict:
        scored = slice_errors(data, probability, full_density, mask & test)
        geo_scored = slice_errors(data, geo_probability, geo_density, mask & test)
        gated_scored = slice_errors(data, gated_probability, gated_density, mask & test)
        scored["geometry_only_density_mse"] = geo_scored["corrected_density_mse"]
        scored["geometry_only_position_m"] = geo_scored["corrected_position_m"]
        scored["near_gate_density_mse"] = gated_scored["corrected_density_mse"]
        scored["near_gate_position_m"] = gated_scored["corrected_position_m"]
        return scored

    distance = data["distance"]
    near_gate = distance < 1.5
    gated_probability = np.where(near_gate, probability, 0.0)
    gated_density = data["extrap_density"].copy()
    gated_density[test_index] = corrected_density(
        {key: data[key][test] for key in ("ped_xy", "ped_v", "robot_future", "yaw_future")},
        gated_probability[test],
    )
    moving = data["moving"]
    stop = data["label"] > 0.5
    test_moving = test & moving
    weights = full.weight.detach().numpy().reshape(-1)
    result = {
        "horizon_s": DT,
        "velocity_source": velocity_source,
        "train_seeds": sorted(TRAIN_SEEDS),
        "test_seeds": sorted(TEST_SEEDS),
        "train_decision_samples": int(train.sum()),
        "test_pairs": int(test.sum()),
        "stop_definition": f"currently >= {MOVING_SPEED} m/s, and within {DT:.0f}s speed < {STOP_SPEED} m/s or progress < half of coasting",
        "model": "logistic regression on 9 interaction features, balanced BCE, 500 Adam steps",
        "repeatability": repeat_stats(tracks),
        "stop_continue_accuracy": {
            "held_out_all_moving": classification(data["label"], probability, test_moving),
            "held_out_within_1.5m": classification(data["label"], probability, test_moving & (distance < 1.5)),
            "held_out_block": classification(data["label"], probability, test_moving & (data["name"] == "block")),
            "held_out_leave": classification(data["label"], probability, test_moving & (data["name"] == "leave")),
            "held_out_aside": classification(data["label"], probability, test_moving & (data["name"] == "aside")),
            "geometry_only_all_moving": classification(data["label"], geo_probability, test_moving),
            "train_all_moving": classification(data["label"], probability, train),
        },
        "feature_weights": {name: float(weights[i]) for i, name in enumerate(FEATURE_NAMES)},
        "density_and_position": {
            "all": errors(np.ones(len(test), dtype=bool)),
            "true_stop": errors(stop),
            "true_continue": errors(moving & ~stop),
            "distance_0_to_1.5m": errors(distance < 1.5),
            "distance_1.5_to_3m": errors((distance >= 1.5) & (distance < 3.0)),
            "distance_3_to_10m": errors((distance >= 3.0) & (distance < 10.0)),
            "block": errors(data["name"] == "block"),
            "leave": errors(data["name"] == "leave"),
            "aside": errors(data["name"] == "aside"),
        },
        "per_test_seed": {},
    }
    for seed in sorted(TEST_SEEDS):
        result["per_test_seed"][str(seed)] = {
            "accuracy": classification(data["label"], probability, test_moving & (data["seed"] == seed)),
            "errors": errors(data["seed"] == seed),
            "near_errors": errors((data["seed"] == seed) & (distance < 1.5)),
        }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def repair() -> None:
    blob = np.load(RAW)
    saved = {key: blob[key] for key in blob.files}
    rclpy.init()
    node = Collector()
    try:
        node.wait(1.0)
        for seed, name, repeat in REPAIR:
            track = stack_episode(node.record(seed, name, repeat))
            if np.linalg.norm(track["ped"][0] - CANONICAL_SPAWN) > 0.15:
                raise RuntimeError(f"spawn miss remained for seed {seed} {name}_{repeat}: {track['ped'][0].tolist()}")
            key = f"s{seed}_{name}_{repeat}"
            for field in ("ped", "ped_v", "robot", "yaw", "command"):
                saved[f"{key}_{field}"] = track[field]
    finally:
        node.command = np.zeros(2)
        node.wait(0.2)
        node.destroy_node()
        rclpy.shutdown()
    np.savez_compressed(RAW, **saved)


def collect() -> list[dict]:
    rclpy.init()
    node = Collector()
    episodes = []
    try:
        node.wait(1.0)
        for seed in SEEDS:
            for name in CONDITIONS:
                for repeat in (0, 1):
                    episodes.append(node.record(seed, name, repeat))
                    save_raw([stack_episode(episode) for episode in episodes])
    finally:
        node.command = np.zeros(2)
        node.wait(0.2)
        node.destroy_node()
        rclpy.shutdown()
    return [stack_episode(episode) for episode in episodes]


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--repair":
        repair()
    if len(sys.argv) > 1 and sys.argv[1] in {"--from-raw", "--repair"}:
        tracks = load_raw()
        tracks, source = observed_velocity(tracks)
    else:
        tracks = collect()
        tracks, source = observed_velocity(tracks)
        save_raw(tracks)
    matched = [track for track in tracks if np.linalg.norm(track["ped"][0] - CANONICAL_SPAWN) <= 0.15]
    skipped = [f"s{track['seed']}_{track['name']}_{track['repeat']}" for track in tracks if track not in matched]
    if skipped:
        raise RuntimeError(f"episodes did not start at the shared spawn: {skipped}")
    analyze(matched, source)


if __name__ == "__main__":
    main()
