"""Online constant-velocity baseline for the recorded PedVO episodes.

The robot pose one second ahead is the unicycle integral of the previous
command [linear.x, angular.z], held constant. Pedestrians move at their
current world velocity. Neither future poses nor future commands are used.
The oracle rasterizes the same pedestrian forecast in the recorded future
robot frame and is diagnostic only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mf_pedvo_forecast import (
    BOOTSTRAP,
    DATA,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    episode_sse,
    load_episodes,
    pooled,
    rasterize_density,
)
from rosnav_rl.observations.utils.semantic import get_relative_pos_to_robot

DT = 1.0
W_EPS = 1e-3
CPA_HORIZON = 2.0
OUT = Path("/opt/arena_ws/data/mf_pedvo_online.json")
LATERAL_ON = 0.30
LATERAL_OFF = 0.10


def integrate_unicycle(x, y, yaw, v, w, tau):
    """Unicycle step. ``tau`` broadcasts on the right of the state arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    small = np.abs(w) < W_EPS
    yaw_next = yaw + w * tau
    v_over_w = np.divide(v, np.where(small, 1.0, w))
    straight_x = x + v * np.cos(yaw) * tau
    straight_y = y + v * np.sin(yaw) * tau
    arc_x = x + v_over_w * (np.sin(yaw_next) - np.sin(yaw))
    arc_y = y + v_over_w * (-np.cos(yaw_next) + np.cos(yaw))
    return (
        np.where(small, straight_x, arc_x),
        np.where(small, straight_y, arc_y),
        np.where(small, yaw, yaw_next),
    )


def to_robot(points: np.ndarray, x, y, yaw) -> np.ndarray:
    dx = points[..., 0] - x
    dy = points[..., 1] - y
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.stack((cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy), axis=-1)


def rasterize_rows(points: np.ndarray, batch: int = 2048) -> np.ndarray:
    chunks = []
    for start in range(0, len(points), batch):
        chunks.append(rasterize_density(points[start : start + batch])[:, 0])
    return np.concatenate(chunks, axis=0)


def bootstrap_delta(reference, other) -> dict:
    sse, count = reference
    present = np.flatnonzero(count > 0)
    result = {"mse": pooled(sse, count), "episodes": int(len(present)), "elements": int(count.sum())}
    if len(present) < 2:
        result["delta"] = {"mean_delta": float("nan"), "ci95": [float("nan"), float("nan")]}
        return result
    rng = np.random.default_rng(0)
    draws = present[rng.integers(0, len(present), size=(BOOTSTRAP, len(present)))]
    delta = np.empty(BOOTSTRAP)
    other_sse, other_count = other
    for row, draw in enumerate(draws):
        delta[row] = pooled(sse, count, draw) - pooled(other_sse, other_count, draw)
    result["delta"] = {"mean_delta": float(delta.mean()), "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))]}
    return result


def pairs_of(episode: dict) -> dict:
    ped = episode["ped"]
    vel = episode["vel"]
    robot = episode["robot"]
    yaw = episode["yaw"]
    command = episode["command"]
    index = np.arange(1, len(ped) - 10)
    ped_t = ped[index]
    vel_t = vel[index]
    ped_cv = ped_t + vel_t * DT
    action = command[index - 1]
    pred_x, pred_y, pred_yaw = integrate_unicycle(robot[index, 0], robot[index, 1], yaw[index], action[:, 0], action[:, 1], DT)
    online_points = to_robot(ped_cv, pred_x[:, None], pred_y[:, None], pred_yaw[:, None])
    oracle_points = np.empty_like(online_points)
    ones = np.ones((ped.shape[1], 1), dtype=np.float64)
    for row, t in enumerate(index):
        homo = np.concatenate([ped_cv[row], ones], axis=1)
        pose = {"x": float(robot[t + 10, 0]), "y": float(robot[t + 10, 1]), "yaw": float(yaw[t + 10])}
        oracle_points[row] = get_relative_pos_to_robot(pose, homo)
    actual = robot[index + 10]
    actual_yaw = yaw[index + 10]
    center = episode["center"]
    current_lat = np.abs(ped_t[:, center, 1] - episode["spawn_y"][center])
    future_lat = np.abs(ped[index + 10, center, 1] - episode["spawn_y"][center])
    others = [i for i in range(ped.shape[1]) if i != center]
    if others:
        neighbor_lat = np.max(np.abs(ped[index + 10][:, others, 1] - episode["spawn_y"][others]), axis=1)
    else:
        neighbor_lat = np.zeros(len(index))
    relative = episode["relative"][index]
    return {
        "target": episode["density"][index + 10],
        "persistence": episode["density"][index],
        "online": rasterize_rows(online_points.astype(np.float64)),
        "oracle": rasterize_rows(oracle_points.astype(np.float64)),
        "pose_error_m": np.linalg.norm(np.stack((pred_x, pred_y), axis=1) - actual, axis=1),
        "yaw_error_rad": np.arctan2(np.sin(pred_yaw - actual_yaw), np.cos(pred_yaw - actual_yaw)),
        "current_lat": current_lat,
        "future_lat": future_lat,
        "neighbor_lat": neighbor_lat,
        "distance": np.linalg.norm(relative[:, center], axis=1),
        "count": np.full(len(index), episode["count"]),
        "name": np.array([episode["name"]] * len(index)),
        "action": action,
    }


def masks(data: dict) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(data["distance"]), dtype=bool),
        "anticipation": (data["current_lat"] < LATERAL_OFF) & (data["future_lat"] > LATERAL_ON) & (data["distance"] < 4.0),
        "active_avoidance": (data["current_lat"] >= LATERAL_ON) & (data["distance"] < 3.0),
        "non_interaction": (data["future_lat"] < LATERAL_OFF) & (data["neighbor_lat"] < LATERAL_OFF),
        **{f"n{count}": data["count"] == count for count in (1, 5, 15, 20)},
        **{f"cmd_{name}": data["name"] == name for name in ("block", "leave", "approach", "slow", "cruise", "yaw")},
    }


def evaluate(episodes: list[dict]) -> dict:
    parts = [pairs_of(episode) for episode in episodes]
    bounds = np.cumsum([0] + [len(part["distance"]) for part in parts])
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0] if key != "name"}
    data["name"] = np.concatenate([part["name"] for part in parts])
    target = data["target"]
    predictions = {
        "persistence": data["persistence"],
        "online_extrapolation": data["online"],
        "oracle_extrapolation": data["oracle"],
    }
    cells = {}
    for label, mask in masks(data).items():
        packed = {name: episode_sse(prediction, target, bounds, mask) for name, prediction in predictions.items()}
        cell = {"pairs": int(mask.sum())}
        for name, parts_sse in packed.items():
            cell[name] = pooled(*parts_sse)
        if mask.any():
            cell["online_vs_persistence"] = bootstrap_delta(packed["online_extrapolation"], packed["persistence"])["delta"]
            cell["online_vs_oracle"] = bootstrap_delta(packed["online_extrapolation"], packed["oracle_extrapolation"])["delta"]
        cells[label] = cell
    pose = {}
    for name in ("block", "leave", "approach", "slow", "cruise", "yaw"):
        chosen = data["name"] == name
        pose[name] = {
            "pairs": int(chosen.sum()),
            "mean_position_error_m": float(data["pose_error_m"][chosen].mean()),
            "mean_abs_yaw_error_rad": float(np.mean(np.abs(data["yaw_error_rad"][chosen]))),
        }
    return {"episodes": len(episodes), "pairs": int(len(data["distance"])), "cells": cells, "command_pose_error": pose}


def main() -> None:
    episodes = load_episodes()
    result = {
        "kinematics": "unicycle integral of previous command [linear.x, angular.z] held for 1.0 s",
        "pedestrians": "current world position plus current world velocity times 1.0 s",
        "oracle": "same pedestrian forecast, rasterized in the recorded robot pose at t+1 s",
        "splits": {},
    }
    for name, seeds in (("train", TRAIN_SEEDS), ("validation", VAL_SEEDS), ("test", TEST_SEEDS)):
        chosen = [episode for episode in episodes if episode["seed"] in seeds]
        print(f"evaluating {name}", flush=True)
        result["splits"][name] = evaluate(chosen)
        test_all = result["splits"][name]["cells"]["all"]
        print(
            name,
            "persistence",
            round(test_all["persistence"], 6),
            "online",
            round(test_all["online_extrapolation"], 6),
            "oracle",
            round(test_all["oracle_extrapolation"], 6),
            flush=True,
        )
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["splits"]["test"]["cells"], indent=2))
    print(json.dumps(result["splits"]["test"]["command_pose_error"], indent=2))


if __name__ == "__main__":
    main()
