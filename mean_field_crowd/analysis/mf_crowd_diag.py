"""Crowd response for ORCA and PedVO on the generated parallel-lane scenarios.

Each pedestrian has their own straight waypoint. The robot starts on the
center lane. Block holds the robot there. Leave, used only for five
pedestrians, turns off that lane. Lateral deviation is measured from each
pedestrian's own starting y, so neighbor avoidance and robot avoidance can
be separated by the block-versus-leave comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy

from mf_planner_diag import Collector, command_at

SEEDS = (5, 6, 7)
REPEATS = 3
OUT = Path("/opt/arena_ws/data/mf_crowd_diag.json")


def reset_crowd(node: Collector, seed: int, count: int) -> None:
    from task_generator_msgs.srv import ResetEpisode

    if not node.reset_client.wait_for_service(timeout_sec=15.0):
        raise RuntimeError("reset service unavailable")
    future = node.reset_client.call_async(ResetEpisode.Request(world="", seed=int(seed)))
    deadline = time.monotonic() + 30.0
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not future.done() or future.result() is None or not future.result().success:
        raise RuntimeError("crowd reset failed")
    deadline = time.monotonic() + 25.0
    peak = None
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.peds is None or len(node.peds.pedestrians) < count:
            continue
        xs = [person.pose.position.x for person in node.peds.pedestrians[:count]]
        peak = max(xs) if peak is None else max(peak, max(xs))
        if peak - max(xs) > 0.05:
            return
    raise RuntimeError(f"crowd did not start walking, count={0 if node.peds is None else len(node.peds.pedestrians)}")


def record_all(node: Collector, planner: str, name: str, seed: int, repeat: int, count: int) -> dict:
    reset_crowd(node, seed, count)
    samples = []
    next_tick = node.get_clock().now().nanoseconds
    period = int(1e9 / 10.0)
    wall_end = time.monotonic() + 45.0
    while len(samples) < 70 and rclpy.ok() and time.monotonic() < wall_end:
        node.command = command_at(name, len(samples))
        while node.get_clock().now().nanoseconds < next_tick and time.monotonic() < wall_end:
            rclpy.spin_once(node, timeout_sec=0.01)
        next_tick += period
        pose = node.robot_pose()
        if pose is None or node.peds is None or len(node.peds.pedestrians) < count:
            continue
        people = []
        for person in node.peds.pedestrians[:count]:
            people.append(
                [
                    person.pose.position.x,
                    person.pose.position.y,
                    person.twist.linear.x,
                    person.twist.linear.y,
                ]
            )
        samples.append((pose["x"], pose["y"], people))
    node.command = np.zeros(2)
    print(f"{planner} n={count} seed {seed} {name}_{repeat}: {len(samples)}", flush=True)
    return {"planner": planner, "count": count, "seed": seed, "name": name, "repeat": repeat, "samples": samples}


def summarize(raw: list[dict]) -> dict:
    rows = []
    for episode in raw:
        samples = episode["samples"]
        if len(samples) < 10:
            continue
        robot = np.array([[sample[0], sample[1]] for sample in samples])
        people = np.array([sample[2] for sample in samples])  # T, N, 4
        spawn = people[0, :, :2]
        end = people[-1]
        lateral = np.max(np.abs(people[:, :, 1] - spawn[:, 1]), axis=0)
        speed = np.linalg.norm(end[:, 2:4], axis=1)
        separation = np.min(np.linalg.norm(people[:, :, :2] - robot[:, None, :], axis=2), axis=0)
        passed = end[:, 0] < robot[0, 0] - 0.4
        stopped = (speed < 0.25) & (end[:, 0] > robot[:, 0].min() + 0.4)
        avoided = (lateral > 0.30) & passed
        rows.append(
            {
                "planner": episode["planner"],
                "count": episode["count"],
                "seed": episode["seed"],
                "name": episode["name"],
                "repeat": episode["repeat"],
                "avoid_rate": float(np.mean(avoided)),
                "stop_rate": float(np.mean(stopped)),
                "pass_rate": float(np.mean(passed)),
                "mean_max_lateral_m": float(np.mean(lateral)),
                "min_separation_m": float(np.min(separation)),
                "center_lateral_m": float(lateral[int(np.argmin(np.abs(spawn[:, 1] - robot[0, 1])))]),
            }
        )
    cells = {}
    for row in rows:
        key = f"{row['planner']}:{row['count']}:{row['name']}"
        cells.setdefault(key, []).append(row)
    summary = {}
    for key, group in cells.items():
        def mean(name: str, group: list[dict] = group) -> float:
            return float(np.mean([item[name] for item in group]))

        summary[key] = {
            "n": len(group),
            "avoid_rate": mean("avoid_rate"),
            "stop_rate": mean("stop_rate"),
            "pass_rate": mean("pass_rate"),
            "mean_max_lateral_m": mean("mean_max_lateral_m"),
            "min_separation_m": mean("min_separation_m"),
            "center_lateral_m": mean("center_lateral_m"),
        }
    return {"episodes": rows, "cells": summary}


def main() -> None:
    planner = sys.argv[1]
    count = int(sys.argv[2])
    names = ("block", "leave") if count == 5 else ("block",)
    rclpy.init()
    node = Collector()
    raw = []
    try:
        error = node.set_planner(planner)
        if error:
            raise RuntimeError(error)
        for seed in SEEDS:
            for name in names:
                for repeat in range(REPEATS):
                    raw.append(record_all(node, planner, name, seed, repeat, count))
    finally:
        node.command = np.zeros(2)
        node.destroy_node()
        rclpy.shutdown()
    previous = json.loads(OUT.read_text()) if OUT.exists() else {"raw": []}
    previous["raw"].extend(raw)
    previous["summary"] = summarize(previous["raw"])
    OUT.write_text(json.dumps(previous, indent=2))
    print(json.dumps(previous["summary"]["cells"], indent=2), flush=True)


if __name__ == "__main__":
    main()
