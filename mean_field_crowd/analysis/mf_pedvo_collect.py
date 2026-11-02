"""Collect PedVO episodes for the offline CrowdResponseNet comparison.

The command list, seeds, horizon, and sample rate are fixed here, before any
forecast is trained. Launch the matching scenario first:

    1 pedestrian  -> task.scenario.file:=mf_headon
    5, 15, 20     -> task.scenario.file:=mf_crowd5 / mf_crowd15 / mf_crowd20

Usage: python mf_pedvo_collect.py <count>
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy

from mf_crowd_diag import reset_crowd
from mf_planner_diag import Collector

SEEDS = (5, 6, 7, 8, 9, 10)
REPEATS = 2
CONDITIONS = ("block", "leave", "approach", "slow", "cruise", "yaw")
SAMPLE_HZ = 10.0
EPISODE_STEPS = 70
OUT = Path("/opt/arena_ws/data/mf_pedvo")


def command_at(name: str, step: int) -> np.ndarray:
    """Previous-action values are [linear.x, angular.z]. Schedules are open-loop."""
    if name == "block":
        return np.zeros(2)
    if name == "leave":
        return np.array([0.0, 1.0]) if step < 16 else np.array([0.6, 0.0])
    if name == "approach":
        return np.array([0.8, 0.0])
    if name == "slow":
        return np.array([0.3, 0.0])
    if name == "cruise":
        return np.array([0.5, 0.0])
    if name == "yaw":
        return np.array([0.0, 0.7]) if step < 20 else np.array([0.4, 0.0])
    raise KeyError(name)


def people_of(node: Collector, count: int):
    if node.peds is None or len(node.peds.pedestrians) < count:
        return None

    def key(person):
        try:
            return (0, int(person.name))
        except ValueError:
            return (1, person.name)

    return sorted(node.peds.pedestrians, key=key)[:count]


def cut_length(ped: np.ndarray, robot: np.ndarray) -> int:
    if len(ped) < 2:
        return len(ped)
    ped_step = np.linalg.norm(np.diff(ped, axis=0), axis=-1).max(axis=1)
    robot_step = np.linalg.norm(np.diff(robot, axis=0), axis=1)
    jumps = np.flatnonzero(np.maximum(ped_step, robot_step) > 0.5)
    return int(jumps[0]) + 1 if len(jumps) else len(ped)


def record(node: Collector, count: int, seed: int, name: str, repeat: int) -> dict:
    if count == 1:
        node.reset_and_settle(seed)
    else:
        reset_crowd(node, seed, count)
    command, robot, yaw, ped, vel = [], [], [], [], []
    next_tick = node.get_clock().now().nanoseconds
    period = int(1e9 / SAMPLE_HZ)
    wall_end = time.monotonic() + 45.0
    while len(command) < EPISODE_STEPS and rclpy.ok() and time.monotonic() < wall_end:
        node.command = command_at(name, len(command))
        while node.get_clock().now().nanoseconds < next_tick and time.monotonic() < wall_end:
            rclpy.spin_once(node, timeout_sec=0.01)
        next_tick += period
        pose = node.robot_pose()
        people = people_of(node, count)
        if pose is None or people is None:
            continue
        command.append(node.command.copy())
        robot.append([pose["x"], pose["y"]])
        yaw.append(pose["yaw"])
        ped.append([[person.pose.position.x, person.pose.position.y] for person in people])
        vel.append([[person.twist.linear.x, person.twist.linear.y] for person in people])
    node.command = np.zeros(2)
    ped_a = np.asarray(ped, dtype=np.float64)
    robot_a = np.asarray(robot, dtype=np.float64)
    kept = cut_length(ped_a, robot_a) if len(ped_a) else 0
    print(
        f"n={count} seed {seed} {name}_{repeat}: {len(command)} samples, kept {kept}",
        flush=True,
    )
    return {
        "command": np.asarray(command, dtype=np.float64)[:kept],
        "robot": robot_a[:kept],
        "yaw": np.asarray(yaw, dtype=np.float64)[:kept],
        "ped": ped_a[:kept],
        "vel": np.asarray(vel, dtype=np.float64)[:kept],
        "kept": kept,
    }


def main() -> None:
    import sys

    count = int(sys.argv[1])
    folder = OUT / f"n{count}"
    folder.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Collector()
    try:
        error = node.set_planner("pedvo")
        if error:
            raise RuntimeError(error)
        for seed in SEEDS:
            for name in CONDITIONS:
                for repeat in range(REPEATS):
                    path = folder / f"seed{seed}_{name}_{repeat}.npz"
                    if path.exists():
                        print(f"skip {path.name}", flush=True)
                        continue
                    episode = record(node, count, seed, name, repeat)
                    np.savez_compressed(path, **episode)
    finally:
        node.command = np.zeros(2)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
