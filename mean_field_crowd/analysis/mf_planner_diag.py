"""Paired robot-response test for the pedestrian planners installed in Arena.

The head-on scenario is mf_headon: one arenian walks from map x=21.3 toward
x=16.3 along y=7.3, and the robot starts on that line at about x=17.3.
mf_walk is not used because the robot starts behind the pedestrian, so a
block command never meets them. Planner parameters are left at their defaults.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrians
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from task_generator_msgs.srv import ResetEpisode
from tf2_ros import Buffer, TransformListener

SEED_LIST = (5, 6, 7)
REPEATS = 3
CONDITIONS = ("block", "leave", "approach")
PLANNERS = ("sfm", "hsfm", "orca", "pedvo", "socialgail", "nsp")
SAMPLE_HZ = 10.0
EPISODE_STEPS = 70
SPAWN = np.array([21.3, 7.3])
RESET_SERVICE = "/arena/env_0/task_generator_node/lifecycle/reset_episode"
PARAM_SERVICE = "/arena/env_0/task_generator_node/arena_humansim/set_parameters"
OUT = Path("/opt/arena_ws/data/mf_planner_diag.json")
RAW = Path("/opt/arena_ws/data/mf_planner_diag.npz")


def command_at(name: str, step: int) -> np.ndarray:
    if name == "block":
        return np.zeros(2)
    if name == "leave":
        if step < 16:
            return np.array([0.0, 1.0])
        return np.array([0.6, 0.0])
    return np.array([0.8, 0.0])


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def cut_index(ped: np.ndarray, robot: np.ndarray) -> int:
    if len(ped) < 2:
        return len(ped)
    step = np.maximum(np.linalg.norm(np.diff(ped, axis=0), axis=1), np.linalg.norm(np.diff(robot, axis=0), axis=1))
    jumps = np.flatnonzero(step > 0.5)
    return int(jumps[0]) + 1 if len(jumps) else len(ped)


class Collector(Node):
    def __init__(self) -> None:
        super().__init__("mf_planner_diag")
        self.peds: Pedestrians | None = None
        self.command = np.zeros(2)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(Twist, "/arena/env_0/task_generator_node/jackal/cmd_vel", 10)
        self.create_subscription(Pedestrians, "/arena/env_0/arena_peds", self._on_peds, 10)
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE)
        self.param_client = self.create_client(SetParameters, PARAM_SERVICE)
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

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def set_planner(self, name: str) -> str:
        if not self.param_client.wait_for_service(timeout_sec=15.0):
            return "parameter service unavailable"
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="local_planner",
                value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=name),
            )
        ]
        future = self.param_client.call_async(request)
        deadline = time.monotonic() + 180.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None:
            return "parameter call timed out"
        result = future.result().results[0]
        if not result.successful:
            return result.reason or "parameter rejected"
        self.spin_for(0.5)
        return ""

    def reset_and_settle(self, seed: int) -> np.ndarray:
        if not self.reset_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("reset service unavailable")
        last_error = "pedestrian did not respawn"
        for _attempt in range(4):
            future = self.reset_client.call_async(ResetEpisode.Request(world="", seed=int(seed)))
            deadline = time.monotonic() + 30.0
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            if not future.done():
                raise RuntimeError("reset timed out")
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError(f"reset failed: {None if response is None else response.error_msg}")
            history: list[np.ndarray] = []
            deadline = time.monotonic() + 20.0
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.02)
                state = self.ped_state()
                if state is None:
                    continue
                history.append(state[0].copy())
                xs = np.array([point[0] for point in history])
                peak = history[int(xs.argmax())]
                if np.linalg.norm(peak - SPAWN) > 0.15 or len(history) < 3:
                    continue
                if float(state[0][0]) < float(peak[0]) - 0.04 and abs(float(state[0][0]) - float(peak[0])) < 0.25:
                    return state[0].copy()
            last_error = f"spawn not matched, last {history[-1].tolist() if history else None}"
        raise RuntimeError(last_error)

    def record(self, planner: str, seed: int, name: str, repeat: int) -> dict:
        spawn = self.reset_and_settle(seed)
        samples = []
        next_tick = self.get_clock().now().nanoseconds
        period = int(1e9 / SAMPLE_HZ)
        wall_end = time.monotonic() + 40.0
        while len(samples) < EPISODE_STEPS and rclpy.ok() and time.monotonic() < wall_end:
            self.command = command_at(name, len(samples))
            while self.get_clock().now().nanoseconds < next_tick and time.monotonic() < wall_end:
                rclpy.spin_once(self, timeout_sec=0.01)
            next_tick += period
            pose = self.robot_pose()
            state = self.ped_state()
            if pose is None or state is None:
                continue
            ped, ped_v = state
            samples.append((self.command.copy(), np.array([pose["x"], pose["y"]]), pose["yaw"], ped.copy(), ped_v.copy()))
        self.command = np.zeros(2)
        self.spin_for(0.1)
        print(f"{planner} seed {seed} {name}_{repeat}: {len(samples)} samples spawn {spawn.round(3).tolist()}", flush=True)
        return {"planner": planner, "seed": seed, "name": name, "repeat": repeat, "spawn": spawn, "samples": samples}


def episode_arrays(episode: dict) -> dict:
    samples = episode["samples"]
    ped = np.stack([sample[3] for sample in samples])
    ped_v = np.stack([sample[4] for sample in samples])
    robot = np.stack([sample[1] for sample in samples])
    command = np.stack([sample[0] for sample in samples])
    end = cut_index(ped, robot)
    return {
        "ped": ped[:end],
        "ped_v": ped_v[:end],
        "robot": robot[:end],
        "command": command[:end],
        "spawn": np.asarray(episode["spawn"], dtype=np.float64),
        "cut": end < len(ped),
    }


def metrics(track: dict) -> dict:
    ped = track["ped"]
    robot = track["robot"]
    spawn = track["spawn"]
    lateral = ped[:, 1] - spawn[1]
    speed = np.linalg.norm(track["ped_v"], axis=1)
    separation = np.linalg.norm(ped - robot, axis=1)
    progress = float(spawn[0] - ped[-1, 0])
    return {
        "n": int(len(ped)),
        "cut": bool(track["cut"]),
        "mean_lateral_m": float(np.mean(np.abs(lateral))),
        "max_lateral_m": float(np.max(np.abs(lateral))),
        "signed_lateral_end_m": float(lateral[-1]),
        "progress_m": progress,
        "end_speed_mps": float(speed[-1]),
        "mean_speed_mps": float(np.mean(speed)),
        "stopped": bool(speed[-1] < 0.25 and ped[-1, 0] > float(np.min(robot[:, 0])) + 0.4),
        "passed_robot": bool(ped[-1, 0] < float(robot[0, 0]) - 0.4),
        "min_separation_m": float(np.min(separation)),
        "close_approach": bool(np.min(separation) < 0.5),
        "robot_in_view_fraction": float(np.mean(separation < 5.0)),
    }


def pairwise(tracks: list[dict]) -> float:
    diffs = []
    for left in range(len(tracks)):
        for right in range(left + 1, len(tracks)):
            n = min(len(tracks[left]["ped"]), len(tracks[right]["ped"]))
            if n < 5:
                continue
            diffs.append(float(np.mean(np.linalg.norm(tracks[left]["ped"][:n] - tracks[right]["ped"][:n], axis=1))))
    return float(np.mean(diffs)) if diffs else float("nan")


def summarize(episodes: list[dict]) -> dict:
    grouped: dict[tuple, list] = {}
    rows = []
    for episode in episodes:
        track = episode_arrays(episode)
        row = metrics(track)
        row.update({key: episode[key] for key in ("planner", "seed", "name", "repeat")})
        rows.append(row)
        grouped.setdefault((episode["planner"], episode["name"], episode["seed"]), []).append(track)
    by_cell: dict[str, dict] = {}
    for planner in PLANNERS:
        planner_rows = [row for row in rows if row["planner"] == planner]
        if not planner_rows:
            continue
        cells = {}
        for name in CONDITIONS:
            cell = [row for row in planner_rows if row["name"] == name]
            if not cell:
                continue
            def column(key: str, cell: list[dict] = cell) -> list[float]:
                return [float(row[key]) for row in cell]

            def mean_ci(values: list[float]) -> dict:
                arr = np.asarray(values, dtype=np.float64)
                mean = float(arr.mean())
                se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
                return {"mean": mean, "ci95": 1.96 * se if len(arr) > 1 else float("nan"), "n": len(arr)}

            repeats = [pairwise(grouped[(planner, name, seed)]) for seed in SEED_LIST if (planner, name, seed) in grouped]
            cells[name] = {
                "mean_lateral_m": mean_ci(column("mean_lateral_m")),
                "max_lateral_m": mean_ci(column("max_lateral_m")),
                "progress_m": mean_ci(column("progress_m")),
                "end_speed_mps": mean_ci(column("end_speed_mps")),
                "min_separation_m": mean_ci(column("min_separation_m")),
                "stop_rate": float(np.mean([row["stopped"] for row in cell])),
                "pass_rate": float(np.mean([row["passed_robot"] for row in cell])),
                "close_rate": float(np.mean([row["close_approach"] for row in cell])),
                "in_view": float(np.mean([row["robot_in_view_fraction"] for row in cell])),
                "repeat_position_m": float(np.nanmean(repeats)),
            }
        by_cell[planner] = cells
    return {"episodes": rows, "cells": by_cell}


def main() -> None:
    rclpy.init()
    node = Collector()
    episodes: list[dict] = []
    failures: dict[str, str] = {}
    try:
        for planner in PLANNERS:
            error = node.set_planner(planner)
            if error:
                failures[planner] = error
                print(f"{planner} FAILED: {error}", flush=True)
                continue
            try:
                for seed in SEED_LIST:
                    for name in CONDITIONS:
                        for repeat in range(REPEATS):
                            episodes.append(node.record(planner, seed, name, repeat))
            except Exception as exc:
                failures[planner] = f"{type(exc).__name__}: {exc}"
                print(f"{planner} FAILED during episodes: {exc}", flush=True)
        payload = summarize(episodes)
        payload["failures"] = failures
        payload["seeds"] = list(SEED_LIST)
        payload["repeats"] = REPEATS
        payload["scenario"] = "mf_headon"
        payload["versions"] = {"arena": "c9de66d1", "humansim": "022b183"}
        OUT.write_text(json.dumps(payload, indent=2))
        print(f"wrote {OUT}", flush=True)
    finally:
        node.command = np.zeros(2)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
