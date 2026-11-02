"""Reset the head-on episode and drive one robot command while a camera records.

Prints DRIVING when the command clock starts, so the recording can begin then.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrians
from arena_runtime_msgs.srv import LifecycleHold
from geometry_msgs.msg import Twist
from rclpy.node import Node
from task_generator_msgs.srv import ResetEpisode

SEED = 5
SAMPLE_HZ = 50.0
SECONDS = 16.0
SPAWN = np.array([21.3, 7.3])
RESET_SERVICE = "/arena/env_0/task_generator_node/lifecycle/reset_episode"


def command_at(name: str, step: int) -> np.ndarray:
    # Steps are 50 Hz. The diagnostic used 10 Hz, so multiply those thresholds by 5.
    if name == "block":
        return np.zeros(2)
    if name == "leave":
        if step < 50:
            return np.array([0.0, 1.0])
        if step < 160:
            return np.array([0.45, 0.0])
        return np.zeros(2)
    if step < 50:
        return np.array([0.0, 1.0])
    if step < 140:
        return np.array([0.45, 0.0])
    return np.zeros(2)


class Driver(Node):
    def __init__(self) -> None:
        super().__init__("mf_video_drive")
        self.ped = None
        self.command = np.zeros(2)
        self.pub = self.create_publisher(Twist, "/arena/env_0/task_generator_node/jackal/cmd_vel", 10)
        self.create_subscription(Pedestrians, "/arena/env_0/arena_peds", self._on_peds, 10)
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE)
        self.hold_client = self.create_client(LifecycleHold, "/arena/sim_lifecycle/hold")
        self.create_timer(1.0 / SAMPLE_HZ, self._publish)
        self._holding = False

    def _on_peds(self, msg: Pedestrians) -> None:
        if msg.pedestrians:
            person = msg.pedestrians[0]
            self.ped = np.array([person.pose.position.x, person.pose.position.y])

    def _publish(self) -> None:
        msg = Twist()
        msg.linear.x = float(self.command[0])
        msg.angular.z = float(self.command[1])
        self.pub.publish(msg)

    def wait(self, seconds: float) -> None:
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        wall_end = time.monotonic() + max(seconds, 1.0) + 5.0
        while rclpy.ok() and self.get_clock().now().nanoseconds < end and time.monotonic() < wall_end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def hold(self, acquire: bool) -> None:
        if not self.hold_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("sim hold service unavailable")
        request = LifecycleHold.Request()
        request.action = LifecycleHold.Request.ACQUIRE if acquire else LifecycleHold.Request.RELEASE
        request.caller_id = "mf_video_drive"
        request.reason = "proof"
        future = self.hold_client.call_async(request)
        deadline = time.monotonic() + 20.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None:
            raise RuntimeError("sim hold call failed")
        self._holding = acquire

    def settle(self) -> None:
        if not self.reset_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("reset service unavailable")
        self.wait(0.5)
        for _attempt in range(4):
            before = None if self.ped is None else self.ped.copy()
            future = self.reset_client.call_async(ResetEpisode.Request(world="", seed=SEED))
            deadline = time.monotonic() + 30.0
            while rclpy.ok() and not future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                if time.monotonic() > deadline:
                    raise RuntimeError("reset timed out")
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError("reset failed")
            history = []
            start = self.get_clock().now().nanoseconds
            wall_end = time.monotonic() + 30.0
            while rclpy.ok() and self.get_clock().now().nanoseconds - start < 12e9 and time.monotonic() < wall_end:
                rclpy.spin_once(self, timeout_sec=0.02)
                if self.ped is None:
                    continue
                history.append(self.ped.copy())
                xs = np.array([point[0] for point in history])
                peak = history[int(xs.argmax())]
                if np.linalg.norm(peak - SPAWN) > 0.15:
                    continue
                if len(history) < 3:
                    continue
                if abs(float(self.ped[0]) - float(peak[0])) > 0.25:
                    break
                if float(self.ped[0]) < float(peak[0]) - 0.04:
                    self.hold(True)
                    print(f"READY {self.ped.tolist()}", flush=True)
                    return
        raise RuntimeError("pedestrian did not respawn at the waypoint start")

    def drive(self, name: str) -> None:
        print("DRIVING", flush=True)
        deadline = time.monotonic() + 60.0
        while rclpy.ok() and not os.path.exists("/tmp/mf_vid/go"):
            if time.monotonic() > deadline:
                raise RuntimeError("camera did not start")
            rclpy.spin_once(self, timeout_sec=0.05)
        self.hold(False)
        print("RELEASED", flush=True)
        steps = int(SECONDS * SAMPLE_HZ)
        next_tick = self.get_clock().now().nanoseconds
        period = int(1e9 / SAMPLE_HZ)
        next_report = 0
        wall_end = time.monotonic() + 150.0
        for step in range(steps):
            self.command = command_at(name, step)
            while self.get_clock().now().nanoseconds < next_tick and rclpy.ok():
                if time.monotonic() > wall_end:
                    raise RuntimeError("drive timed out waiting for the recording to step the sim")
                rclpy.spin_once(self, timeout_sec=0.01)
            if step >= next_report and self.ped is not None:
                print(f"STEP {step / SAMPLE_HZ:.1f} ped {self.ped.tolist()}", flush=True)
                next_report += int(SAMPLE_HZ)
            next_tick += period
        self.command = np.zeros(2)


def main() -> None:
    name = sys.argv[1]
    rclpy.init()
    node = Driver()
    try:
        node.hold(False)
        node.settle()
        node.drive(name)
    finally:
        node.command = np.zeros(2)
        if node._holding:
            node.hold(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
