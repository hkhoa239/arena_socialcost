"""Closed-loop social cost estimate for Nav2 (overlay socialcost_estimate).

Publishes nav_msgs/OccupancyGrid on <env>/socialcost/estimate: the 10 m label grid centred on
base_link, frame base_link, stamped with the input's time, values round(cost * 100 / 254) <= 99
(nav2_mask_overlay_layer maps v -> v * 254 / 100, so cost is reproduced within 2.54).

methods
  zero       an all-zero grid (bodies-only arm through the same path)
  rule       personal space only, from arena_peds positions (clean positions, no interactions)
  oracle     arena-native label from the full arena_peds state
  oracle_ext extended label from arena_peds + interactions
  pos        learned position model on rasterised arena_peds positions/headings (clean positions)
  rgbd       learned RGB-D model on the camera images
  depth/rgb  learned depth-only / RGB-only model
  oracle_ext_replay  the extended oracle label sent through exactly the RGB-D path: triggered by the
             camera frames, painted in the robot-centred label grid at the camera stamp, FOV-masked
             and resampled like a prediction (integration control, privileged)

diagnostics (never change what a method publishes unless --post is given)
  --post spec.json     post-process the robot-frame prediction (socialcost.postprocess; sensor data only)
  --shadow             also compute oracle_ext_replay for learned methods and log the agreement (not published)
  --dump-dir D         save every --dump-every-th tick's intermediate grids to D/tick_<seq>.npz

    python -m socialcost.estimator_node --method rgbd --checkpoint <best.pt> [--log out.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from arena_people_msgs.msg import Pedestrians
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from .labels import GridSpec, LayerParams, Ped, paint, yaw_from_quat

ENV = "/arena/env_0"
NS = f"{ENV}/task_generator_node/turtlebot"
BASE = "env_0/turtlebot/base_link"
CAM = "env_0/turtlebot/oakd_rgb_camera_optical_frame"


def to_occupancy(cost: np.ndarray) -> np.ndarray:
    return np.minimum(np.round(cost.astype(np.float32) * 100.0 / 254.0), 99).astype(np.int8)


class Estimator(Node):
    def __init__(self, method: str, checkpoint: str | None, log: str | None, rate: float, frame: str | None = None,
                 post: str | None = None, shadow: bool = False, dump_dir: str | None = None, dump_every: int = 10) -> None:
        super().__init__("socialcost_estimator", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.method = method
        from . import postprocess
        self.pp = postprocess
        self.post = postprocess.load_spec(post)
        self.shadow = shadow and method in ("pos", "rgbd", "depth", "rgb")
        self.dump_dir, self.dump_every, self.seq = dump_dir, max(1, dump_every), 0
        if dump_dir:
            import os
            os.makedirs(dump_dir, exist_ok=True)
        self.params = LayerParams.from_overlay()
        self.grid = GridSpec.centered(10.0, 0.05)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, f"{ENV}/socialcost/estimate", latched)
        self.peds = None
        self.rgb = self.depth = None
        # recent camera frames, paired by stamp in tick() (exact match preferred, else nearest within PAIR_TOL_NS):
        # requiring the two latest messages to share a stamp can fail for a whole launch (render products one
        # frame apart / arrival order vs the timer) and then nothing is ever published
        from collections import deque
        self.rgb_buf, self.depth_buf = deque(maxlen=12), deque(maxlen=12)
        self.last_pair_stamp = None
        self.pair_dt_ns = 0
        self.create_subscription(Pedestrians, f"{ENV}/arena_peds", lambda m: setattr(self, "peds", m), qos_profile_sensor_data)
        self.model = None
        if method in ("pos", "rgbd", "depth", "rgb"):
            import torch

            from .learn.models import build
            ck = torch.load(checkpoint, map_location="cuda")
            self.inputs = ck["config"]["inputs"]
            self.model = build(self.inputs).cuda().eval()
            self.model.load_state_dict(ck["model"])
            self.torch = torch
        self.K = None
        self.fov = None
        if method in ("pos", "rgbd", "depth", "rgb", "oracle_ext_replay"):
            # learned models are trained on the camera FOV only; their output elsewhere is untrained
            from sensor_msgs.msg import CameraInfo
            self.create_subscription(CameraInfo, f"{NS}/oakd/camera_info", self._info, qos_profile_sensor_data)
        if method in ("rgbd", "depth", "rgb", "oracle_ext_replay"):
            self.create_subscription(Image, f"{NS}/oakd/image", lambda m: self.rgb_buf.append(m), qos_profile_sensor_data)
            self.create_subscription(Image, f"{NS}/oakd/depth_image", lambda m: self.depth_buf.append(m), qos_profile_sensor_data)
        self.frames = None
        if method in ("oracle_ext", "oracle_ext_replay") or self.shadow:
            from arena_humansim_msgs.msg import InteractionFrame
            self.create_subscription(InteractionFrame, f"{ENV}/task_generator_node/interactions", self._frames, latched)
        self.log = open(log, "a") if log else None
        self.frame = frame or self._costmap_param("global_frame")
        # the mask must share the costmap's cell grid: nav2_mask_overlay_layer samples the mask at the
        # costmap's cell centres, and a 0.05 m mask on the 0.10 m local costmap is read 2.5 cm off-centre
        self.out_res = float(self._costmap_param("resolution"))
        self.get_logger().info(f"publishing in {self.frame} at {self.out_res} m")
        self.create_timer(1.0 / rate, self.tick)

    @staticmethod
    def _costmap_param(name: str) -> str:
        """A local costmap parameter. The mask is published in its global frame (applied without a TF
        lookup: nav2_mask_overlay_layer transforms other frames with the latest TF, which misplaces a
        robot-centred mask while the robot moves) and at its resolution."""
        import subprocess
        for _ in range(60):
            r = subprocess.run(["ros2", "param", "get", f"{NS}/local_costmap/local_costmap", name], capture_output=True, text=True)
            if r.returncode == 0 and " value is:" in r.stdout:
                return r.stdout.split(" value is:")[1].strip()
            time.sleep(2.0)
        raise RuntimeError(f"local costmap {name} not available")

    def _tf(self, target: str, source: str, stamp) -> np.ndarray | None:
        try:
            t = self.tf.lookup_transform(target, source, stamp, rclpy.duration.Duration(seconds=0.3))
        except Exception:  # noqa: BLE001
            return None
        q = t.transform.rotation
        return np.array([t.transform.translation.x, t.transform.translation.y, yaw_from_quat(q.x, q.y, q.z, q.w)])

    def _info(self, m) -> None:
        self.K = np.array(m.k).reshape(3, 3)

    def _fov_mask(self) -> np.ndarray | None:
        """Label cells inside the camera FOV (as in training); None until calibration and TF are known."""
        if self.fov is None and self.K is not None:
            from .learn.data import fov_mask
            from .qa import pose_matrix
            try:
                tb = self.tf.lookup_transform(BASE, CAM, Time())
            except Exception:  # noqa: BLE001
                return None
            tr, q = tb.transform.translation, tb.transform.rotation
            self.fov = fov_mask(self.grid, self.K, pose_matrix({"x": tr.x, "y": tr.y, "z": tr.z, "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w}))
        return self.fov

    def _frames(self, msg) -> None:
        offs = list(msg.member_offset) + [len(msg.member_id)]
        self.frames = {"interactions": [{"interaction_id": int(i), "interaction_type": int(msg.interaction_type[k]), "outcome": int(msg.outcome[k]),
                                         "member_ids": [int(x) for x in msg.member_id[offs[k]:offs[k + 1]]]} for k, i in enumerate(msg.interaction_id)]}

    def _pose(self, stamp) -> dict | None:
        try:
            t = self.tf.lookup_transform("map", BASE, stamp, rclpy.duration.Duration(seconds=0.2))
        except Exception:  # noqa: BLE001
            return None
        q = t.transform.rotation
        return {"x": t.transform.translation.x, "y": t.transform.translation.y, "yaw": yaw_from_quat(q.x, q.y, q.z, q.w)}

    def _states(self) -> list[dict]:
        from rosidl_runtime_py.convert import message_to_ordereddict
        out = []
        for p in self.peds.pedestrians:
            q = p.pose.orientation
            out.append({"name": p.name, "id": int(p.id), "x": p.pose.position.x, "y": p.pose.position.y, "yaw": yaw_from_quat(q.x, q.y, q.z, q.w),
                        "animation_state": int(p.animation_state), "interaction_id": int(p.interaction_id), "interaction_type": int(p.interaction_type),
                        "agent_type": p.agent_type, "model_uri": p.model_uri, "gestures": [json.loads(json.dumps(message_to_ordereddict(g))) for g in p.gestures]})
        return out

    def _target_grid(self, pose_f: np.ndarray) -> GridSpec:
        """Frame-aligned 10 m grid around the robot at the costmap resolution, origin on the costmap's cell lattice."""
        r = self.out_res
        size = self.grid.width * self.grid.resolution
        n = int(round(size / r))
        return GridSpec(resolution=r, width=n, height=n,
                        origin_x=round((pose_f[0] - size / 2) / r) * r, origin_y=round((pose_f[1] - size / 2) / r) * r, frame=self.frame)

    def _resample(self, cost_robot: np.ndarray, pose_f: np.ndarray, target: GridSpec) -> np.ndarray:
        """Robot-frame prediction (label grid) -> frame-aligned grid, nearest cell."""
        i, j = np.meshgrid(np.arange(target.width), np.arange(target.height))
        wx = target.origin_x + (i + 0.5) * target.resolution - pose_f[0]
        wy = target.origin_y + (j + 0.5) * target.resolution - pose_f[1]
        c, s_ = math.cos(pose_f[2]), math.sin(pose_f[2])
        lx, ly = c * wx + s_ * wy, -s_ * wx + c * wy
        g = self.grid
        ii = np.floor((lx - g.origin_x) / g.resolution).astype(int)
        jj = np.floor((ly - g.origin_y) / g.resolution).astype(int)
        ok = (ii >= 0) & (ii < g.width) & (jj >= 0) & (jj < g.height)
        out = np.zeros((target.height, target.width), np.uint8)
        out[ok] = cost_robot[jj[ok], ii[ok]]
        return out

    def _states_in(self, T: np.ndarray) -> list[dict]:
        """Pedestrian states moved from map into the mask frame (positions, yaw, gesture targets)."""
        c, s_ = math.cos(T[2]), math.sin(T[2])
        out = []
        for p in self._states():
            q = dict(p, x=T[0] + c * p["x"] - s_ * p["y"], y=T[1] + s_ * p["x"] + c * p["y"], yaw=p["yaw"] + T[2])
            q["gestures"] = [dict(g, at={**g.get("at", {}), "x": T[0] + c * g.get("at", {}).get("x", 0.0) - s_ * g.get("at", {}).get("y", 0.0),
                                          "y": T[1] + s_ * g.get("at", {}).get("x", 0.0) + c * g.get("at", {}).get("y", 0.0)}) for g in p["gestures"]]
            out.append(q)
        return out

    PAIR_TOL_NS = 40_000_000

    def _pair(self) -> bool:
        """Newest (rgb, depth) pair: same stamp if available, else the nearest depth within PAIR_TOL_NS. False if
        no pair, or the newest pair was already used."""
        def ns(m):
            return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        depths = list(self.depth_buf)
        if not depths:
            return False
        dstamps = [ns(d) for d in depths]
        for r in reversed(list(self.rgb_buf)):
            t = ns(r)
            k = min(range(len(depths)), key=lambda i: abs(dstamps[i] - t))
            if abs(dstamps[k] - t) <= self.PAIR_TOL_NS:
                if self.last_pair_stamp == t:
                    return False
                self.rgb, self.depth, self.pair_dt_ns, self.last_pair_stamp = r, depths[k], dstamps[k] - t, t
                return True
        return False

    def _oracle_robot(self, pose: dict) -> np.ndarray:
        """Extended oracle label in the robot-centred label grid at `pose` (map frame), as the dataset labels are painted."""
        from .dataset import EXTENDED_CONFIG
        from .extensions import ExtParams, paint_extended
        if not hasattr(self, "_ext"):
            self._ext = ExtParams.load(EXTENDED_CONFIG)
        cost, _ = paint_extended(self.params, self._ext, self._states(), self.frames, self.grid, (pose["x"], pose["y"], pose["yaw"]))
        return cost.astype(np.uint8)

    def tick(self) -> None:
        t0 = time.monotonic()
        if self.method in ("rgbd", "depth", "rgb", "oracle_ext_replay"):
            if self.K is None or not self._pair():
                return
            stamp = self.rgb.header.stamp
        elif self.method == "zero":
            stamp = self.get_clock().now().to_msg()
        else:
            if self.peds is None:
                return
            stamp = self.peds.header.stamp
        # robot pose in the mask frame at the input's time: the mask is placed where the input was taken
        pose_f = self._tf(self.frame, BASE, Time.from_msg(stamp))
        if pose_f is None:
            return
        target = self._target_grid(pose_f)
        dbg: dict = {}
        if self.method == "zero":
            cost = np.zeros((target.height, target.width), np.uint8)
        elif self.method in ("rule", "oracle", "oracle_ext"):
            T_fm = self._tf(self.frame, "map", Time())
            if T_fm is None:
                return
            states = self._states_in(T_fm)
            if self.method == "rule":
                cost = paint(self.params, [Ped(p["x"], p["y"], p["animation_state"]) for p in states], target)
            elif self.method == "oracle":
                cost = paint(self.params, [Ped(p["x"], p["y"], p["animation_state"], p["interaction_id"], p["interaction_type"]) for p in states], target)
            else:
                from .dataset import EXTENDED_CONFIG
                from .extensions import ExtParams, paint_extended
                cost, _ = paint_extended(self.params, ExtParams.load(EXTENDED_CONFIG), states, self.frames, target, (0.0, 0.0, 0.0))
        elif self.method == "pos":
            from .learn.data import raster_peds
            pose = self._pose(Time.from_msg(stamp))
            if pose is None:
                return
            x = {"raster": self.torch.from_numpy(raster_peds(self._states(), pose, self.grid)).unsqueeze(0).cuda()}
            with self.torch.no_grad():
                robot = (self.model(x)[0].cpu().numpy() * self.params.max_cost_value).round().clip(0, 252).astype(np.uint8)
            fov = self._fov_mask()
            if fov is None:
                return
            dbg["robot_raw"] = robot
            if self.post:
                robot = self.pp.apply(robot, self.post)
                dbg["robot_post"] = robot
            cost = self._resample(np.where(fov, robot, 0).astype(np.uint8), pose_f, target)
        elif self.method == "oracle_ext_replay":
            pose = self._pose(Time.from_msg(stamp))
            fov = self._fov_mask()
            if pose is None or fov is None or self.peds is None:
                return
            robot = self._oracle_robot(pose)
            dbg["robot_raw"] = robot
            cost = self._resample(np.where(fov, robot, 0).astype(np.uint8), pose_f, target)
        else:
            from .learn.data import lift_indices
            from .qa import pose_matrix
            try:
                tb = self.tf.lookup_transform(BASE, CAM, Time())
            except Exception:  # noqa: BLE001
                return
            tr, q = tb.transform.translation, tb.transform.rotation
            T_bc = pose_matrix({"x": tr.x, "y": tr.y, "z": tr.z, "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w})
            rgb = np.frombuffer(bytes(self.rgb.data), np.uint8).reshape(self.rgb.height, self.rgb.width, 3).astype(np.float32) / 255.0
            depth = np.frombuffer(bytes(self.depth.data), np.float32).reshape(self.depth.height, self.depth.width).copy()
            depth[~np.isfinite(depth)] = np.nan
            valid = np.isfinite(depth).astype(np.float32)
            d = np.nan_to_num(depth, nan=0.0) / 10.0
            chans = []
            if self.inputs in ("rgbd", "rgb"):
                chans.append(rgb.transpose(2, 0, 1) - 0.5)
            if self.inputs in ("rgbd", "depth"):
                chans += [d[None], valid[None]]
            x = {"image": self.torch.from_numpy(np.concatenate(chans, 0)).unsqueeze(0).cuda(),
                 "lift": self.torch.from_numpy(lift_indices(depth, self.K, T_bc, mode="ground" if self.inputs == "rgb" else "depth")).unsqueeze(0).cuda()}
            with self.torch.no_grad():
                robot = (self.model(x)[0].cpu().numpy() * self.params.max_cost_value).round().clip(0, 252).astype(np.uint8)
            fov = self._fov_mask()
            if fov is None:
                return
            dbg["robot_raw"] = robot
            if self.post:
                vis = self.pp.visibility_mask(depth, self.K, T_bc, self.grid, self.post.get("margin_m", 0.6)) if self.post.get("visibility") else None
                robot = self.pp.apply(robot, self.post, vis)
                dbg["robot_post"] = robot
            cost = self._resample(np.where(fov, robot, 0).astype(np.uint8), pose_f, target)
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame
        msg.info.resolution = target.resolution
        msg.info.width, msg.info.height = target.width, target.height
        msg.info.origin.position.x, msg.info.origin.position.y = target.origin_x, target.origin_y
        msg.info.origin.orientation.w = 1.0
        occ = to_occupancy(cost)
        msg.data = occ.ravel().tolist()
        self.pub.publish(msg)
        compute_ms = (time.monotonic() - t0) * 1000
        self.seq += 1
        if self.log:
            now = self.get_clock().now().nanoseconds
            rec = {"t_ns": now, "input_stamp_ns": stamp.sec * 1_000_000_000 + stamp.nanosec, "compute_ms": compute_ms,
                   "nonzero": int((cost > 0).sum()), "seq": self.seq, "frame": self.frame, "res": target.resolution, "w": target.width, "h": target.height,
                   "origin": [target.origin_x, target.origin_y], "pose_f": [float(v) for v in pose_f], "cost_max": int(cost.max()),
                   "occ_max": int(occ.max()), "occ_clipped": int((occ >= 99).sum()),
                   "gt10": int((cost > 10).sum()), "gt50": int((cost > 50).sum()), "gt100": int((cost > 100).sum())}
            if self.method in ("rgbd", "depth", "rgb", "oracle_ext_replay"):
                rec["pair_dt_ms"] = self.pair_dt_ns / 1e6
            if "robot_raw" in dbg:
                rr = dbg["robot_raw"]
                rec["raw_fov_nonzero"] = int(((rr > 0) & self.fov).sum())
                rec["raw_fov_mean"] = float(rr[self.fov].mean())
            if self.shadow and "robot_raw" in dbg and self.peds is not None:
                pose = self._pose(Time.from_msg(stamp))
                if pose is not None:
                    sh = self._oracle_robot(pose)
                    dbg["shadow_robot"] = sh
                    pr = dbg.get("robot_post", dbg["robot_raw"]).astype(np.int16)
                    f, t_ = self.fov, sh.astype(np.int16)
                    a, b = pr[f], t_[f]
                    rec["shadow"] = {"mae_fov": float(np.abs(a - b).mean()), "label_nonzero_fov": int((b > 0).sum()), "pred_nonzero_fov": int((a > 0).sum()),
                                     **{f"tp{th}": int(((a >= th) & (b >= th)).sum()) for th in (10, 50, 100)},
                                     **{f"fp{th}": int(((a >= th) & (b < th)).sum()) for th in (10, 50, 100)},
                                     **{f"fn{th}": int(((a < th) & (b >= th)).sum()) for th in (10, 50, 100)}}
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        if self.dump_dir and self.seq % self.dump_every == 0:
            import os
            np.savez_compressed(os.path.join(self.dump_dir, f"tick_{self.seq:06d}.npz"), cost=cost, occ=occ, fov=self.fov if self.fov is not None else np.zeros(1),
                                stamp_ns=stamp.sec * 1_000_000_000 + stamp.nanosec, pose_f=pose_f, origin=np.array([target.origin_x, target.origin_y]),
                                res=target.resolution, frame=self.frame, **dbg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["zero", "rule", "oracle", "oracle_ext", "pos", "rgbd", "depth", "rgb", "oracle_ext_replay"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--frame", default=None, help="mask frame; default: the local costmap global frame")
    ap.add_argument("--post", default=None, help="post-processing spec (json) for learned methods")
    ap.add_argument("--shadow", action="store_true", help="log agreement with oracle_ext_replay (not published)")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--dump-every", type=int, default=10)
    args = ap.parse_args()
    rclpy.init()
    n = Estimator(args.method, args.checkpoint, args.log, args.rate, args.frame, args.post, args.shadow, args.dump_dir, args.dump_every)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
