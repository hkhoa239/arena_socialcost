"""Matched rollout with a per-step trace.

Same stages, seeds, and launch arguments as eval_one.py. The env runs in-process
(debug_mode, GatheredDummyVecEnv) so every step's obs_dict and reward breakdown can be
recorded: observed goal vs the goal in the task generator's episode record, distance to
the final goal, commanded and measured velocity, minimum laser range, reward terms, robot
heading, laser hits in the world frame (every 4th beam), and pedestrian positions.

usage: trace_eval.py CONFIG CHECKPOINT STAGE EPISODES RUN_SEED DEST_JSON
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy

sys.path.insert(0, "/opt/arena_ws/data/mf_eval")
sys.path.insert(0, "/opt/arena_ws/data/mf_nav")
from eval_one import STAGES, Trace, _info, expected_seed  # noqa: E402
from matched_eval import COMMON, _reason, _spin  # noqa: E402

sys.path.insert(0, "/opt/arena_ws/data/mf_bench")
from stages import STAGES as BENCH_STAGES  # noqa: E402

STAGES = {**STAGES, **BENCH_STAGES}

from arena_training.arena_rosnav_rl.cfg import RobotCfg
from arena_training.arena_rosnav_rl.trainer import get_trainer
from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import spawn_envs, wait_for_simulation


BEAM_STRIDE = 4  # laser beams kept per step for the world-frame scan (render_episode.py draws them)


class PedTrace(Trace):
    """Trace that also keeps the latest pedestrian states and active collision events, sampled into every step."""

    def __init__(self) -> None:
        super().__init__()
        self.latest_peds: list[list[float]] = []
        self.latest_ped_vel: list[list[float]] = []
        self.latest_ped_names: list[str] = []
        self.latest_events: list[list[str]] = []
        # Every contact seen since the last reset_episode(); the vec env auto-resets on done, so the
        # contact that ended an episode can be gone from latest_events by the time the step is recorded.
        self.events_seen: set[tuple[str, str]] = set()

    def _on_peds(self, msg) -> None:
        super()._on_peds(msg)
        people = sorted(msg.pedestrians, key=lambda p: p.name)
        self.latest_peds = [[round(float(p.pose.position.x), 3), round(float(p.pose.position.y), 3)] for p in people]
        self.latest_ped_vel = [[round(float(p.twist.linear.x), 3), round(float(p.twist.linear.y), 3)] for p in people]
        self.latest_ped_names = [str(p.name) for p in people]

    def _on_events(self, msg) -> None:
        super()._on_events(msg)
        self.latest_events = [[str(event.kind), str(event.obstacle_id)] for event in msg.events]
        self.events_seen.update((str(event.kind), str(event.obstacle_id)) for event in msg.events)

    def reset_episode(self) -> None:
        super().reset_episode()
        self.latest_peds = []
        self.latest_ped_vel = []
        self.latest_ped_names = []
        self.latest_events = []
        self.events_seen = set()


def _base_env(vec):
    while not hasattr(vec, "envs"):
        vec = vec.venv
    return vec.envs[0].unwrapped


def _record_goal(record, robot: str) -> tuple[list | None, list | None, float | None]:
    """(abstract goto, realized map pose, tolerance) of the robot's first goto phase."""
    if record is None:
        return None, None, None
    entry = json.loads(record.phases or "{}").get(robot, {})
    phases = entry.get("phases", [])
    poses = entry.get("map_poses", [])
    for index, phase in enumerate(phases):
        if "goto" in phase:
            realized = poses[index] if index < len(poses) else None
            return phase["goto"], realized, phase.get("tolerance_radius")
    return None, None, None


def _xy(value) -> list[float] | None:
    if value is None:
        return None
    if getattr(value, "dtype", None) is not None and value.dtype.names:
        return [float(value["x"]), float(value["y"])]
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(flat[0]), float(flat[1])]


class StepRecorder:
    """Wraps the env's observation manager and records what the reward saw."""

    def __init__(self, env, trace: PedTrace | None = None) -> None:
        self.env = env
        self.trace = trace
        self.last: dict | None = None
        manager = env.observation_collector
        original = manager.get_observations

        def capture(**kwargs):
            obs = original(**kwargs)
            # reset() passes is_terminal, step() does not. The vec env auto-resets on
            # done, so a reset observation must not overwrite the terminal step's.
            if "is_terminal" in kwargs:
                self.reset_obs = obs
            else:
                self.last = obs
            return obs

        self.reset_obs: dict | None = None

        manager.get_observations = capture

    def snapshot(self, obs: dict | None = None) -> dict:
        obs = (self.last if obs is None else obs) or {}
        laser = obs.get("front_laser")
        twist = obs.get("robot_twist")
        action = obs.get("last_action")
        dist_goal = obs.get("dist_angle_to_goal")
        dist_sub = obs.get("dist_angle_to_subgoal")
        pose = obs.get("robot_pose")
        yaw = scan_xy = None
        if pose is not None:
            yaw = float(pose["yaw"])
            if laser is not None and len(laser):
                ranges = np.asarray(laser, dtype=np.float64)
                angles = -math.pi + np.arange(len(ranges)) * (2 * math.pi / len(ranges))
                keep = np.arange(0, len(ranges), BEAM_STRIDE)
                keep = keep[ranges[keep] < 11.9]
                x = float(pose["x"]) + ranges[keep] * np.cos(yaw + angles[keep])
                y = float(pose["y"]) + ranges[keep] * np.sin(yaw + angles[keep])
                scan_xy = np.round(np.stack([x, y], 1), 2).tolist()
        return {
            "yaw": yaw,
            "scan_xy": scan_xy,
            "peds": list(self.trace.latest_peds) if self.trace is not None else [],
            "ped_vel": list(self.trace.latest_ped_vel) if self.trace is not None else [],
            "ped_names": list(self.trace.latest_ped_names) if self.trace is not None else [],
            "events": list(self.trace.latest_events) if self.trace is not None else [],
            "collision_monitor": bool(np.any(obs.get("collision_monitor"))) if obs.get("collision_monitor") is not None else None,
            "robot": _xy(obs.get("robot_pose")),
            "goal_obs": _xy(obs.get("goal_pose")),
            "subgoal": _xy(obs.get("subgoal_pose")),
            "d_goal_obs": float(dist_goal[0]) if dist_goal is not None else None,
            "d_subgoal": float(dist_sub[0]) if dist_sub is not None else None,
            "min_laser": float(np.min(laser)) if laser is not None and len(laser) else None,
            "twist": [float(x) for x in np.asarray(twist, dtype=np.float64).reshape(-1)[:3]] if twist is not None else None,
            "cmd": [float(x) for x in np.asarray(action, dtype=np.float64).reshape(-1)[:3]] if action is not None else None,
            "terms": {k: float(v) for k, v in self.env._reward_function.state.reward_overview.items()},
            "goal_radius": float(self.env.agent_parameters.goal_radius),
        }


def rollout(config_path: str, checkpoint: str, stage: str, episodes: int, run_seed: str) -> dict:
    spec = STAGES[stage]
    config = load_training_config(config_path)
    config.arena_cfg.robot = RobotCfg(robot_model="jackal")
    config.arena_cfg.general.n_envs = 1
    config.arena_cfg.general.debug_mode = True
    config.arena_cfg.general.max_num_moves_per_eps = spec["steps"]
    config.arena_cfg.general.env_launch_args = [[*COMMON, f"run_seed:={run_seed}", *spec["args"]]]
    config.arena_cfg.task.tm_modules = "none"
    config.agent_config.name = f"NAV_TRACE_{Path(checkpoint).parent.name}_{stage}"
    config.resume = False
    wait_for_simulation(timeout=180.0)
    launch = [[
        *config.arena_cfg.general.env_launch_args[0],
        config.arena_cfg.robot.launch_arg,
        f"task.episode.spawn.robot_clearance:={config.arena_cfg.general.safety_distance}",
    ]]
    env_map = spawn_envs(1, launch)
    trace = PedTrace()
    trace.listen(env_map[0])
    stop = threading.Event()
    thread = threading.Thread(target=_spin, args=(trace, stop), daemon=True)
    thread.start()
    trainer = get_trainer(config, namespace_fn=lambda index, mapping=env_map: mapping[index])
    try:
        fresh = trainer.agent.model._model
        model = None
        for device in ("cuda", "cpu"):
            try:
                model = type(fresh).load(checkpoint, env=fresh.get_env(), device=device)
                print(f"LOADED {checkpoint} on {device}", flush=True)
                break
            except Exception as exc:
                print(f"LOAD_FAIL {device} {exc}", flush=True)
        if model is None:
            raise RuntimeError(f"could not load {checkpoint}")
        env = model.get_env()
        base = _base_env(env)
        recorder = StepRecorder(base, trace)
        env.step(np.zeros((env.num_envs,) + env.action_space.shape))
        previous = trace.episode_id
        obs = env.reset()
        rows = []
        for index in range(episodes):
            episode_id, episode_seed = trace.wait_episode(previous)
            world = trace.world
            time.sleep(0.2)
            record = base._latest_episode
            goto, realized, tolerance = _record_goal(record, base._robot_name)
            first = recorder.snapshot(recorder.reset_obs)
            done = False
            steps = 0
            reason, success = "running", False
            terms = {}
            series = []
            while not done and steps < spec["steps"]:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                steps += 1
                snap = recorder.snapshot()
                snap["action"] = [float(x) for x in np.asarray(action, dtype=np.float64).reshape(-1)]
                if realized is not None and snap["robot"] is not None:
                    snap["d_goal_true"] = math.hypot(snap["robot"][0] - realized[0], snap["robot"][1] - realized[1])
                series.append(snap)
                done = bool(np.asarray(dones).reshape(-1)[0])
                if done:
                    info = _info(infos)
                    reason, success = _reason(info)
                    terms = {key: float(value) for key, value in (info.get("reward_terms") or {}).items()}
            if not done:
                raise RuntimeError(f"{stage} episode {episode_id} did not terminate")
            expected = expected_seed(run_seed, world, episode_id) if world else None
            previous = episode_id
            goal_mismatch_steps = sum(
                1 for s in series
                if realized is not None and s["goal_obs"] is not None and math.hypot(s["goal_obs"][0] - realized[0], s["goal_obs"][1] - realized[1]) > 0.05
            )
            row = {
                "index": index,
                "episode_id": episode_id,
                "episode_seed": episode_seed,
                "seed_matches_formula": expected == episode_seed if expected is not None else None,
                "world": world,
                "record_goto": goto,
                "record_goal_map": realized,
                "record_tolerance": tolerance,
                "first_obs": first,
                "goal_mismatch_steps": goal_mismatch_steps,
                "steps": steps,
                "reason": reason,
                "success": success,
                "goal_bonus": terms.get("RewardGoalReached", 0.0) > 0,
                "contacts": int(trace.contacts),
                "contact_events": sorted([list(e) for e in trace.events_seen]),
                "terms": terms,
                "return": float(sum(terms.values())),
                "series": series,
            }
            rows.append(row)
            trace.reset_episode()
            last = series[-1] if series else {}
            print(
                f"TRACE {Path(checkpoint).parent.name} {stage} id={episode_id} seed_ok={row['seed_matches_formula']} "
                f"reason={reason} success={success} bonus={row['goal_bonus']} steps={steps} "
                f"goal_rec={realized} goal_obs_first={first.get('goal_obs')} mismatch_steps={goal_mismatch_steps} "
                f"d_true_end={last.get('d_goal_true')} d_obs_end={last.get('d_goal_obs')} return={row['return']:.3f}",
                flush=True,
            )
        return {"checkpoint": checkpoint, "config": config_path, "stage": stage, "run_seed": run_seed,
                "forecast_mode": os.environ.get("MF_FORECAST_MODE", ""), "episodes": rows}
    finally:
        stop.set()
        trainer.close()


def main() -> int:
    if not rclpy.ok():
        rclpy.init()
    config_path, checkpoint, stage, episodes, run_seed, dest = sys.argv[1:]
    result = rollout(config_path, checkpoint, stage, int(episodes), run_seed)
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
