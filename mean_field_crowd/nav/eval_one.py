"""One matched rollout. run_seed is forwarded into the task generator at spawn."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy

sys.path.insert(0, "/opt/arena_ws/data/mf_eval")
from matched_eval import COMMON, _Trace, _reason, _spin  # noqa: E402

from arena_training.arena_rosnav_rl.cfg import RobotCfg
from arena_training.arena_rosnav_rl.trainer import get_trainer
from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import spawn_envs, wait_for_simulation

STAGES = {
    "empty": {
        "steps": 600,
        "args": [
            "task.robots:=random",
            "task.obstacles:=random",
            "task.modules:=",
            "task.random.static.n:=[0,0]",
            "task.random.interactive.n:=[0,0]",
            "task.random.dynamic.n:=[0,0]",
            "task.episode.goto_pose.tolerance.radius:=1.5",
        ],
    },
    "static": {
        "steps": 600,
        "args": [
            "task.robots:=random",
            "task.obstacles:=random",
            "task.modules:=",
            "task.random.static.n:=[2,4]",
            "task.random.static.models:=[Hospital/SM_Chair_01a,Hospital/SM_BedSideTable_01b]",
            "task.random.interactive.n:=[0,0]",
            "task.random.dynamic.n:=[0,0]",
            "task.episode.goto_pose.tolerance.radius:=1.2",
        ],
    },
    "ped1": {
        "steps": 600,
        "args": [
            "task.robots:=random",
            "task.obstacles:=random",
            "task.modules:=",
            "task.random.static.n:=[0,0]",
            "task.random.interactive.n:=[0,0]",
            "task.random.dynamic.n:=[1,1]",
            "task.random.dynamic.models:=[Common/Human/arenian]",
            "task.episode.goto_pose.tolerance.radius:=0.8",
        ],
    },
    "headon": {
        "steps": 450,
        "args": [
            "task.robots:=scenario",
            "task.obstacles:=scenario",
            "task.modules:=",
            "task.scenario.file:=mf_headon",
        ],
    },
}


def expected_seed(run_seed: str, world: str, episode_id: int) -> int:
    digest = hashlib.blake2b(f"{run_seed}|{world}|{episode_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


class Trace(_Trace):
    def __init__(self) -> None:
        super().__init__()
        self.world = ""

    def _on_episode(self, msg) -> None:
        super()._on_episode(msg)
        self.world = str(getattr(msg, "world", "") or "")


def _info(infos):
    if isinstance(infos, (list, tuple)):
        return infos[0] if infos else {}
    return infos


def rollout(config_path: str, checkpoint: str, stage: str, episodes: int, run_seed: str) -> dict:
    spec = STAGES[stage]
    config = load_training_config(config_path)
    config.arena_cfg.robot = RobotCfg(robot_model="jackal")
    config.arena_cfg.general.n_envs = 1
    config.arena_cfg.general.max_num_moves_per_eps = spec["steps"]
    config.arena_cfg.general.env_launch_args = [[*COMMON, f"run_seed:={run_seed}", *spec["args"]]]
    config.arena_cfg.task.tm_modules = "none"
    config.agent_config.name = f"NAV_EVAL_{Path(checkpoint).parent.name}_{stage}"
    config.resume = False
    wait_for_simulation(timeout=180.0)
    launch = [[
        *config.arena_cfg.general.env_launch_args[0],
        config.arena_cfg.robot.launch_arg,
        f"task.episode.spawn.robot_clearance:={config.arena_cfg.general.safety_distance}",
    ]]
    env_map = spawn_envs(1, launch)
    trace = Trace()
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
        env.step(np.zeros((env.num_envs,) + env.action_space.shape))
        previous = trace.episode_id
        obs = env.reset()
        rows = []
        for index in range(episodes):
            episode_id, episode_seed = trace.wait_episode(previous)
            world = trace.world
            time.sleep(0.2)
            done = False
            steps = 0
            speeds = []
            turns = []
            reason, success = "running", False
            terms = {}
            while not done and steps < spec["steps"]:
                action, _ = model.predict(obs, deterministic=True)
                flat = np.asarray(action, dtype=np.float64).reshape(-1)
                speeds.append(abs(float(flat[0])))
                if flat.size > 1:
                    turns.append(abs(float(flat[-1])))
                obs, _, dones, infos = env.step(action)
                steps += 1
                done = bool(np.asarray(dones).reshape(-1)[0])
                if done:
                    info = _info(infos)
                    reason, success = _reason(info)
                    terms = {key: float(value) for key, value in (info.get("reward_terms") or {}).items()}
            if not done:
                raise RuntimeError(f"{stage} episode {episode_id} did not terminate")
            expected = expected_seed(run_seed, world, episode_id) if world else None
            previous = episode_id
            row = {
                "index": index,
                "episode_id": episode_id,
                "episode_seed": episode_seed,
                "world": world,
                "expected_seed": expected,
                "seed_matches_formula": expected == episode_seed if expected is not None else None,
                "steps": steps,
                "reason": reason,
                "success": success,
                "contacts": int(trace.contacts),
                "mean_abs_linear": float(np.mean(speeds)) if speeds else 0.0,
                "mean_abs_yaw": float(np.mean(turns)) if turns else 0.0,
                "terms": terms,
                "return": float(sum(terms.values())),
            }
            rows.append(row)
            trace.reset_episode()
            print(
                f"EVAL {Path(checkpoint).parent.name} {stage} id={episode_id} seed={episode_seed} "
                f"world={world!r} formula={expected} reason={reason} success={success} "
                f"v={row['mean_abs_linear']:.3f} steps={steps} return={row['return']:.3f}",
                flush=True,
            )
        return {
            "checkpoint": checkpoint,
            "config": config_path,
            "stage": stage,
            "run_seed": run_seed,
            "episodes": rows,
        }
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
    path.write_text(json.dumps(result, indent=2))
    print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
