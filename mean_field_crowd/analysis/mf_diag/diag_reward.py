"""Roll out one checkpoint and record the reward terms of each episode.

The four logged pieces are the existing units: approach (progress), the two
safety terms, the goal bonus, and the collision penalty. A proximity term
appears only when that unit is in the config.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy

sys.path.insert(0, "/opt/arena_ws/data/mf_eval")
from matched_eval import COMMON, STAGES, _Trace, _reason, _spin  # noqa: E402

from arena_training.arena_rosnav_rl.cfg import RobotCfg
from arena_training.arena_rosnav_rl.trainer import get_trainer
from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import spawn_envs, wait_for_simulation

OUT = Path("/opt/arena_ws/data/mf_diag")


def _info(infos):
    if isinstance(infos, (list, tuple)):
        return infos[0] if infos else {}
    return infos


def rollout(config_path: str, checkpoint: str, stage: str, episodes: int) -> dict:
    spec = STAGES[stage]
    config = load_training_config(config_path)
    config.arena_cfg.robot = RobotCfg(robot_model="jackal")
    config.arena_cfg.general.n_envs = 1
    config.arena_cfg.general.max_num_moves_per_eps = spec["steps"]
    config.arena_cfg.general.env_launch_args = [[*COMMON, f"run_seed:=diag-{stage}", *spec["args"]]]
    config.arena_cfg.task.tm_modules = "none"
    config.agent_config.name = f"DIAG_ROLLOUT_{Path(checkpoint).parent.name}_{stage}"
    config.resume = False
    wait_for_simulation(timeout=180.0)
    launch = [[*config.arena_cfg.general.env_launch_args[0], config.arena_cfg.robot.launch_arg, f"task.episode.spawn.robot_clearance:={config.arena_cfg.general.safety_distance}"]]
    env_map = spawn_envs(1, launch)
    trace = _Trace()
    trace.listen(env_map[0])
    stop = threading.Event()
    thread = threading.Thread(target=_spin, args=(trace, stop), daemon=True)
    thread.start()
    trainer = get_trainer(config, namespace_fn=lambda index, mapping=env_map: mapping[index])
    try:
        model = trainer.agent.model._model
        if checkpoint != "fresh":
            model = type(model).load(checkpoint, env=model.get_env(), device="cuda")
        env = model.get_env()
        env.step(np.zeros((env.num_envs,) + env.action_space.shape))
        previous = trace.episode_id
        obs = env.reset()
        rows = []
        for index in range(episodes):
            episode_id, episode_seed = trace.wait_episode(previous)
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
            previous = episode_id
            row = {
                "index": index,
                "episode_id": episode_id,
                "episode_seed": episode_seed,
                "steps": steps,
                "reason": reason,
                "success": success,
                "mean_abs_linear": float(np.mean(speeds)) if speeds else 0.0,
                "mean_abs_yaw": float(np.mean(turns)) if turns else 0.0,
                "terms": terms,
                "return": float(sum(terms.values())),
            }
            rows.append(row)
            print(
                f"ROLLOUT {Path(checkpoint).stem} {stage} id={episode_id} steps={steps} reason={reason} "
                f"v={row['mean_abs_linear']:.3f} w={row['mean_abs_yaw']:.3f} return={row['return']:.3f} terms={terms}",
                flush=True,
            )
        return {"checkpoint": checkpoint, "stage": stage, "episodes": rows}
    finally:
        stop.set()
        trainer.close()


def main() -> int:
    if not rclpy.ok():
        rclpy.init()
    config_path, checkpoint, stage, episodes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    result = rollout(config_path, checkpoint, stage, episodes)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"rollout_{Path(checkpoint).parent.name}_{stage}.json"
    dest.write_text(json.dumps(result, indent=2))
    print(f"WROTE {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
