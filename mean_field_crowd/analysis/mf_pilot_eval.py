"""Deterministic PedVO evaluation for the 50k pilots.

Same head-on scenario, PedVO, and run seed for every agent. Five episodes,
150 steps maximum. Navigation numbers are a pilot diagnostic.

The residual policy is also queried on the recorded observations with the
residual forecast, the online forecast, and a zero forecast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rclpy
import torch

from arena_training.arena_rosnav_rl.cfg import RobotCfg
from arena_training.arena_rosnav_rl.trainer import get_trainer
from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import spawn_envs, wait_for_simulation

AGENTS = (
    ("AGENT_5", "AGENT_5_PILOT50K", "/opt/arena_ws/src/Arena/arena_training/configs/sb_agent_5_pilot50k.yaml"),
    ("AGENT_MF_ONLINE", "AGENT_MF_ONLINE_PILOT50K", "/opt/arena_ws/src/Arena/arena_training/configs/sb_agent_mf_online_pilot50k.yaml"),
    ("AGENT_MF_RESIDUAL", "AGENT_MF_RESIDUAL_PILOT50K", "/opt/arena_ws/src/Arena/arena_training/configs/sb_agent_mf_residual_pilot50k.yaml"),
)
EPISODES = 5
MAX_STEPS = 150
EVAL_SEED = 50
OUT = Path("/opt/arena_ws/data/mf_pilot_eval.json")


def _config(path: str):
    config = load_training_config(path)
    config.arena_cfg.robot = RobotCfg(robot_model="jackal")
    config.arena_cfg.general.n_envs = 1
    config.arena_cfg.general.max_num_moves_per_eps = MAX_STEPS
    config.arena_cfg.general.env_seed = EVAL_SEED
    config.arena_cfg.general.env_launch_args = [[
        "task.robots:=scenario",
        "task.obstacles:=scenario",
        "task.scenario.file:=mf_headon",
        "human:=arena",
        "humansim.local_planner:=pedvo",
    ]]
    config.arena_cfg.task.tm_modules = "none"
    config.agent_config.name = config.agent_config.name + "_EVAL"
    config.resume = False
    return config


def _reason(info) -> str:
    if isinstance(info, list):
        info = info[0] if info else {}
    reason = info.get("done_reason", "")
    return getattr(reason, "name", str(reason))


def _rollout(model, env) -> tuple[list[dict], list[dict]]:
    episodes = []
    observations = []
    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        steps = 0
        reward = 0.0
        reason = "running"
        while not done and steps < MAX_STEPS:
            if episode == 0 and steps < 30:
                observations.append({key: np.array(value) for key, value in obs.items()})
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            reward += float(np.asarray(rewards).reshape(-1)[0])
            steps += 1
            done = bool(np.asarray(dones).reshape(-1)[0])
            if done:
                reason = _reason(infos[0] if isinstance(infos, (list, tuple)) else infos)
        episodes.append({"episode": episode, "steps": steps, "reward": reward, "reason": reason})
    return episodes, observations


def _extractors(model):
    policy = model.policy
    found = []
    for name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        extractor = getattr(policy, name, None)
        if extractor is not None and extractor not in found and hasattr(extractor, "_apply_correction"):
            found.append(extractor)
    return found


def _action_sensitivity(model, observations) -> dict:
    from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field import residual_extractor

    original = residual_extractor.forecast_density

    def predict(mode: str) -> np.ndarray:
        def forecast(*args, **kwargs):
            if mode == "zero":
                positions = args[1]
                return torch.zeros(positions.shape[0], 16, 16, dtype=positions.dtype, device=positions.device)
            return original(*args, **kwargs)

        residual_extractor.forecast_density = forecast
        for extractor in _extractors(model):
            extractor._apply_correction = mode == "residual"
        actions = []
        for obs in observations:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
        return np.stack(actions)

    try:
        residual = predict("residual")
        online = predict("online")
        zero = predict("zero")
    finally:
        residual_extractor.forecast_density = original
        for extractor in _extractors(model):
            extractor._apply_correction = True
    return {
        "samples": int(len(observations)),
        "mean_abs_residual_vs_online": float(np.abs(residual - online).mean()) if len(observations) else None,
        "mean_abs_residual_vs_zero": float(np.abs(residual - zero).mean()) if len(observations) else None,
        "mean_abs_online_vs_zero": float(np.abs(online - zero).mean()) if len(observations) else None,
        "fraction_residual_differs_from_online": float(np.any(np.abs(residual - online) > 1e-4, axis=1).mean()) if len(observations) else None,
    }


def evaluate(architecture: str, folder: str, config_path: str) -> dict:
    config = _config(config_path)
    wait_for_simulation(timeout=120.0)
    clearance = f"task.episode.spawn.robot_clearance:={config.arena_cfg.general.safety_distance}"
    launch_args = [[
        "robot.train:=true",
        "task.episode.auto_reset:=false",
        config.arena_cfg.robot.launch_arg,
        clearance,
        f"run_seed:={EVAL_SEED}-0",
        *config.arena_cfg.general.env_launch_args[0],
    ]]
    env_map = spawn_envs(1, launch_args)
    trainer = get_trainer(config, namespace_fn=lambda index, mapping=env_map: mapping[index])
    try:
        model = trainer.agent.model._model
        checkpoint = Path(f"/opt/arena_ws/data/agents/{folder}/last_model.zip")
        loaded = type(model).load(str(checkpoint), env=model.get_env(), device="cuda")
        episodes, observations = _rollout(loaded, loaded.get_env())
        result = {"architecture": architecture, "checkpoint": str(checkpoint), "episodes": episodes}
        if architecture == "AGENT_MF_RESIDUAL":
            result["forecast_action_sensitivity"] = _action_sensitivity(loaded, observations)
        return result
    finally:
        trainer.close()


def main() -> int:
    if not rclpy.ok():
        rclpy.init()
    requested = sys.argv[1:] or [item[0] for item in AGENTS]
    results = []
    for architecture, folder, config_path in AGENTS:
        if architecture not in requested:
            continue
        print(f"EVAL {architecture}", flush=True)
        results.append(evaluate(architecture, folder, config_path))
        OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
