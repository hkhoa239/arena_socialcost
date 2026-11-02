"""Matched PedVO evaluation of the 50k pilots.

One process evaluates one agent on one stage. The stage's run_seed and the
episode counter make episode i the same initial condition for every agent.
A reference process sends zero commands on that same seed.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from arena_people_msgs.msg import Pedestrians
from arena_robots_msgs.msg import CollisionEvents
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from task_generator_msgs.msg import EpisodeRecord

from arena_training.arena_rosnav_rl.cfg import RobotCfg
from arena_training.arena_rosnav_rl.trainer import get_trainer
from arena_training.arena_rosnav_rl.utils.config import load_training_config
from arena_training.arena_rosnav_rl.utils.sim_bootstrap import spawn_envs, wait_for_simulation

ROOT = Path("/opt/arena_ws/data/mf_eval")
AGENTS = {
    "AGENT_5": ("AGENT_5_PILOT50K", "/opt/arena_ws/src/Arena/arena_training/configs/sb_agent_5_pilot50k.yaml"),
    "AGENT_MF_ONLINE": ("AGENT_MF_ONLINE_PILOT50K", "/opt/arena_ws/src/Arena/arena_training/configs/sb_agent_mf_online_pilot50k.yaml"),
    "AGENT_MF_RESIDUAL": ("AGENT_MF_RESIDUAL_PILOT50K", "/opt/arena_ws/src/Arena/arena_training/configs/sb_agent_mf_residual_pilot50k.yaml"),
}
STAGES = {
    "peds_5": {"seed": "mf50k-501", "steps": 600, "peds": 5, "args": [
        "task.robots:=random", "task.obstacles:=random", "task.modules:=",
        "task.random.dynamic.n:=[5,5]", "task.random.dynamic.models:=[Common/Human/arenian]",
        "task.random.static.n:=[3,3]", "task.random.static.models:=[Hospital/SM_Chair_01a,Hospital/SM_TrashCan]",
        "task.random.interactive.n:=[0,0]",
    ]},
    "peds_15": {"seed": "mf50k-515", "steps": 600, "peds": 15, "args": [
        "task.robots:=random", "task.obstacles:=random", "task.modules:=",
        "task.random.dynamic.n:=[15,15]", "task.random.dynamic.models:=[Common/Human/arenian]",
        "task.random.static.n:=[3,3]", "task.random.static.models:=[Hospital/SM_Chair_01a,Hospital/SM_TrashCan]",
        "task.random.interactive.n:=[0,0]",
    ]},
    "peds_20": {"seed": "mf50k-520", "steps": 600, "peds": 20, "args": [
        "task.robots:=random", "task.obstacles:=random", "task.modules:=",
        "task.random.dynamic.n:=[20,20]", "task.random.dynamic.models:=[Common/Human/arenian]",
        "task.random.static.n:=[3,3]", "task.random.static.models:=[Hospital/SM_Chair_01a,Hospital/SM_TrashCan]",
        "task.random.interactive.n:=[0,0]",
    ]},
    "headon": {"seed": "mf50k-530", "steps": 450, "peds": 1, "args": [
        "task.robots:=scenario", "task.obstacles:=scenario", "task.modules:=",
        "task.scenario.file:=mf_headon",
    ]},
}
EPISODES = 5
COMMON = ["human:=arena", "humansim.local_planner:=pedvo", "task.episode.auto_reset:=false", "robot.train:=true"]


class _Trace(Node):
    def __init__(self) -> None:
        super().__init__("mf_eval_trace")
        self.clock_s = None
        self.paths: dict[str, list[list[float]]] = {}
        self._active: set[str] = set()
        self.contacts = 0
        self.samples = 0
        self.episode_id: int | None = None
        self.episode_seed: int | None = None
        self.create_subscription(Clock, "/clock", self._on_clock, 10)

    def _on_clock(self, msg: Clock) -> None:
        self.clock_s = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9

    def listen(self, namespace: str) -> None:
        env_name = namespace.split("/")[1] if "/" in namespace else namespace
        deadline = time.time() + 20.0
        names: list[str] = []
        while time.time() < deadline:
            names = [name for name, _ in self.get_topic_names_and_types()]
            if any(name.endswith("arena_peds") and f"/{env_name}/" in name for name in names):
                break
            time.sleep(0.5)
        peds = [name for name in names if name.endswith("arena_peds") and f"/{env_name}/" in name and "mujoco" not in name]
        events = [name for name in names if name.endswith("collision_events") and env_name in name]
        episodes = [name for name in names if name.endswith("state/episode") and f"/{env_name}/" in name]
        print(f"TRACE ns={namespace} peds={peds} events={events} episodes={episodes}", flush=True)
        for topic in peds:
            self.create_subscription(Pedestrians, topic, self._on_peds, 10)
        for topic in events:
            self.create_subscription(CollisionEvents, topic, self._on_events, 10)
        episode_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for topic in episodes:
            self.create_subscription(EpisodeRecord, topic, self._on_episode, episode_qos)

    def _on_peds(self, msg: Pedestrians) -> None:
        self.samples += 1
        if self.samples % 5:
            return
        for ped in msg.pedestrians:
            self.paths.setdefault(ped.name, []).append([float(ped.pose.position.x), float(ped.pose.position.y)])

    def _on_events(self, msg: CollisionEvents) -> None:
        current = {event.obstacle_id for event in msg.events if event.kind == "pedestrian"}
        self.contacts += len(current - self._active)
        self._active = current

    def _on_episode(self, msg: EpisodeRecord) -> None:
        self.episode_id = int(msg.episode_id)
        self.episode_seed = int(msg.seed)

    def wait_episode(self, previous: int | None) -> tuple[int, int]:
        deadline = time.time() + 30.0
        while time.time() < deadline:
            current = self.episode_id
            if current is not None and current != previous:
                return current, int(self.episode_seed or 0)
            time.sleep(0.05)
        raise RuntimeError(f"no episode record after {previous}")

    def reset_episode(self) -> None:
        self.paths = {}
        self._active = set()
        self.contacts = 0
        self.samples = 0


def _config(agent: str, stage: str):
    spec = STAGES[stage]
    folder, path = AGENTS["AGENT_5" if agent == "reference" else agent]
    config = load_training_config(path)
    config.arena_cfg.robot = RobotCfg(robot_model="jackal")
    config.arena_cfg.general.n_envs = 1
    config.arena_cfg.general.max_num_moves_per_eps = spec["steps"]
    config.arena_cfg.general.env_launch_args = [[*COMMON, f"run_seed:={spec['seed']}", *spec["args"]]]
    config.arena_cfg.task.tm_modules = "none"
    config.agent_config.name = f"{folder}_EVAL_{stage}"
    config.resume = False
    return config, folder


def _reason(info) -> tuple[str, bool]:
    if isinstance(info, (list, tuple)):
        info = info[0] if info else {}
    reason = info.get("done_reason", "")
    text = getattr(reason, "name", str(reason))
    return text, bool(info.get("is_success", False)) or "SUCCESS" in text


def _spin(node: Node, stop: threading.Event) -> None:
    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)


def _sensitivity(model, observations: list[dict]) -> dict:
    from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field import residual_extractor

    original = residual_extractor.forecast_density

    def extractors():
        found = []
        for name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
            extractor = getattr(model.policy, name, None)
            if extractor is not None and extractor not in found and hasattr(extractor, "_apply_correction"):
                found.append(extractor)
        return found

    def predict(mode: str) -> np.ndarray:
        def forecast(*args, **kwargs):
            if mode == "zero":
                positions = args[1]
                return torch.zeros(positions.shape[0], 16, 16, dtype=positions.dtype, device=positions.device)
            return original(*args, **kwargs)

        residual_extractor.forecast_density = forecast
        for extractor in extractors():
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
        for extractor in extractors():
            extractor._apply_correction = True
    if not observations:
        return {"samples": 0}
    return {
        "samples": int(len(observations)),
        "mean_abs_residual_vs_online": float(np.abs(residual - online).mean()),
        "mean_abs_residual_vs_zero": float(np.abs(residual - zero).mean()),
        "mean_abs_online_vs_zero": float(np.abs(online - zero).mean()),
        "fraction_residual_differs_from_online": float(np.any(np.abs(residual - online) > 1e-4, axis=1).mean()),
        "fraction_residual_differs_from_zero": float(np.any(np.abs(residual - zero) > 1e-4, axis=1).mean()),
    }


def _frozen(model) -> dict:
    path = Path("/opt/arena_ws/data/mf_pedvo/interaction_residual.pt")
    if not path.is_file():
        return {"checkpoint": str(path), "present": False}
    ref = torch.load(path, map_location="cpu", weights_only=True)
    diffs = {}
    grads = {}
    for name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        extractor = getattr(model.policy, name, None)
        if extractor is None or not hasattr(extractor, "residual"):
            continue
        state = extractor.residual.state_dict()
        diffs[name] = max(float((state[key].detach().cpu() - value).abs().max()) for key, value in ref.items())
        grads[name] = any(param.requires_grad for param in extractor.residual.parameters())
    return {"checkpoint": str(path), "parameters": int(sum(v.numel() for v in ref.values())), "max_abs_diff": diffs, "any_requires_grad": grads}


def _expected_ids(agent: str, stage: str) -> list[int] | None:
    if agent == "reference":
        return None
    path = ROOT / f"reference_{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"missing reference episodes for {stage}")
    payload = json.loads(path.read_text())
    return [int(episode["episode_id"]) for episode in payload["episodes"]]


def evaluate(agent: str, stage: str) -> dict:
    spec = STAGES[stage]
    config, folder = _config(agent, stage)
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
        mode = "reference"
        frozen = None
        if agent != "reference":
            checkpoint = Path(f"/opt/arena_ws/data/agents/{folder}/last_model.zip")
            model = type(model).load(str(checkpoint), env=model.get_env(), device="cuda")
            mode = "policy"
            if agent == "AGENT_MF_RESIDUAL":
                frozen = _frozen(model)
        env = model.get_env()
        expected = _expected_ids(agent, stage)
        zero = np.zeros((env.num_envs,) + env.action_space.shape)
        env.step(zero)
        previous_episode = trace.episode_id
        obs = env.reset()
        episodes = []
        observations = []
        for index in range(EPISODES):
            episode_id, episode_seed = trace.wait_episode(previous_episode)
            time.sleep(0.2)
            trace.reset_episode()
            if index and episode_id != episodes[0]["episode_id"] + index:
                raise RuntimeError(f"{agent} {stage} episode ids {[item['episode_id'] for item in episodes]} then {episode_id}")
            if expected is not None and episode_id != expected[index]:
                raise RuntimeError(f"{agent} {stage} episode {index} id {episode_id} != reference {expected[index]}")
            start = trace.clock_s
            done = False
            steps = 0
            reason, success = "running", False
            while not done and steps < spec["steps"]:
                if agent == "AGENT_MF_RESIDUAL" and len(observations) < 40 and steps % 15 == 0:
                    observations.append({key: np.array(value) for key, value in obs.items()})
                if agent == "reference":
                    action = np.zeros((env.num_envs,) + env.action_space.shape)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                steps += 1
                done = bool(np.asarray(dones).reshape(-1)[0])
                if done:
                    reason, success = _reason(infos[0] if isinstance(infos, (list, tuple)) else infos)
            end = trace.clock_s
            elapsed = None if start is None or end is None or not success else float(end - start)
            if not done:
                raise RuntimeError(f"{agent} {stage} episode {episode_id} did not terminate")
            previous_episode = episode_id
            episodes.append({
                "index": index,
                "episode_id": episode_id,
                "episode_seed": episode_seed,
                "steps": steps,
                "reason": reason,
                "success": success,
                "ped_contacts": int(trace.contacts),
                "zero_contact_success": bool(success and trace.contacts == 0),
                "time_to_goal_s": elapsed,
                "paths": trace.paths,
            })
            print(f"EPISODE {agent} {stage} {index} id={episode_id} seed={episode_seed} reason={reason} success={success} contacts={trace.contacts} steps={steps} peds={len(trace.paths)}", flush=True)
            if steps < 5 and "cancel" in reason.lower():
                raise RuntimeError(f"{agent} {stage} episode {index} was cancelled after {steps} steps")
        result = {"agent": agent, "stage": stage, "seed": spec["seed"], "pedestrians": spec["peds"], "mode": mode, "episodes": episodes}
        if frozen is not None:
            result["residual_frozen"] = frozen
        if observations:
            result["action_sensitivity"] = _sensitivity(model, observations)
        return result
    finally:
        stop.set()
        trainer.close()


def main() -> int:
    if not rclpy.ok():
        rclpy.init()
    agent, stage = sys.argv[1], sys.argv[2]
    print(f"EVAL {agent} {stage}", flush=True)
    result = evaluate(agent, stage)
    mode = os.environ.get("MF_FORECAST_MODE", "")
    result["forecast_mode"] = mode or "native"
    out = ROOT / f"{agent}_{stage}{'_' + mode if mode else ''}.json"
    out.write_text(json.dumps(result))
    print(f"WROTE {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
