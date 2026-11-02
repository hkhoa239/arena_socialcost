"""Training configs of the mean-field navigation benchmark.

Every config is agent_5_nav_progress.yaml (final-goal approach reward, PPO settings of the
pilots) with only these changes:
  name / architecture_name      AGENT_5, AGENT_MF_ONLINE or AGENT_MF_RESIDUAL
  action_space                  normalized [-1,1]^2 policy box; linear (0, 0.5) forward-only
                                or (-0.5, 0.5) unrestricted control; angular (-1.5, 1.5)
  seed / env_seed / mixture     the training seed (5, 6, 7)
  budget                        100k ticks, checkpoints at 20k/40k/60k/80k (+ last_model)
  curriculum                    stages.TRAIN_STAGES drawn per episode with TRAIN_WEIGHTS
                                (no promotion/demotion, periodic eval past the budget)
  callbacks.action_stats        clip rate, velocity distribution, reverse fraction, GPU, fps

usage: prepare_configs.py DEVICE [SEED ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from stages import TOLERANCE, TRAIN_STAGES, TRAIN_WEIGHTS  # noqa: E402

SOURCE = Path("/opt/arena_ws/src/Arena/arena_training/configs/agent_5_nav_progress.yaml")
OUT = Path("/opt/arena_ws/data/mf_bench/configs")
BUDGET = 100_000
TICKS = [20_000, 40_000, 60_000, 80_000]
ANGULAR = [-1.5, 1.5]
LINEAR = {"FWD": [0.0, 0.5], "FREE": [-0.5, 0.5]}
# (architecture, action variant): the primary three-agent comparison plus the unrestricted AGENT_5 control.
ARMS = [("AGENT_5", "FWD"), ("AGENT_MF_ONLINE", "FWD"), ("AGENT_MF_RESIDUAL", "FWD"), ("AGENT_5", "FREE")]
SHORT = {"AGENT_5": "A5", "AGENT_MF_ONLINE": "ONLINE", "AGENT_MF_RESIDUAL": "RESIDUAL"}


def name_of(arch: str, variant: str, seed: int) -> str:
    return f"BENCH_{SHORT[arch]}_{variant}_S{seed}"


def build(arch: str, variant: str, seed: int, device: str, budget: int = BUDGET, ticks: list[int] = TICKS) -> dict:
    cfg = yaml.safe_load(SOURCE.read_text())
    cfg["resume"] = False
    agent = cfg["agent_config"]
    agent["name"] = name_of(arch, variant, seed)
    agent["action_space"] = {"type": "differential_drive", "linear_range": LINEAR[variant], "angular_range": ANGULAR, "normalized": True}
    algo = agent["framework"]["algorithm"]
    algo["architecture_name"] = arch
    params = algo["parameters"]
    params["total_timesteps"] = budget
    params["seed"] = seed
    params["device"] = device
    general = cfg["arena_cfg"]["general"]
    general["n_timesteps"] = budget
    general["env_seed"] = seed
    staged = cfg["arena_cfg"]["task"]["staged"]
    staged["starting_stage"] = 0
    staged["upper_threshold"] = 1.01
    staged["lower_threshold"] = 0.0
    staged["curriculum_definition"] = [{**stage, "task.episode.goto_pose.tolerance.radius": TOLERANCE} for stage in TRAIN_STAGES]
    staged["mixture_weights"] = TRAIN_WEIGHTS
    staged["mixture_seed"] = seed
    callbacks = cfg["arena_cfg"]["callbacks"]
    callbacks["periodic_evaluation"]["eval_freq"] = 10_000_000
    callbacks["checkpoint_ticks"] = [t for t in ticks if t < budget]
    callbacks["action_stats"] = True
    return cfg


def main() -> None:
    device = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [5, 6, 7]
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for arch, variant in ARMS:
            cfg = build(arch, variant, seed, device)
            dest = OUT / f"{cfg['agent_config']['name']}.yaml"
            dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(dest)


if __name__ == "__main__":
    main()
