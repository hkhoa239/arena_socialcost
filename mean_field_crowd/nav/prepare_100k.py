"""Configs for the controlled 100k AGENT_5 runs.

Source is agent_5_nav_progress.yaml (approach_goal on the final goal). Changes from it:
name, starting_stage, budget, device, checkpoint ticks, and a pinned stage (no curriculum
promotion or demotion, no periodic eval inside the budget). PPO, network, reward
coefficients, and seeds are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

SOURCE = Path("/opt/arena_ws/src/Arena/arena_training/configs/agent_5_nav_progress.yaml")
OUT = Path("/opt/arena_ws/data/mf_nav/configs")
BUDGET = 100_000
TICKS = [20_000, 40_000, 60_000, 80_000]
JOBS = [("AGENT_5_NAV100K_EMPTY", 0), ("AGENT_5_NAV100K_STATIC", 1)]


def build(name: str, stage: int, device: str) -> dict:
    cfg = yaml.safe_load(SOURCE.read_text())
    cfg["resume"] = False
    cfg["agent_config"]["name"] = name
    params = cfg["agent_config"]["framework"]["algorithm"]["parameters"]
    params["total_timesteps"] = BUDGET
    params["device"] = device
    general = cfg["arena_cfg"]["general"]
    general["n_timesteps"] = BUDGET
    staged = cfg["arena_cfg"]["task"]["staged"]
    staged["starting_stage"] = stage
    staged["upper_threshold"] = 1.01
    staged["lower_threshold"] = 0.0
    callbacks = cfg["arena_cfg"]["callbacks"]
    callbacks["periodic_evaluation"]["eval_freq"] = 10_000_000
    callbacks["checkpoint_ticks"] = TICKS
    return cfg


def main() -> None:
    device = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, stage in JOBS:
        dest = OUT / f"{name}.yaml"
        dest.write_text(yaml.safe_dump(build(name, stage, device), sort_keys=False))
        print(dest)


if __name__ == "__main__":
    main()
