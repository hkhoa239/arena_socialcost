"""Smoke-test configs: the benchmark configs with a 1,536-tick budget and one checkpoint at 1,024."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from prepare_configs import build  # noqa: E402

OUT = Path("/opt/arena_ws/data/mf_bench/smoke")
for arch, variant in [("AGENT_MF_RESIDUAL", "FWD"), ("AGENT_MF_ONLINE", "FWD"), ("AGENT_5", "FREE")]:
    cfg = build(arch, variant, 5, sys.argv[1], budget=1536, ticks=[1024])
    cfg["agent_config"]["name"] += "_SMOKE"
    dest = OUT / f"{cfg['agent_config']['name']}.yaml"
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(dest)
