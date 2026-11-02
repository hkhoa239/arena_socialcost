"""Write the pilot configs. One reward change: approach_goal follows the final goal."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/opt/arena_ws/src/Arena/arena_training/configs")
OUT = Path("/opt/arena_ws/data/mf_nav/configs")

ORIGINAL = ROOT / "agent_5_diag_empty.yaml"
PROGRESS = ROOT / "agent_5_nav_progress.yaml"

PED_STAGE = """
        # Stage 5: one moving pedestrian, no static obstacles
        - task.random.static.n: [0, 0]
          task.random.interactive.n: [0, 0]
          task.random.dynamic.n: [1, 1]
          task.random.static.models: []
          task.random.interactive.models: []
          task.random.dynamic.models: [Common/Human/arenian]
          task.episode.goto_pose.tolerance.radius: 0.8
"""

PROXIMITY = """
      pedestrian_proximity:
        margin: 1.0
        factor: -0.5
"""


def _current_name(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("name: AGENT_"):
            return stripped.split("name: ", 1)[1].split()[0]
    raise RuntimeError("agent name not found")


def build(source: Path, name: str, stage: int, device: str, proximity: bool, ped: bool) -> str:
    text = source.read_text()
    current = _current_name(text)
    text = text.replace(f"name: {current}", f"name: {name}", 1)
    text = text.replace("starting_stage: 0", f"starting_stage: {stage}", 1)
    lines = []
    replaced = False
    for line in text.splitlines(keepends=True):
        if not replaced and line.strip().startswith("device:"):
            lines.append(f"        device: {device}\n")
            replaced = True
        else:
            lines.append(line)
    text = "".join(lines)
    if ped and "Stage 5:" not in text:
        text = text.replace("\n  monitoring:\n", PED_STAGE + "\n  monitoring:\n", 1)
    if proximity and "pedestrian_proximity:" not in text:
        marker = "# CONTAINS THE FRAMEWORK CONFIGURATION FOR TRAINING"
        text = text.replace(marker, PROXIMITY + "\n" + marker, 1)
    return text


JOBS = [
    # source, name, stage, proximity, ped
    (PROGRESS, "AGENT_5_NAV_PROGRESS", 0, False, False),
    (ORIGINAL, "AGENT_5_NAV_STATIC", 1, False, False),
    (PROGRESS, "AGENT_5_NAV_STATIC_PROGRESS", 1, False, False),
    (ORIGINAL, "AGENT_5_NAV_PED1", 4, False, True),
    (ORIGINAL, "AGENT_5_NAV_PED1_RISK", 4, True, True),
    (PROGRESS, "AGENT_5_NAV_PED1_PROGRESS", 4, False, True),
]


def main() -> None:
    device = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    for source, name, stage, proximity, ped in JOBS:
        dest = OUT / f"{name}.yaml"
        dest.write_text(build(source, name, stage, device, proximity, ped))
        print(dest)


if __name__ == "__main__":
    main()
