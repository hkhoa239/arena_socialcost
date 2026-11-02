"""Write RESULTS.md from the pilot logs and matched eval json files."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/opt/arena_ws/data/mf_nav")
AGENTS = Path("/opt/arena_ws/data/agents")


def episodes_from_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    rows = []
    pat = re.compile(r"Episode (\d+) ended, reason: ([^,]+), steps: (\d+).*?return: ([-\d.]+), terms: (.*)")
    for match in pat.finditer(text):
        terms = {key: float(value) for key, value in re.findall(r"(Reward\w+)=([-\d.]+)", match.group(5))}
        rows.append({
            "reason": match.group(2).split(".")[-1],
            "steps": int(match.group(3)),
            "ret": float(match.group(4)),
            "terms": terms,
        })
    return rows


def train_line(name: str, log_name: str) -> str:
    rows = episodes_from_log(ROOT / log_name)
    if not rows:
        zip_path = AGENTS / name / "last_model.zip"
        return f"| {name} | pending | | | | zip={'yes' if zip_path.exists() else 'no'} |"
    counts = Counter(row["reason"] for row in rows)
    apps = [row["terms"].get("RewardApproachGoal", 0.0) for row in rows]
    speeds = ""
    return (
        f"| {name} | {len(rows)} | {counts.get('SUCCESS', 0)} | {counts.get('COLLISION', 0)} | "
        f"{counts.get('STEP_LIMIT', 0)} | {sum(apps) / len(apps):.3f} |"
    )


def eval_rows() -> list[dict]:
    found = []
    for path in sorted((ROOT / "eval").glob("*.json")) if (ROOT / "eval").exists() else []:
        found.append(json.loads(path.read_text()))
    return found


def bootstrap(success_a: list[bool], success_b: list[bool]) -> str:
    if len(success_a) != len(success_b) or not success_a:
        return "unpaired"
    rng = np.random.default_rng(0)
    a = np.asarray(success_a, dtype=float)
    b = np.asarray(success_b, dtype=float)
    idx = rng.integers(0, len(a), size=(10000, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return f"{diffs.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"


def eval_table(records: list[dict]) -> str:
    if not records:
        return "No matched evaluation has been written yet.\n"
    lines = [
        "| Policy | Stage | Episodes | Goals | Bumper | Timeout | Mean speed | Seeds identical across this file |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    by_stage: dict[str, list[dict]] = {}
    for record in records:
        name = Path(record["checkpoint"]).parent.name
        rows = record["episodes"]
        reasons = Counter(row["reason"] for row in rows)
        seeds = [row["episode_seed"] for row in rows]
        speed = sum(row["mean_abs_linear"] for row in rows) / len(rows)
        lines.append(
            f"| {name} | {record['stage']} | {len(rows)} | {sum(row['success'] for row in rows)} | "
            f"{reasons.get('COLLISION', 0)} | {reasons.get('STEP_LIMIT', 0)} | {speed:.3f} | {seeds} |"
        )
        by_stage.setdefault(record["stage"], []).append(record)
    lines.append("")
    lines.append("Paired bootstrap of goal-rate difference, first policy minus second, 10,000 resamples:")
    for stage, group in by_stage.items():
        if len(group) < 2:
            continue
        seeds_ok = [row["episode_seed"] for row in group[0]["episodes"]] == [row["episode_seed"] for row in group[1]["episodes"]]
        if not seeds_ok:
            lines.append(f"- {stage}: episode seeds differ, so the comparison is not paired.")
            continue
        for other in group[1:]:
            a = [row["success"] for row in group[0]["episodes"]]
            b = [row["success"] for row in other["episodes"]]
            lines.append(
                f"- {stage}: {Path(group[0]['checkpoint']).parent.name} minus {Path(other['checkpoint']).parent.name}: {bootstrap(a, b)}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    status = (ROOT / "STATUS.txt").read_text() if (ROOT / "STATUS.txt").exists() else ""
    body = f"""# Navigation reward pilot

This file is rewritten as the detached pipeline finishes each stage. Status tail:

```
{status.strip()[-2000:]}
```

## Root cause

`approach_goal` was paid on the straight-line subgoal. That subgoal sits `clamp(speed * 1.5 s, 0.5 m, 2.5 m)` ahead of the robot. The Jackal command limit is 0.5 m/s, so the lookahead is at most 0.75 m. Stage success radii are 1.5, 1.2, 1.0, and 0.8 m. The episode therefore ends at a radius that is still outside the lookahead, and the distance to the subgoal does not fall while the robot closes on the goal. Speeding up lengthens the lookahead and is charged as a retreat.

The empty-map run of the original reward (20,000 steps, seed 5) ended 1 goal / 1 bumper / 32 timeouts. The one goal, episode 7 at step 426, had approach reward -0.729 and the +20 bonus. At that instant the robot was 1.50 m from the goal and the subgoal was 0.75 m ahead, the full-speed lookahead. Across the 33 logged returns the approach term averaged -1.61.

The policy observes the subgoal angle, which does point at the goal, but the reward does not pay for reducing goal distance. Laser safety is the other dense term. On the empty map most timeouts had no safety penalty, so walls are not why the robot misses the goal. Gamma 0.99 makes the sparse +20 small when the crossing takes a few hundred steps. A 60 s episode at 0.5 m/s can cover the 30 m by 23 m empty map (mean random start-goal distance about 14 m). The horizon is long enough if the robot drives toward the goal.

Current pedestrian proximity removes the head-on bumper and still does not reach the goal, because progress is still unpaid. The one-second proximity term keeps charging an oncoming person and the short run slowed to about 0.04 m/s.

## The one change

`approach_goal` follows the final goal (`_follow_subgoal: false`). Coefficients are unchanged:

```
goal_reached: +20 inside the stage tolerance
safe_distance: -0.1 when the nearest laser return is inside 1.0 m
factored_safe_distance: -0.2 * (1.0 + robot_radius - min_laser) on that same event
approach_goal: +0.3 per meter closer to the final goal, -0.4 per meter farther
collision: -15, episode ends
pedestrian_proximity, only on the third arm: -0.5 * (1.0 - distance) while the nearest person is inside 1.0 m
```

A detour that adds L meters of path costs about 0.1 L because the retreat factor is 0.4 and the approach factor is 0.3. A 3 m detour costs 0.3, against a collision charge of 15 and a goal bonus of 20. No future pedestrian state and no residual forecast enter the reward. PPO, the network, and the seeds are unchanged. The empty-map original run is the existing `AGENT_5_DIAG_EMPTY` 20,000-step result and is not retrained.

`run_seed` is now a task-generator launch parameter. Before this, the launch file dropped it, each process drew a uuid, and repeated evaluations were not the same crowd.

## Training counts

These are the training episodes, not a matched evaluation. Approach is the mean episode sum of `RewardApproachGoal`.

| Run | Episodes | Goals | Bumper | Timeout | Mean approach |
|---|---:|---:|---:|---:|---:|
| AGENT_5_DIAG_EMPTY | 34 | 1 | 1 | 32 | -1.610 |
{train_line("AGENT_5_NAV_PROGRESS", "train_AGENT_5_NAV_PROGRESS.log")}
{train_line("AGENT_5_NAV_STATIC", "train_AGENT_5_NAV_STATIC.log")}
{train_line("AGENT_5_NAV_STATIC_PROGRESS", "train_AGENT_5_NAV_STATIC_PROGRESS.log")}
{train_line("AGENT_5_NAV_PED1", "train_AGENT_5_NAV_PED1.log")}
{train_line("AGENT_5_NAV_PED1_RISK", "train_AGENT_5_NAV_PED1_RISK.log")}
{train_line("AGENT_5_NAV_PED1_PROGRESS", "train_AGENT_5_NAV_PED1_PROGRESS.log")}

The attached CPU attempt that was stopped at 12,288 steps is `progress_empty_cpu.log`. It is not a finished policy. At that point it had 3 goals, 2 bumpers, and 17 timeouts in 22 episodes, and the two finished goals had positive approach reward (+0.93 and +1.83).

## Matched evaluation

{eval_table(eval_rows())}

## Mean-field comparison

Not started. The forecasting model, PedVO, and the AGENT_5 network were not changed. A mean-field run is justified only after the goal-tracking reward shows reliable collision-free goals on the matched empty, static, and single-pedestrian evaluations. Five episodes make the bootstrap interval wide; treat a small difference as unresolved.
"""
    (ROOT / "RESULTS.md").write_text(body)
    print(f"WROTE {ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
