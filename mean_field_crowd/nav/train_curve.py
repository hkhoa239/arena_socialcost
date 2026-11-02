"""Training curve from an `arena train` log: episode outcomes and reward terms per timestep bin.

usage: train_curve.py LOG [BIN_STEPS]
Timesteps are the running sum of finished episode lengths over both envs, so a bin's
episodes are those that ended inside it.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

END = re.compile(r"Episode (\d+) ended, reason: ([^,]+), steps: (\d+).*?terms: (.*)$")
TERM = re.compile(r"(\w+)=(-?[\d.]+)")
SHOW = ["RewardGoalReached", "RewardApproachGoal", "RewardSafeDistance", "RewardFactoredSafeDistance", "RewardCollision"]


def parse(path: str) -> list[dict]:
    rows, total = [], 0
    for line in open(path, errors="replace"):
        m = END.search(line)
        if not m:
            continue
        steps = int(m.group(3))
        total += steps
        reason = m.group(2)
        terms = {k: float(v) for k, v in TERM.findall(m.group(4))}
        kind = "goal" if "SUCCESS" in reason and terms.get("RewardGoalReached", 0) > 0 else "collision" if "COLLISION" in reason else "timeout" if "STEP_LIMIT" in reason else "other"
        rows.append({"t": total, "steps": steps, "kind": kind, "terms": terms})
    return rows


def main() -> None:
    path = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    bins: dict[int, list[dict]] = defaultdict(list)
    for row in parse(path):
        bins[(row["t"] - 1) // width].append(row)
    print("| Steps | Episodes | Goal | Collision | Timeout | Mean length | " + " | ".join(t.replace("Reward", "") for t in SHOW) + " |")
    print("|---|--:|--:|--:|--:|--:|" + "--:|" * len(SHOW))
    for b in sorted(bins):
        eps = bins[b]
        n = len(eps)
        count = lambda k: sum(e["kind"] == k for e in eps)  # noqa: E731
        means = [sum(e["terms"].get(t, 0.0) for e in eps) / n for t in SHOW]
        print(f"| {b * width // 1000}-{(b + 1) * width // 1000}k | {n} | {count('goal')} | {count('collision')} | {count('timeout')} | "
              f"{sum(e['steps'] for e in eps) / n:.0f} | " + " | ".join(f"{v:.2f}" for v in means) + " |")


if __name__ == "__main__":
    main()
