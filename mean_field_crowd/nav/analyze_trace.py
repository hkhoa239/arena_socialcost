"""Summarize trace/*.json into a matched table, failure modes, and reward decomposition.

usage: analyze_trace.py [GLOB ...]   (default: trace/*.json, skipping *_NAV2BUG / *_fixcheck5)
Writes trace/summary.json and prints markdown tables.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/robotics/arena_ws/data/mf_nav/trace")
if not ROOT.exists():
    ROOT = Path("/opt/arena_ws/data/mf_nav/trace")

TERMS = ["RewardGoalReached", "RewardApproachGoal", "RewardSafeDistance", "RewardFactoredSafeDistance", "RewardCollision", "RewardPedestrianProximity"]


def outcome(ep: dict) -> str:
    reason = str(ep["reason"])
    if ep["goal_bonus"] and "SUCCESS" in reason:
        return "goal"
    if "COLLISION" in reason:
        return "collision"
    if "STEP_LIMIT" in reason:
        return "timeout"
    if ep["success"]:
        return "tg_success_no_bonus"
    return "other"


def episode_metrics(ep: dict) -> dict:
    s = ep["series"]
    goal = ep["record_goal_map"]
    d = [x.get("d_goal_true") for x in s if x.get("d_goal_true") is not None]
    start = ep["first_obs"].get("robot")
    d0 = math.hypot(start[0] - goal[0], start[1] - goal[1]) if start and goal else (d[0] if d else float("nan"))
    act = np.array([x["action"][0] for x in s]) if s else np.zeros(1)
    yaw = np.array([x["action"][-1] for x in s]) if s else np.zeros(1)
    twist = np.array([x["twist"][0] for x in s if x.get("twist")]) if s else np.zeros(1)
    laser = np.array([x["min_laser"] for x in s if x.get("min_laser") is not None])
    end = d[-1] if d else float("nan")
    # Steps after which distance never improves by more than 0.5 m: "stuck" tail
    best, stall = float("inf"), 0
    for value in d:
        if value < best - 0.5:
            best, stall = value, 0
        else:
            stall += 1
    tol = ep["record_tolerance"] or float("nan")
    radius = s[-1]["goal_radius"] if s else float("nan")
    # Goal-obs mismatch excluding a possible terminal snapshot artefact
    return {
        "outcome": outcome(ep),
        "steps": ep["steps"],
        "d_start": d0,
        "d_end": end,
        "d_min": min(d) if d else float("nan"),
        "progress": d0 - end,
        "progress_frac": (d0 - end) / d0 if d0 else float("nan"),
        "mean_v_cmd": float(np.mean(act)),
        "mean_abs_v": float(np.mean(np.abs(act))),
        "reverse_frac": float(np.mean(act < -0.05)),
        "mean_abs_w": float(np.mean(np.abs(yaw))),
        "mean_twist_v": float(np.mean(twist)) if twist.size else float("nan"),
        "min_laser_min": float(laser.min()) if laser.size else float("nan"),
        "min_laser_end": float(laser[-1]) if laser.size else float("nan"),
        "frac_inside_1m": float(np.mean(laser < 1.0)) if laser.size else float("nan"),
        "stall_tail": stall,
        "tolerance": tol,
        "reward_radius": radius,
        "radius_matches": abs(tol - radius) < 1e-6 if tol == tol else None,
        "goal_mismatch_steps": ep["goal_mismatch_steps"],
        "seed_ok": ep["seed_matches_formula"],
        "contacts": ep["contacts"],
        "terms": {t: ep["terms"].get(t, 0.0) for t in TERMS if t in ep["terms"] or t != "RewardPedestrianProximity"},
        "return": ep["return"],
    }


def summarize(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    eps = [episode_metrics(ep) for ep in data["episodes"]]
    n = len(eps)
    counts = {k: sum(e["outcome"] == k for e in eps) for k in ("goal", "collision", "timeout", "tg_success_no_bonus", "other")}
    mean = lambda key: float(np.nanmean([e[key] for e in eps])) if eps else float("nan")  # noqa: E731
    terms = {t: float(np.mean([e["terms"].get(t, 0.0) for e in eps])) for t in TERMS}
    return {
        "file": Path(path).name,
        "checkpoint": data["checkpoint"],
        "stage": data["stage"],
        "run_seed": data["run_seed"],
        "episodes": n,
        **counts,
        "seeds": [ep["episode_seed"] for ep in data["episodes"]],
        "seed_ok": all(e["seed_ok"] for e in eps),
        "goal_obs_ok": all(e["goal_mismatch_steps"] <= 1 for e in eps),
        "radius_ok": all(e["radius_matches"] in (True, None) for e in eps),
        "mean_progress_frac": mean("progress_frac"),
        "mean_d_end": mean("d_end"),
        "mean_abs_v": mean("mean_abs_v"),
        "reverse_frac": mean("reverse_frac"),
        "mean_abs_w": mean("mean_abs_w"),
        "frac_inside_1m": mean("frac_inside_1m"),
        "terms": terms,
        "per_episode": eps,
    }


def main() -> None:
    if sys.argv[1:]:
        files = sorted(f for p in sys.argv[1:] for f in glob.glob(p))
    else:
        files = sorted(f for f in glob.glob(str(ROOT / "*.json")) if "NAV2BUG" not in f and "fixcheck" not in f and not f.endswith("summary.json"))
    rows = [summarize(f) for f in files]
    (ROOT / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    print("| Run | Stage | N | Goal | Collision | Timeout | TG-only success | Progress frac | Final dist (m) | mean abs v | Reverse frac | inside 1 m | seeds/goal/radius ok |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    for r in rows:
        tag = r["file"].rsplit("_", 1)[0]
        print(f"| {tag} | {r['stage']} | {r['episodes']} | {r['goal']} | {r['collision']} | {r['timeout']} | {r['tg_success_no_bonus']} | "
              f"{r['mean_progress_frac']:.2f} | {r['mean_d_end']:.2f} | {r['mean_abs_v']:.2f} | {r['reverse_frac']:.2f} | {r['frac_inside_1m']:.2f} | "
              f"{'/'.join('y' if x else 'N' for x in (r['seed_ok'], r['goal_obs_ok'], r['radius_ok']))} |")
    print()
    print("| Run | Stage | " + " | ".join(t.replace("Reward", "") for t in TERMS) + " |")
    print("|---|---|" + "--:|" * len(TERMS))
    for r in rows:
        tag = r["file"].rsplit("_", 1)[0]
        print(f"| {tag} | {r['stage']} | " + " | ".join(f"{r['terms'][t]:.2f}" for t in TERMS) + " |")


if __name__ == "__main__":
    main()
