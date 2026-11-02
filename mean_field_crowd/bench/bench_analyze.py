"""Navigation metrics of the benchmark from trace_eval.py rollouts.

Layout: eval/<RUN>/<CKPT>[__<MODE>]_<STAGE>.json, RUN = BENCH_<ARM>_S<seed>, CKPT = t20000..t100000,
MODE = online|zero for the forecast ablation; val_<CKPT>_<STAGE>.json is the validation sweep (mode "val"). Episode i of a stage is the same initial condition
for every run (same run_seed, same seed formula), which the analysis checks.

Per episode:
  goal               done_reason SUCCESS with RewardGoalReached = +20
  cf_goal            goal and no contact event of any kind during the episode
  static_collision   a wall/static contact event (physical collision with the map or an obstacle)
  ped_contact        a pedestrian contact event
  laser_bumper       COLLISION ending with no contact event: the laser fallback (min laser <= 0.317 m)
  timeout            STEP_LIMIT
  ped_clearance      min over steps of (robot-pedestrian distance - 0.267 - 0.3) m, pedestrians only
  safety_violation   fraction of steps with min laser < 1.0 m (laser safety margin)
  reverse_frac       fraction of steps with executed v < -0.05 m/s
  forward_speed      mean executed v over the episode
  time_to_goal       seconds, successful episodes
  path_efficiency    (start-goal distance - tolerance) / path length, successful episodes
  progress           (d_start - d_end) / d_start

usage: bench_analyze.py EVAL_DIR OUT_JSON
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROBOT_R, PED_R = 0.267, 0.3
BOOT = 2000
METRICS = ["cf_goal", "goal", "static_collision", "ped_contact", "laser_bumper", "timeout", "ped_clearance", "safety_violation",
           "reverse_frac", "forward_speed", "time_to_goal", "path_efficiency", "progress"]
NAME = re.compile(r"^(?P<val>val_)?(?P<ckpt>t\d+)(?:__(?P<mode>[a-z]+))?_(?P<stage>b_[a-z_]+)\.json$")
RUN = re.compile(r"^BENCH_(?P<arm>[A-Z0-9]+_(?:FWD|FREE))_S(?P<seed>\d+)$")


def episode_metrics(ep: dict) -> dict:
    s = ep["series"]
    goal_xy = ep["record_goal_map"][:2]
    tol = ep["record_tolerance"] or 1.0
    start = ep["first_obs"].get("robot") or s[0]["robot"]
    d0 = math.hypot(start[0] - goal_xy[0], start[1] - goal_xy[1])
    xy = np.array([x["robot"] for x in s])
    d_end = math.hypot(xy[-1][0] - goal_xy[0], xy[-1][1] - goal_xy[1])
    kinds = {k for x in s for k, _ in (x.get("events") or [])} | {k for k, _ in (ep.get("contact_events") or [])}
    reason = str(ep["reason"])
    goal = bool(ep.get("goal_bonus")) and "SUCCESS" in reason
    v = np.array([(x.get("cmd") or [0.0])[0] for x in s])
    laser = np.array([x["min_laser"] for x in s if x.get("min_laser") is not None])
    clear = [min(math.hypot(p[0] - x["robot"][0], p[1] - x["robot"][1]) for p in x["peds"]) - ROBOT_R - PED_R for x in s if x.get("peds")]
    path = float(np.sum(np.hypot(*np.diff(np.vstack([start, xy]), axis=0).T)))
    return {
        "goal": float(goal),
        "cf_goal": float(goal and not kinds),
        "static_collision": float(bool(kinds & {"wall", "static"})),
        "ped_contact": float("pedestrian" in kinds),
        "timeout": float("STEP_LIMIT" in reason),
        "collision_end": float("COLLISION" in reason),
        # COLLISION ending without any contact event: RewardCollision's laser fallback
        # (min laser <= robot radius + 0.05 m bumper zone), a proximity stop, not a contact.
        "laser_bumper": float("COLLISION" in reason and not kinds),
        "ped_clearance": float(min(clear)) if clear else float("nan"),
        "safety_violation": float(np.mean(laser < 1.0)) if laser.size else float("nan"),
        "reverse_frac": float(np.mean(v < -0.05)),
        "forward_speed": float(np.mean(v)),
        "time_to_goal": ep["steps"] / 10.0 if goal else float("nan"),
        "path_efficiency": max(d0 - tol, 0.0) / path if goal and path > 0 else float("nan"),
        "progress": (d0 - d_end) / d0 if d0 > 0 else float("nan"),
        "seed": ep["episode_seed"],
        "start_goal": [round(start[0], 1), round(start[1], 1), round(goal_xy[0], 2), round(goal_xy[1], 2)],
    }


def load(eval_dir: Path) -> dict:
    """runs[(arm, seed)][(ckpt, mode, stage)] = list of episode metric dicts."""
    runs: dict = defaultdict(dict)
    for run_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        m = RUN.match(run_dir.name)
        if not m:
            continue
        for f in sorted(run_dir.glob("*.json")):
            n = NAME.match(f.name)
            if not n:
                continue
            data = json.loads(f.read_text())
            runs[(m["arm"], int(m["seed"]))][(n["ckpt"], "val" if n["val"] else (n["mode"] or ""), n["stage"])] = [episode_metrics(e) for e in data["episodes"]]
    return runs


def mean(eps: list[dict], key: str) -> float:
    vals = [e[key] for e in eps if not math.isnan(e[key])]
    return float(np.mean(vals)) if vals else float("nan")


def paired(a: list[dict], b: list[dict], key: str, rng) -> dict | None:
    """Paired bootstrap of mean(a) - mean(b) over matched episodes (episodes where both are defined)."""
    n = min(len(a), len(b))
    if any(a[i]["seed"] != b[i]["seed"] for i in range(n)):
        raise ValueError("episodes are not matched (different episode seeds)")
    x = np.array([a[i][key] for i in range(n)], dtype=float)
    y = np.array([b[i][key] for i in range(n)], dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 2:
        return None
    d = x[ok] - y[ok]
    draws = rng.integers(0, len(d), size=(BOOT, len(d)))
    boot = d[draws].mean(1)
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n": int(ok.sum())}


def main() -> None:
    eval_dir, dest = Path(sys.argv[1]), Path(sys.argv[2])
    runs = load(eval_dir)
    rng = np.random.default_rng(5)
    table = []
    for (arm, seed), cells in sorted(runs.items()):
        for (ckpt, mode, stage), eps in sorted(cells.items()):
            table.append({"arm": arm, "seed": seed, "ckpt": ckpt, "mode": mode, "stage": stage, "episodes": len(eps),
                          **{k: mean(eps, k) for k in METRICS}})
    # Paired comparisons at the same checkpoint, stage and seed, then pooled over seeds per episode.
    comparisons = []
    pairs = [("ONLINE_FWD", "A5_FWD"), ("RESIDUAL_FWD", "A5_FWD"), ("RESIDUAL_FWD", "ONLINE_FWD"), ("A5_FREE", "A5_FWD")]
    stages = sorted({k[2] for cells in runs.values() for k in cells})
    ckpts = sorted({k[0] for cells in runs.values() for k in cells}, key=lambda c: int(c[1:]))
    seeds = sorted({s for _, s in runs})
    for a, b in pairs:
        for ckpt in ckpts:
            for stage in stages:
                per_seed = {}
                pooled_a, pooled_b = [], []
                for seed in seeds:
                    ea = runs.get((a, seed), {}).get((ckpt, "", stage))
                    eb = runs.get((b, seed), {}).get((ckpt, "", stage))
                    if not ea or not eb:
                        continue
                    per_seed[seed] = {k: mean(ea, k) - mean(eb, k) for k in ("cf_goal", "goal", "static_collision", "ped_contact", "timeout", "ped_clearance")}
                    pooled_a += ea
                    pooled_b += eb
                if not per_seed:
                    continue
                comparisons.append({
                    "a": a, "b": b, "ckpt": ckpt, "stage": stage, "seeds": sorted(per_seed),
                    "per_seed": per_seed,
                    "across_seed_std_cf_goal": float(np.std([v["cf_goal"] for v in per_seed.values()])) if len(per_seed) > 1 else None,
                    **{k: paired(pooled_a, pooled_b, k, rng) for k in ("cf_goal", "goal", "static_collision", "ped_contact", "laser_bumper", "timeout", "ped_clearance", "progress")},
                })
    # Forecast ablation: residual-trained runs evaluated with online/zero forecast vs their normal residual forecast.
    ablation = []
    for (arm, seed), cells in sorted(runs.items()):
        if arm != "RESIDUAL_FWD":
            continue
        for (ckpt, mode, stage), eps in sorted(cells.items()):
            base = cells.get((ckpt, "", stage))
            if mode and mode != "val" and base:
                ablation.append({"seed": seed, "ckpt": ckpt, "stage": stage, "mode": mode,
                                 **{k: paired(eps, base, k, rng) for k in ("cf_goal", "goal", "static_collision", "ped_contact", "laser_bumper", "timeout", "ped_clearance", "progress")}})
    # Episodes failed by every evaluated policy (candidate impossible / blocked layouts).
    hard = defaultdict(lambda: defaultdict(list))
    for (arm, seed), cells in runs.items():
        for (ckpt, mode, stage), eps in cells.items():
            if mode:
                continue
            for i, e in enumerate(eps):
                hard[stage][i].append(e["goal"])
    never = {stage: [i for i, g in sorted(v.items()) if len(g) >= 3 and not any(g)] for stage, v in hard.items()}
    out = {"table": table, "comparisons": comparisons, "ablation": ablation, "never_solved_episode_indices": never}
    dest.write_text(json.dumps(out, indent=1))
    print(f"WROTE {dest}: {len(table)} cells, {len(comparisons)} comparisons, {len(ablation)} ablation cells")


if __name__ == "__main__":
    main()
