"""Benchmark report from whatever results exist. Partial results are labelled as such.

Inputs: results/forecast_bench.json, eval/ (via bench_analyze.py), logs/train_<RUN>.log, STATUS.txt.
Outputs: results/REPORT.md and results/plots/*.png.

usage: report.py [SEEDS...]   (seeds expected for completion, default 5)
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

B = Path(__file__).resolve().parent
RES = B / "results"
PLOTS = RES / "plots"
ARMS = ["A5_FWD", "ONLINE_FWD", "RESIDUAL_FWD", "A5_FREE"]
COLORS = {"A5_FWD": "#555555", "ONLINE_FWD": "#2364aa", "RESIDUAL_FWD": "#d1342f", "A5_FREE": "#e09a1f"}
STAGES = ["b_empty", "b_static", "b_moderate", "b_dense", "b_stress", "b_headon", "b_crossing"]
CROWD = {"b_empty": 0, "b_static": 0, "b_moderate": 4, "b_dense": 12.5, "b_stress": 20}
FC_METHODS = ["persistence", "online_cv", "residual", "cnn", "oracle_cv"]
SB3 = re.compile(r"\|\s+([a-z_0-9]+)\s+\|\s+([-0-9.e+]+)\s+\|")
END = re.compile(r"Episode (\d+) ended, reason: ([^,]+), steps: (\d+).*?terms: (.*)$")
MIX = re.compile(r"\[MIXTURE\] env (\d+) started stage (\d+)")
TERM = re.compile(r"(\w+)=(-?[\d.]+)")


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def ci(c):
    return "–" if not c else f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"


# ---------------------------------------------------------------- forecasting
def forecast_section(lines: list[str]) -> None:
    path = RES / "forecast_bench.json"
    lines.append("## Phase 1: forecasting benchmark\n")
    if not path.exists():
        lines.append("_Not available yet._\n")
        return
    fc = json.loads(path.read_text())
    lines.append("Held-out closed-loop PedVO rollouts (run seeds `bench-fc-*`, never used for training the residual or for navigation evaluation). "
                 "Density 16×16, ROI 20 m, σ 0.5 m, horizon 1 s. Lower is better. CIs are paired bootstrap over episodes.\n")
    for region, title in (("all", "whole grid"), ("near2m", "cells within 2 m of the robot"), ("near3m", "cells within 3 m")):
        lines.append(f"**MSE ×1e3, all pairs, {title}**\n")
        lines.append("| Scenario | Episodes | Pairs | " + " | ".join(FC_METHODS) + " | Residual − Online CV (95% CI) |")
        lines.append("|---|--:|--:|" + "--:|" * len(FC_METHODS) + "---|")
        for name, cell in [("ALL", fc["all"]), *sorted(fc["stages"].items())]:
            c = cell.get(f"all_pairs/{region}", {})
            if "online_cv" not in c:
                continue
            d = c["residual_minus_online_cv"]
            lines.append(f"| {name} | {cell['episodes']} | {cell['pairs']} | " + " | ".join(fmt(c[m] * 1e3) for m in FC_METHODS)
                         + f" | {d['mean'] * 1e3:+.3f} [{d['ci95'][0] * 1e3:+.3f}, {d['ci95'][1] * 1e3:+.3f}] |")
        lines.append("")
    lines.append("**Active pedestrian avoidance** (a pedestrian within 3 m whose true 1 s displacement deviates > 0.3 m laterally from constant velocity), MSE ×1e3\n")
    lines.append("| Scenario | Active pairs | " + " | ".join(FC_METHODS) + " | Residual − Online CV (95% CI) |")
    lines.append("|---|--:|" + "--:|" * len(FC_METHODS) + "---|")
    for name, cell in [("ALL", fc["all"]), *sorted(fc["stages"].items())]:
        c = cell.get("active_avoidance/all", {})
        if "online_cv" not in c:
            continue
        d = c["residual_minus_online_cv"]
        lines.append(f"| {name} | {cell['active_pairs']} | " + " | ".join(fmt(c[m] * 1e3) for m in FC_METHODS)
                     + f" | {d['mean'] * 1e3:+.3f} [{d['ci95'][0] * 1e3:+.3f}, {d['ci95'][1] * 1e3:+.3f}] |")
    lines.append("")
    if fc.get("by_count"):
        counts = sorted(fc["by_count"], key=lambda k: int(k[1:]))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, title in ((axes[0], "all_pairs/all", "whole grid"), (axes[1], "all_pairs/near3m", "within 3 m of robot")):
            for m in FC_METHODS:
                ys = [fc["by_count"][k].get(key, {}).get(m, np.nan) for k in counts]
                ax.plot([int(k[1:]) for k in counts], ys, "o-", label=m)
            ax.set_xlabel("pedestrians")
            ax.set_ylabel("density MSE")
            ax.set_title(title)
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        PLOTS.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS / "forecast_mse_vs_crowd.png", dpi=110)
        plt.close(fig)
        lines.append("![forecast MSE vs crowd size](plots/forecast_mse_vs_crowd.png)\n")
    lat = fc.get("latency", {})
    if lat:
        lines.append("**Forecast latency per call (batch 1, ms)**: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(lat.items())) + "\n")


# ---------------------------------------------------------------- training
def parse_training(log: Path) -> dict:
    tables, cur = [], {}
    episodes, mixture = [], defaultdict(list)
    total = 0
    for line in log.read_text(errors="replace").splitlines():
        m = SB3.search(line)
        if m:
            try:
                cur[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
            if m.group(1) == "total_timesteps" and len(cur) > 3:
                tables.append(cur)
                cur = {}
            continue
        m = MIX.search(line)
        if m:
            mixture[int(m.group(1))].append(int(m.group(2)))
            continue
        m = END.search(line)
        if m:
            steps = int(m.group(3))
            total += steps
            terms = {k: float(v) for k, v in TERM.findall(m.group(4))}
            reason = m.group(2)
            kind = "goal" if "SUCCESS" in reason and terms.get("RewardGoalReached", 0) > 0 else "collision" if "COLLISION" in reason else "timeout" if "STEP_LIMIT" in reason else "other"
            episodes.append({"t": total, "kind": kind, "steps": steps, "return": sum(terms.values()), "terms": terms})
    return {"tables": tables, "episodes": episodes, "mixture": mixture}


def training_section(lines: list[str], runs: list[str]) -> None:
    lines.append("## Phase 3: training curves\n")
    parsed = {r: parse_training(B / "logs" / f"train_{r}.log") for r in runs if (B / "logs" / f"train_{r}.log").exists()}
    if not parsed:
        lines.append("_No training log yet._\n")
        return
    keys = [("entropy_loss", "entropy loss"), ("std", "action std"), ("approx_kl", "approx KL"), ("clip_fraction", "clip fraction"),
            ("explained_variance", "explained variance"), ("reverse_frac", "reverse fraction"), ("v_mean", "mean v (m/s)"), ("clip_rate_0", "action clip rate (v)")]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for run, p in parsed.items():
        arm = re.match(r"BENCH_(.+)_S\d+", run).group(1)
        style = {"color": COLORS.get(arm, "k"), "alpha": 0.85, "lw": 1.2, "label": run}
        t = [x.get("total_timesteps", np.nan) / 1000 for x in p["tables"]]
        for ax, (key, title) in zip(axes.flat[:8], keys):
            ax.plot(t, [x.get(key, np.nan) for x in p["tables"]], **style)
            ax.set_title(title, fontsize=9)
        eps = p["episodes"]
        if eps:
            edges = np.arange(0, 110_001, 10_000)
            for ax, kind in zip(axes.flat[8:11], ("goal", "collision", "timeout")):
                rate = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    chunk = [e for e in eps if lo < e["t"] <= hi]
                    rate.append(np.mean([e["kind"] == kind for e in chunk]) if chunk else np.nan)
                ax.plot((edges[:-1] + 5000) / 1000, rate, "o-", **style)
                ax.set_title(f"training {kind} rate per 10k", fontsize=9)
            ax = axes.flat[11]
            ret = [np.mean([e["return"] for e in eps if lo < e["t"] <= hi]) if any(lo < e["t"] <= hi for e in eps) else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot((edges[:-1] + 5000) / 1000, ret, "o-", **style)
            ax.set_title("mean episode return per 10k", fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.set_xlabel("steps (k)", fontsize=8)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "training_curves.png", dpi=100)
    plt.close(fig)
    lines.append("Training episodes are not a matched evaluation; they show learning dynamics only.\n")
    lines.append("![training curves](plots/training_curves.png)\n")
    lines.append("| Run | Episodes | Goal | Collision | Timeout | Mixture stages started (empty/static/moderate/dense) | Last fps | GPU MiB |")
    lines.append("|---|--:|--:|--:|--:|---|--:|--:|")
    for run, p in parsed.items():
        eps = p["episodes"]
        counts = [0, 0, 0, 0]
        for stages in p["mixture"].values():
            for s in stages:
                counts[s] += 1
        last = p["tables"][-1] if p["tables"] else {}
        lines.append(f"| {run} | {len(eps)} | {sum(e['kind'] == 'goal' for e in eps)} | {sum(e['kind'] == 'collision' for e in eps)} | "
                     f"{sum(e['kind'] == 'timeout' for e in eps)} | {'/'.join(map(str, counts))} | {fmt(last.get('fps'), 0)} | {fmt(last.get('gpu_max_mem_mib'), 0)} |")
    lines.append("")


# ---------------------------------------------------------------- navigation
def navigation_section(lines: list[str]) -> None:
    out = RES / "navigation.json"
    subprocess.run([sys.executable, str(B / "bench_analyze.py"), str(B / "eval"), str(out)], check=False, capture_output=True)
    lines.append("## Phase 4: matched navigation evaluation\n")
    if not out.exists():
        lines.append("_No evaluation yet._\n")
        return
    nav = json.loads(out.read_text())
    table = nav["table"]
    val = [r for r in table if r["mode"] == "val"]
    if val:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, stage in zip(axes, ("b_static", "b_dense")):
            for arm in ARMS:
                for seed in sorted({r["seed"] for r in val}):
                    rows = sorted([r for r in val if r["arm"] == arm and r["seed"] == seed and r["stage"] == stage], key=lambda r: int(r["ckpt"][1:]))
                    if rows:
                        ax.plot([int(r["ckpt"][1:]) / 1000 for r in rows], [r["cf_goal"] for r in rows], "o-", color=COLORS[arm], label=f"{arm} s{seed}")
            ax.set_title(f"validation collision-free success, {stage}")
            ax.set_xlabel("checkpoint (k steps)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        PLOTS.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS / "validation_vs_steps.png", dpi=110)
        plt.close(fig)
        lines.append("![validation success vs checkpoint](plots/validation_vs_steps.png)\n")
    test = [r for r in table if r["mode"] == ""]
    if test:
        lines.append("**Test set** (run seeds `bench-test-*`; 25 episodes on random scenarios, 10 on head-on/crossing). Rates are fractions of episodes.\n")
        cols = ["cf_goal", "goal", "static_collision", "ped_contact", "laser_bumper", "timeout", "ped_clearance", "safety_violation", "reverse_frac", "forward_speed", "time_to_goal", "path_efficiency", "progress"]
        lines.append("| Arm | Seed | Ckpt | Scenario | N | " + " | ".join(cols) + " |")
        lines.append("|---|--:|---|---|--:|" + "--:|" * len(cols))
        for r in sorted(test, key=lambda r: (STAGES.index(r["stage"]) if r["stage"] in STAGES else 99, ARMS.index(r["arm"]) if r["arm"] in ARMS else 9, r["seed"], int(r["ckpt"][1:]))):
            lines.append(f"| {r['arm']} | {r['seed']} | {r['ckpt']} | {r['stage']} | {r['episodes']} | " + " | ".join(fmt(r[c], 2) for c in cols) + " |")
        lines.append("")
        final = [r for r in test if r["ckpt"] == "t100000"]
        crowd = [s for s in CROWD if any(r["stage"] == s for r in final)]
        if crowd:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, key, title in zip(axes, ("cf_goal", "laser_bumper", "ped_contact"), ("collision-free success", "laser-bumper stop rate", "pedestrian contact rate")):
                for arm in ARMS:
                    xs, ys = [], []
                    for s in crowd:
                        vals = [r[key] for r in final if r["arm"] == arm and r["stage"] == s]
                        if vals:
                            xs.append(CROWD[s] + (0.3 if s == "b_static" else 0))
                            ys.append(np.mean(vals))
                    if xs:
                        ax.plot(xs, ys, "o-", color=COLORS[arm], label=arm)
                ax.set_xlabel("pedestrians (empty, static, moderate, dense, stress)")
                ax.set_title(f"{title}, 100k, mean over seeds")
                ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS / "rates_vs_crowd.png", dpi=110)
            plt.close(fig)
            lines.append("![rates vs crowd size](plots/rates_vs_crowd.png)\n")
    comps = nav.get("comparisons", [])
    if comps:
        lines.append("**Paired comparisons** (same episodes; Δ = A − B with 95% paired-bootstrap CI, pooled over available seeds; per-seed deltas give seed variability)\n")
        lines.append("| A − B | Ckpt | Scenario | Seeds | Δ collision-free success | Δ laser bumper | Δ ped contact | Δ timeout | Δ ped clearance (m) | per-seed Δ cf_goal |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in comps:
            if c["ckpt"] != "t100000":
                continue
            per = ", ".join(f"s{s}:{v['cf_goal']:+.2f}" for s, v in sorted(c["per_seed"].items(), key=lambda kv: int(kv[0])))
            lines.append(f"| {c['a']} − {c['b']} | {c['ckpt']} | {c['stage']} | {','.join(map(str, c['seeds']))} | {ci(c['cf_goal'])} | {ci(c.get('laser_bumper'))} | {ci(c['ped_contact'])} | {ci(c['timeout'])} | {ci(c['ped_clearance'])} | {per} |")
        lines.append("")
    abl = nav.get("ablation", [])
    lines.append("## Phase 5: forecast ablation (residual-trained policies)\n")
    if abl:
        lines.append("Δ = metric with the substituted forecast − metric with the residual forecast, same episodes. The zero channel is an observation distribution shift and is not a clean 'no forecast' baseline.\n")
        lines.append("| Seed | Ckpt | Scenario | Mode | Δ collision-free success | Δ laser bumper | Δ ped contact | Δ timeout | Δ ped clearance | Δ progress |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for a in abl:
            lines.append(f"| {a['seed']} | {a['ckpt']} | {a['stage']} | {a['mode']} | {ci(a['cf_goal'])} | {ci(a.get('laser_bumper'))} | {ci(a['ped_contact'])} | {ci(a['timeout'])} | {ci(a['ped_clearance'])} | {ci(a['progress'])} |")
        lines.append("")
    else:
        lines.append("_Not available yet._\n")
    never = {k: v for k, v in nav.get("never_solved_episode_indices", {}).items() if v}
    lines.append("## Failure analysis\n")
    lines.append("Episodes no evaluated policy solved (≥3 policy evaluations, candidate impossible or very hard layouts): "
                 + (", ".join(f"{k}: {v}" for k, v in sorted(never.items())) if never else "none so far") + "\n")


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [5]
    runs = [f"BENCH_{arm}_S{s}" for s in seeds for arm in ARMS]
    status = (B / "STATUS.txt").read_text() if (B / "STATUS.txt").exists() else ""
    done_runs = [r for r in runs if f"RUN_DONE {r}" in status]
    complete = "FORECAST_DONE" in status and len(done_runs) == len(runs)
    lines = [f"# Mean-field navigation benchmark — {'FINAL' if complete else 'PARTIAL'} report",
             f"\nGenerated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Expected seeds: {seeds}. "
             f"Completed runs: {', '.join(done_runs) or 'none'}. Forecasting: {'done' if 'FORECAST_DONE' in status else 'not finished'}.\n"]
    if not complete:
        lines.append("> **Partial results.** Runs and evaluations still in progress are missing from the tables; nothing here is a final conclusion.\n")
    forecast_section(lines)
    training_section(lines, runs)
    navigation_section(lines)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {RES / 'REPORT.md'}")


if __name__ == "__main__":
    main()
