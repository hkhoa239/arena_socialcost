"""Figures for the LaTeX progress report (docs/sleek-template-master/resources/figures).

usage: report_figures.py OUT_DIR
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
sys.path.insert(0, "/opt/arena_ws/data/mf_nav")
import forecast_bench as fb  # noqa: E402
import render_episode as rd  # noqa: E402
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.interaction import InteractionResidual, residual_checkpoint  # noqa: E402
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.response_model import CrowdResponseNet  # noqa: E402

SB3 = re.compile(r"\|\s+([a-z_0-9]+)\s+\|\s+([-0-9.e+]+)\s+\|")
RUNS = {"BENCH_A5_FWD_S5": ("AGENT_5 (fwd)", "#555555"), "BENCH_ONLINE_FWD_S5": ("MF_ONLINE (fwd)", "#2364aa"),
        "BENCH_RESIDUAL_FWD_S5": ("MF_RESIDUAL (fwd)", "#d1342f"), "BENCH_A5_FREE_S5": ("AGENT_5 (unrestricted)", "#e09a1f")}


def density_example(out: Path) -> None:
    residual = InteractionResidual()
    residual.load_state_dict(torch.load(residual_checkpoint(), map_location="cpu", weights_only=True))
    residual.eval()
    cnn = CrowdResponseNet(conditioned=True)
    cnn.load_state_dict(torch.load(fb.CNN, map_location="cpu"))
    cnn.eval()
    data = json.loads((B / "forecast" / "fc_n10.json").read_text())
    best = None
    for ep in data["episodes"]:
        p = fb.pairs_of(ep, residual, cnn)
        if p is None:
            continue
        gap = ((p["residual"] - p["online_cv"]) ** 2).sum(axis=(1, 2)) * p["active"]
        i = int(np.argmax(gap))
        if best is None or gap[i] > best[0]:
            best = (gap[i], p, i)
    _, p, i = best
    panels = [("current density $\\rho_t$\n(persistence)", "persistence"), ("online CV forecast\n$\\hat\\rho^{cv}_{t+1s}$", "online_cv"),
              ("CV + residual\n$\\hat\\rho^{res}_{t+1s}$", "residual"), ("conditioned CNN", "cnn"),
              ("oracle-pose CV\n(diagnostic)", "oracle_cv"), ("target $\\rho_{t+1s}$", "target")]
    vmax = max(float(p[k][i].max()) for _, k in panels)
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
    for ax, (title, key) in zip(axes, panels):
        im = ax.imshow(p[key][i], origin="lower", extent=(-10, 10, -10, 10), vmin=0, vmax=vmax, cmap="magma")
        ax.plot(0, 0, marker=(3, 0, -90), color="cyan", ms=10)
        mse = float(((p[key][i] - p["target"][i]) ** 2).mean()) * 1e3
        ax.set_title(title + ("" if key == "target" else f"\nMSE×1e3 = {mse:.2f}"), fontsize=8.5)
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-10, 0, 10])
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    fig.savefig(out / "density_example.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def training_collapse(out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    keys = [("v_mean", "mean executed $v$ (m/s)"), ("stop_frac", "fraction of steps with $|v|\\leq0.05$"),
            ("clip_rate_0", "clip rate of sampled $v$"), ("std", "policy action std")]
    for run, (label, color) in RUNS.items():
        log = B / "logs" / f"train_{run}.log"
        if not log.exists():
            continue
        tables, cur = [], {}
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
        t = np.array([x.get("total_timesteps", np.nan) for x in tables]) / 1000
        for ax, (key, title) in zip(axes, keys):
            y = np.array([x.get(key, np.nan) for x in tables])
            ok = ~np.isnan(y)
            kernel = np.ones(5) / 5
            smooth = np.convolve(y[ok], kernel, mode="valid")
            ax.plot(t[ok][2:-2], smooth, color=color, label=label, lw=1.5)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("training steps (k)", fontsize=8)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_collapse.png", dpi=130)
    plt.close(fig)


def overviews(out: Path) -> None:
    picks = [("BENCH_RESIDUAL_FWD_S5", "t100000_b_dense", "traj_residual_dense"),
             ("BENCH_A5_FWD_S5", "t100000_b_dense", "traj_a5_dense"),
             ("BENCH_ONLINE_FWD_S5", "t100000_b_headon", "traj_online_headon")]
    for run, name, dest in picks:
        f = B / "eval" / run / f"{name}.json"
        if not f.exists():
            continue
        trace = json.loads(f.read_text())
        trace["episodes"] = trace["episodes"][:10]
        rd.overview(trace, out / f"{dest}.png")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("forecast_mse_vs_crowd.png", "validation_vs_steps.png", "rates_vs_crowd.png"):
        src = B / "results" / "plots" / name
        if src.exists():
            shutil.copy(src, out / name)
    shutil.copy("/opt/arena_ws/data/mf_nav/viz/overview/NAV100K_STATIC_100k_headon.png", out / "nav100k_static_headon.png")
    density_example(out)
    training_collapse(out)
    overviews(out)
    print("\n".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
