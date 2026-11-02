"""Render matched-eval traces as 2D trajectory videos and overview plots.

usage:
  render_episode.py video  TRACE_JSON EPISODE_INDEX OUT.mp4 [--stride 2] [--fps 10]
  render_episode.py overview TRACE_JSON OUT.png

Video: left, the world (robot footprint and heading, path so far, start, goal with its
tolerance circle, subgoal, and, when the trace has them, laser points in the world frame
and pedestrians with trails); right, time series synced to the frame: distance to the
final goal, commanded v and w, minimum laser range, and the cumulative reward per term.
Overview: every episode of one trace on one axis, coloured by outcome, plus a reward bar.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

FOOTPRINT = np.array([[0.229, 0.2078], [0.229, -0.2078], [-0.229, -0.2078], [-0.229, 0.2078]])
TERMS = ["RewardGoalReached", "RewardApproachGoal", "RewardSafeDistance", "RewardFactoredSafeDistance", "RewardCollision", "RewardPedestrianProximity"]
TERM_COLORS = {"RewardGoalReached": "#2a9d4b", "RewardApproachGoal": "#2364aa", "RewardSafeDistance": "#e09a1f", "RewardFactoredSafeDistance": "#b5651d", "RewardCollision": "#d1342f", "RewardPedestrianProximity": "#c2185b"}
COLORS = {"goal": "#2a9d4b", "collision": "#d1342f", "timeout": "#e09a1f", "tg_success_no_bonus": "#7d4fb3", "other": "#777777"}


def outcome(ep: dict) -> str:
    reason = str(ep["reason"])
    if ep.get("goal_bonus") and "SUCCESS" in reason:
        return "goal"
    if "COLLISION" in reason:
        return "collision"
    if "STEP_LIMIT" in reason:
        return "timeout"
    return "tg_success_no_bonus" if ep.get("success") else "other"


def yaw_of(series: list[dict]) -> np.ndarray:
    """Heading from the trace's robot pose if stored, else from the motion direction and command sign."""
    if series and series[0].get("yaw") is not None:
        return np.array([s["yaw"] for s in series])
    xy = np.array([s["robot"] for s in series])
    yaw = np.zeros(len(xy))
    last = 0.0
    for i in range(len(xy)):
        j = min(i + 1, len(xy) - 1)
        d = xy[j] - xy[max(i - 1, 0)]
        if np.hypot(*d) > 1e-3:
            heading = math.atan2(d[1], d[0])
            if series[i]["action"][0] < 0:  # driving backward: the body faces opposite the motion
                heading += math.pi
            last = heading
        yaw[i] = last
    return yaw


def step_terms(ep: dict) -> dict[str, np.ndarray]:
    """Per-step reward per term. The terminal step's values are recovered from the episode totals,
    because the vec env has already reset the reward function when the step snapshot is taken."""
    series = ep["series"]
    n = len(series)
    out = {}
    for term in TERMS:
        values = np.array([s["terms"].get(term, 0.0) for s in series], dtype=float)
        if n:
            values[-1] = ep["terms"].get(term, 0.0) - values[:-1].sum()
        if np.any(values) or term in ep["terms"]:
            out[term] = values
    return out


def bounds(ep: dict, pad: float = 2.0) -> tuple[float, float, float, float]:
    pts = [s["robot"] for s in ep["series"]] + [ep["record_goal_map"][:2]]
    for s in ep["series"]:
        for p in s.get("peds", []) or []:
            pts.append(p[:2])
    pts = np.array(pts)
    x0, y0 = pts.min(0) - pad
    x1, y1 = pts.max(0) + pad
    span = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2


def video(trace: dict, index: int, dest: Path, stride: int, fps: int) -> None:
    ep = trace["episodes"][index]
    series = ep["series"]
    n = len(series)
    xy = np.array([s["robot"] for s in series])
    yaw = yaw_of(series)
    goal = ep["record_goal_map"]
    tol = ep["record_tolerance"] or series[-1]["goal_radius"]
    t = np.arange(1, n + 1) / 10.0
    d = np.array([s.get("d_goal_true", np.nan) for s in series])
    v = np.array([s["action"][0] for s in series])
    w = np.array([s["action"][-1] for s in series])
    laser = np.array([s["min_laser"] for s in series])
    terms = step_terms(ep)
    cum = {k: np.cumsum(vals) for k, vals in terms.items()}
    total = np.sum(list(terms.values()), axis=0) if terms else np.zeros(n)
    kind = outcome(ep)
    name = Path(trace["checkpoint"]).parent.name
    tick = Path(trace["checkpoint"]).stem

    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    grid = fig.add_gridspec(4, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(grid[:, 0])
    ax_d, ax_v, ax_l, ax_r = (fig.add_subplot(grid[i, 1]) for i in range(4))
    fig.suptitle(
        f"{name} ({tick}), {trace['stage']}, episode {ep['episode_id']} (seed {ep['episode_seed']}): "
        f"{kind.upper()} after {ep['steps']} steps, return {ep['return']:.2f}",
        fontsize=11,
    )

    x0, x1, y0, y1 = bounds(ep)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (map)")
    ax.set_ylabel("y [m] (map)")
    ax.grid(alpha=0.25)
    ax.add_patch(Circle(goal[:2], tol, fill=False, ls="--", color=COLORS["goal"], lw=1.2))
    ax.plot(*goal[:2], marker="*", ms=16, color=COLORS["goal"], label=f"goal (tolerance {tol:.1f} m)")
    ax.plot(*xy[0], marker="o", ms=8, color="#333", label="start")
    path_line, = ax.plot([], [], color="#2364aa", lw=2, label="robot path")
    has_scan = any(s.get("scan_xy") for s in series)
    has_peds = any(s.get("peds") for s in series)
    scan_pts = ax.scatter([], [], s=3, c="#555", alpha=0.55, label="laser hits (walls, obstacles)" if has_scan else None)
    ped_pts = ax.scatter([], [], s=90, c="#c2185b", marker="o", label="pedestrian" if has_peds else None, zorder=5)
    ped_trails = []
    sub_pt, = ax.plot([], [], marker="x", ms=8, color="#888", ls="", label="subgoal")
    body = Polygon(FOOTPRINT, closed=True, fc="#2364aa", ec="k", alpha=0.85, zorder=6)
    ax.add_patch(body)
    heading = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.6}, zorder=7)
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
                   bbox={"boxstyle": "round", "fc": "white", "alpha": 0.85})
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    def series_axis(axis, ylabel, curves, hlines=()):
        axis.set_xlim(0, t[-1] if n else 1)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(alpha=0.25)
        for y, style in hlines:
            axis.axhline(y, **style)
        lines = []
        for values, style in curves:
            axis.plot(t, values, alpha=0.18, **{k: v for k, v in style.items() if k != "label"})
            line, = axis.plot([], [], **style)
            lines.append((line, values))
        lo = np.nanmin([np.nanmin(c[0]) for c in curves])
        hi = np.nanmax([np.nanmax(c[0]) for c in curves])
        margin = 0.08 * (hi - lo + 1e-6)
        axis.set_ylim(lo - margin, hi + margin)
        cursor = axis.axvline(0, color="k", lw=0.8)
        return lines, cursor

    d_lines, d_cur = series_axis(ax_d, "goal dist [m]", [(d, {"color": "#2364aa", "label": "d to final goal"})],
                                 [(tol, {"color": COLORS["goal"], "ls": "--", "lw": 1})])
    v_lines, v_cur = series_axis(ax_v, "command", [(v, {"color": "#2364aa", "label": "v [m/s]"}), (w, {"color": "#e07b1f", "label": "w [rad/s]"})],
                                 [(0, {"color": "#999", "lw": 0.8})])
    ax_v.legend(fontsize=7, loc="upper right", ncol=2)
    l_lines, l_cur = series_axis(ax_l, "min laser [m]", [(laser, {"color": "#555"})],
                                 [(1.0, {"color": "#e09a1f", "ls": "--", "lw": 1}), (0.267, {"color": COLORS["collision"], "ls": ":", "lw": 1})])
    term_curves = [(cum[k], {"label": k.replace("Reward", ""), "lw": 1.4, "color": TERM_COLORS[k]}) for k in cum]
    term_curves.append((np.cumsum(total), {"label": "total", "color": "k", "lw": 2}))
    r_lines, r_cur = series_axis(ax_r, "cum. reward", term_curves)
    ax_r.set_xlabel("time [s]")
    ax_r.legend(fontsize=6.5, loc="upper left", ncol=3)

    frames = list(range(0, n, stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    frames += [n - 1] * fps  # hold the last frame for a second

    writer = FFMpegWriter(fps=fps, bitrate=2400)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(dest), dpi=100):
        for i in frames:
            s = series[i]
            path_line.set_data(xy[: i + 1, 0], xy[: i + 1, 1])
            c, sn = math.cos(yaw[i]), math.sin(yaw[i])
            rot = np.array([[c, -sn], [sn, c]])
            body.set_xy(FOOTPRINT @ rot.T + xy[i])
            heading.xy = (xy[i][0] + 0.6 * c, xy[i][1] + 0.6 * sn)
            heading.set_position(xy[i])
            if s.get("subgoal"):
                sub_pt.set_data([s["subgoal"][0]], [s["subgoal"][1]])
            if s.get("scan_xy"):
                scan_pts.set_offsets(np.array(s["scan_xy"]))
            peds = s.get("peds") or []
            ped_pts.set_offsets(np.array([p[:2] for p in peds]) if peds else np.empty((0, 2)))
            for line in ped_trails:
                line.remove()
            ped_trails.clear()
            if peds:
                for k in range(len(peds)):
                    trail = np.array([series[j]["peds"][k][:2] for j in range(max(0, i - 40), i + 1) if len(series[j].get("peds") or []) > k])
                    if len(trail):
                        line, = ax.plot(trail[:, 0], trail[:, 1], color="#c2185b", lw=1, alpha=0.6)
                        ped_trails.append(line)
            info.set_text(
                f"t = {t[i]:5.1f} s   step {i + 1}/{n}\n"
                f"d goal = {d[i]:5.2f} m\n"
                f"v = {v[i]:+.2f} m/s  w = {w[i]:+.2f} rad/s\n"
                f"min laser = {laser[i]:.2f} m\n"
                + (f"ped dist = {min(math.hypot(p[0] - xy[i][0], p[1] - xy[i][1]) for p in s['peds']):5.2f} m\n" if s.get("peds") else "")
                + f"return so far = {np.sum(total[: i + 1]):+.2f}"
            )
            for lines, cursor in ((d_lines, d_cur), (v_lines, v_cur), (l_lines, l_cur), (r_lines, r_cur)):
                for line, values in lines:
                    line.set_data(t[: i + 1], values[: i + 1])
                cursor.set_xdata([t[i], t[i]])
            writer.grab_frame()
    plt.close(fig)


def overview(trace: dict, dest: Path) -> None:
    eps = trace["episodes"]
    name = Path(trace["checkpoint"]).parent.name
    tick = Path(trace["checkpoint"]).stem
    cols = min(5, len(eps))
    rows = math.ceil(len(eps) / cols)
    fig, axes = plt.subplots(rows + 1, cols, figsize=(3.6 * cols, 3.6 * rows + 3.2), dpi=100,
                             gridspec_kw={"height_ratios": [1] * rows + [0.9]})
    axes = np.atleast_2d(axes)
    counts = {}
    for k, ep in enumerate(eps):
        axis = axes[k // cols, k % cols]
        kind = outcome(ep)
        counts[kind] = counts.get(kind, 0) + 1
        xy = np.array([s["robot"] for s in ep["series"]])
        goal = ep["record_goal_map"]
        tol = ep["record_tolerance"] or 1.0
        scan = [p for s in ep["series"][::5] for p in (s.get("scan_xy") or [])]
        if scan:
            scan = np.array(scan)
            axis.scatter(scan[:, 0], scan[:, 1], s=1, c="#999", alpha=0.3)
        for j in range(len(ep["series"][0].get("peds") or [])):
            trail = np.array([s["peds"][j][:2] for s in ep["series"] if len(s.get("peds") or []) > j], dtype=float)
            if len(trail) > 1:
                # Respawns and reordered detections show up as jumps; do not draw them as walking.
                jumps = np.flatnonzero(np.hypot(*np.diff(trail, axis=0).T) > 1.0) + 1
                trail = np.insert(trail, jumps, np.nan, axis=0)
            axis.plot(trail[:, 0], trail[:, 1], color="#c2185b", lw=0.8, alpha=0.5)
        axis.add_patch(Circle(goal[:2], tol, fill=False, ls="--", color=COLORS["goal"]))
        axis.plot(*goal[:2], "*", ms=12, color=COLORS["goal"])
        rev = np.array([s["action"][0] < -0.05 for s in ep["series"]])
        axis.plot(xy[:, 0], xy[:, 1], color=COLORS[kind], lw=1.6)
        if rev.any():
            axis.plot(np.where(rev, xy[:, 0], np.nan), np.where(rev, xy[:, 1], np.nan), color="k", lw=0.6, ls=":")
        axis.plot(*xy[0], "o", color="#333", ms=5)
        axis.plot(*xy[-1], "X" if kind == "collision" else "s", color=COLORS[kind], ms=8)
        x0, x1, y0, y1 = bounds(ep, pad=1.5)
        axis.set_xlim(x0, x1)
        axis.set_ylim(y0, y1)
        axis.set_aspect("equal")
        axis.tick_params(labelsize=7)
        axis.set_title(f"ep {ep['episode_id']}: {kind} ({ep['steps']} st, R={ep['return']:.1f})", fontsize=8.5, color=COLORS[kind])
    for k in range(len(eps), rows * cols):
        axes[k // cols, k % cols].axis("off")
    bar = fig.add_subplot(axes[rows, 0].get_gridspec()[rows, :])
    for axis in axes[rows]:
        axis.remove()
    present = [term for term in TERMS if any(term in ep["terms"] for ep in eps)]
    width = 0.8 / max(1, len(present))
    x = np.arange(len(eps))
    for j, term in enumerate(present):
        bar.bar(x + j * width - 0.4 + width / 2, [ep["terms"].get(term, 0.0) for ep in eps], width, label=term.replace("Reward", ""), color=TERM_COLORS[term])
    bar.plot(x, [ep["return"] for ep in eps], "k_", ms=18, mew=2, label="return")
    bar.axhline(0, color="#666", lw=0.8)
    bar.set_xticks(x)
    bar.set_xticklabels([f"ep {ep['episode_id']}" for ep in eps], fontsize=8)
    bar.set_ylabel("episode reward by term")
    bar.legend(fontsize=7.5, ncol=len(present) + 1, loc="lower left")
    bar.grid(axis="y", alpha=0.25)
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    fig.suptitle(f"{name} ({tick}) on {trace['stage']}, run_seed {trace['run_seed']}: {summary}\n"
                 "path colour = outcome; dotted black = driving backward; X = collision; square = end", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["video", "overview"])
    parser.add_argument("trace")
    parser.add_argument("rest", nargs="+")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text())
    if args.mode == "video":
        video(trace, int(args.rest[0]), Path(args.rest[1]), args.stride, args.fps)
    else:
        overview(trace, Path(args.rest[0]))


if __name__ == "__main__":
    main()
