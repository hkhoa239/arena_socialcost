"""Phase 1 forecasting benchmark on held-out closed-loop PedVO traces.

Data: trace_eval.py rollouts on the fc_* stages (run_seed "bench-fc-<stage>"), disjoint from
the residual's training seeds (5, 6 of the scripted dataset) and from the navigation
validation/test seeds. The robot is driven by a navigation policy, so the commands are the
closed-loop commands the forecast sees inside PPO.

For every step t with the same pedestrians present at t and t+10 (1 s at 10 Hz):
  target       actual pedestrians at t+10, rasterized in the actual robot frame at t+10
  persistence  pedestrians at t in the frame at t
  online_cv    forecast_density(apply_correction=False): pedestrians at constant velocity,
               robot pose by holding the executed command cmd[t] for 1 s (the agents' input)
  residual     forecast_density(apply_correction=True) with the frozen 211-parameter residual
  cnn          conditioned CrowdResponseNet(rho_t, rho_{t-1}, cmd[t])
  oracle_cv    constant-velocity pedestrians rasterized in the actual frame at t+10 (diagnostic)

Density grid 16x16, ROI 20 m, sigma 0.5 m (the agents' constants). MSE is pooled over cells
and pairs; confidence intervals are paired bootstrap over episodes.

usage: forecast_bench.py OUT_JSON TRACE_JSON [TRACE_JSON ...]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.density import rasterize_density
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.interaction import (
    InteractionResidual,
    forecast_density,
    integrate_unicycle,
    residual_checkpoint,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.response_model import CrowdResponseNet

HORIZON = 10
NEAR = (2.0, 3.0)
ACTIVE_DIST = 3.0
ACTIVE_DEV = 0.30
BOOTSTRAP = 2000
CNN = Path("/opt/arena_ws/data/mf_pedvo/conditioned.pt")
METHODS = ("persistence", "online_cv", "residual", "cnn", "oracle_cv")
COORDS = (np.arange(16) - 8.0) * (20.0 / 16.0)
CELL_DIST = np.hypot(COORDS[None, :], COORDS[:, None])


def to_frame(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    dx, dy = points[..., 0] - x, points[..., 1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack((c * dx + s * dy, -s * dx + c * dy), axis=-1)


def episode_arrays(ep: dict):
    """Steps with a consistent pedestrian set, as arrays aligned by name."""
    series = ep["series"]
    if not series or not series[0].get("ped_names") and not any(s.get("ped_names") for s in series):
        return None
    names = max((tuple(s.get("ped_names") or ()) for s in series), key=len)
    if not names:
        return None
    rows = []
    for s in series:
        if tuple(s.get("ped_names") or ()) != names or s.get("yaw") is None or s.get("cmd") is None:
            rows.append(None)
            continue
        rows.append(s)
    return names, rows


def pairs_of(ep: dict, residual: InteractionResidual, cnn: CrowdResponseNet) -> dict | None:
    got = episode_arrays(ep)
    if got is None:
        return None
    names, rows = got
    n = len(rows)
    idx = [t for t in range(1, n - HORIZON) if rows[t] is not None and rows[t - 1] is not None and rows[t + HORIZON] is not None]
    if not idx:
        return None
    count = len(names)
    robot = np.array([rows[t]["robot"] for t in idx])
    yaw = np.array([rows[t]["yaw"] for t in idx])
    ped = np.array([rows[t]["peds"] for t in idx], dtype=np.float64)
    vel = np.array([rows[t]["ped_vel"] for t in idx], dtype=np.float64)
    cmd = np.array([[rows[t]["cmd"][0], rows[t]["cmd"][-1]] for t in idx], dtype=np.float64)
    fut_robot = np.array([rows[t + HORIZON]["robot"] for t in idx])
    fut_yaw = np.array([rows[t + HORIZON]["yaw"] for t in idx])
    fut_ped = np.array([rows[t + HORIZON]["peds"] for t in idx], dtype=np.float64)
    prev_robot = np.array([rows[t - 1]["robot"] for t in idx])
    prev_yaw = np.array([rows[t - 1]["yaw"] for t in idx])
    prev_ped = np.array([rows[t - 1]["peds"] for t in idx], dtype=np.float64)

    rel = np.stack([to_frame(ped[i], *robot[i], yaw[i]) for i in range(len(idx))])
    rel_prev = np.stack([to_frame(prev_ped[i], *prev_robot[i], prev_yaw[i]) for i in range(len(idx))])
    rel_future = np.stack([to_frame(fut_ped[i], *fut_robot[i], fut_yaw[i]) for i in range(len(idx))])
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    vel_rob = np.stack((c * vel[..., 0] + s * vel[..., 1], -s * vel[..., 0] + c * vel[..., 1]), axis=-1)
    heading = np.stack((np.cos(yaw), -np.sin(yaw)), axis=1)

    target = rasterize_density(rel_future)[:, 0]
    persistence = rasterize_density(rel)[:, 0]
    rho_prev = rasterize_density(rel_prev)[:, 0]
    cv_world = ped + vel * 1.0
    oracle = rasterize_density(np.stack([to_frame(cv_world[i], *fut_robot[i], fut_yaw[i]) for i in range(len(idx))]))[:, 0]

    tp = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    valid = torch.ones(len(idx), count, dtype=torch.bool)
    with torch.no_grad():
        online = forecast_density(None, tp(rel), tp(vel_rob), valid, tp(cmd), apply_correction=False, slow_heading=tp(heading)).numpy()
        corrected = forecast_density(residual, tp(rel), tp(vel_rob), valid, tp(cmd), apply_correction=True, slow_heading=tp(heading)).numpy()
        cnn_pred = cnn(tp(persistence[:, None]), tp(rho_prev[:, None]), tp(cmd))[:, 0].numpy()

    # Interaction masks, per pair.
    dist_now = np.linalg.norm(rel, axis=-1)
    cv_future_world = ped + vel * 1.0
    dev = fut_ped - cv_future_world
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    fwd = np.where(speed > 0.2, vel / np.maximum(speed, 1e-6), 0.0)
    lateral_dev = np.abs(fwd[..., 0] * dev[..., 1] - fwd[..., 1] * dev[..., 0])
    near = dist_now < ACTIVE_DIST
    active = np.any(near & (lateral_dev > ACTIVE_DEV) & (speed[..., 0] > 0.2), axis=1)
    any_near = np.any(near, axis=1)
    # Pose error of holding the command for 1 s.
    px, py, pyaw = integrate_unicycle(torch.zeros(len(idx)), torch.zeros(len(idx)), torch.zeros(len(idx)), tp(cmd[:, 0]), tp(cmd[:, 1]), torch.ones(len(idx)))
    true_rel = np.stack([to_frame(fut_robot[i][None], *robot[i], yaw[i])[0] for i in range(len(idx))])
    pose_err = np.hypot(px.numpy() - true_rel[:, 0], py.numpy() - true_rel[:, 1])
    return {
        "target": target,
        "persistence": persistence,
        "online_cv": online,
        "residual": corrected,
        "cnn": cnn_pred,
        "oracle_cv": oracle,
        "active": active,
        "near": any_near,
        "pose_err": pose_err,
        "count": count,
    }


def sse(pred: np.ndarray, target: np.ndarray, rows: np.ndarray, cells: np.ndarray | None = None) -> tuple[float, float]:
    sq = (pred[rows] - target[rows]) ** 2
    if cells is not None:
        sq = sq[:, cells]
    return float(sq.sum()), float(sq.size)


def cell_mask(radius: float | None) -> np.ndarray | None:
    return None if radius is None else (CELL_DIST <= radius)


def summarize(parts: list[dict], rng: np.random.Generator) -> dict:
    regions = {"all": None, "near2m": cell_mask(NEAR[0]), "near3m": cell_mask(NEAR[1])}
    subsets = {"all_pairs": "all", "active_avoidance": "active", "robot_near_ped": "near"}
    out = {"episodes": len(parts), "pairs": int(sum(len(p["target"]) for p in parts)),
           "active_pairs": int(sum(p["active"].sum() for p in parts)),
           "mean_pose_err_m": float(np.mean(np.concatenate([p["pose_err"] for p in parts]))) if parts else None}
    for sub_name, key in subsets.items():
        for reg_name, cells in regions.items():
            label = f"{sub_name}/{reg_name}"
            table = {m: np.zeros((len(parts), 2)) for m in METHODS}
            for e, p in enumerate(parts):
                rows = np.ones(len(p["target"]), bool) if key == "all" else p[key]
                if not rows.any():
                    continue
                for m in METHODS:
                    table[m][e] = sse(p[m], p["target"], rows, cells)
            keep = table[METHODS[0]][:, 1] > 0
            if not keep.any():
                out[label] = {"pairs_episodes": 0}
                continue
            cell = {m: float(table[m][keep, 0].sum() / table[m][keep, 1].sum()) for m in METHODS}
            draws = rng.integers(0, keep.sum(), size=(BOOTSTRAP, keep.sum()))
            for a, b in (("residual", "online_cv"), ("online_cv", "persistence"), ("residual", "cnn"), ("online_cv", "oracle_cv")):
                ta, tb = table[a][keep], table[b][keep]
                deltas = ta[draws, 0].sum(1) / ta[draws, 1].sum(1) - tb[draws, 0].sum(1) / tb[draws, 1].sum(1)
                cell[f"{a}_minus_{b}"] = {"mean": cell[a] - cell[b], "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}
            cell["episodes_with_pairs"] = int(keep.sum())
            out[label] = cell
    return out


def latency(residual: InteractionResidual, repeats: int = 200) -> dict:
    res = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        mod = residual.to(device)
        for count in (1, 10, 20):
            pos = torch.randn(1, count, 2, device=device) * 4
            vel = torch.randn(1, count, 2, device=device)
            valid = torch.ones(1, count, dtype=torch.bool, device=device)
            cmd = torch.tensor([[0.4, 0.2]], device=device)
            head = torch.tensor([[1.0, 0.0]], device=device)
            for correct in (False, True):
                with torch.no_grad():
                    for _ in range(10):
                        forecast_density(mod, pos, vel, valid, cmd, apply_correction=correct, slow_heading=head)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(repeats):
                        forecast_density(mod, pos, vel, valid, cmd, apply_correction=correct, slow_heading=head)
                    if device == "cuda":
                        torch.cuda.synchronize()
                res[f"{device}/n{count}/{'residual' if correct else 'online_cv'}_ms"] = (time.perf_counter() - start) / repeats * 1000
        residual.to("cpu")
    return res


def main() -> None:
    dest = Path(sys.argv[1])
    residual = InteractionResidual()
    residual.load_state_dict(torch.load(residual_checkpoint(), map_location="cpu", weights_only=True))
    residual.eval()
    cnn = CrowdResponseNet(conditioned=True)
    cnn.load_state_dict(torch.load(CNN, map_location="cpu"))
    cnn.eval()
    rng = np.random.default_rng(5)
    by_stage: dict[str, list[dict]] = {}
    for path in sys.argv[2:]:
        data = json.loads(Path(path).read_text())
        for ep in data["episodes"]:
            p = pairs_of(ep, residual, cnn)
            if p is not None:
                by_stage.setdefault(data["stage"], []).append(p)
    result = {"stages": {}, "by_count": {}, "latency": latency(residual)}
    for stage, parts in sorted(by_stage.items()):
        result["stages"][stage] = summarize(parts, rng)
    counts = sorted({p["count"] for parts in by_stage.values() for p in parts})
    for count in counts:
        parts = [p for stage, ps in by_stage.items() if stage.startswith("fc_n") and "static" not in stage for p in ps if p["count"] == count]
        if parts:
            result["by_count"][f"n{count}"] = summarize(parts, rng)
    everything = [p for ps in by_stage.values() for p in ps]
    result["all"] = summarize(everything, rng)
    dest.write_text(json.dumps(result, indent=1))
    print(f"WROTE {dest}")
    for stage, cell in [("ALL", result["all"]), *result["stages"].items()]:
        c = cell.get("all_pairs/all", {})
        if "online_cv" in c:
            print(f"{stage:14s} pairs={cell['pairs']:6d} active={cell['active_pairs']:5d} " + " ".join(f"{m}={c[m]:.5f}" for m in METHODS)
                  + f"  res-cv={c['residual_minus_online_cv']['mean']:+.5f} {np.round(c['residual_minus_online_cv']['ci95'], 5).tolist()}")


if __name__ == "__main__":
    main()
