"""Paired differences for the 50k matched PedVO evaluation.

Episode-cluster bootstrap resamples episode indices inside each stage.
A pooled interval resamples those indices independently in every stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/opt/arena_ws/data/mf_eval")
AGENTS = ("reference", "AGENT_5", "AGENT_MF_ONLINE", "AGENT_MF_RESIDUAL")
STAGES = ("headon", "peds_5", "peds_15", "peds_20")
METRICS = ("success", "zero_contact_success", "ped_contacts", "time_to_goal_s", "ped_deviation_m")
DRAWS = 4000
SEED = 5


def _load(agent: str, stage: str, mode: str = "") -> dict | None:
    suffix = f"_{mode}" if mode else ""
    path = ROOT / f"{agent}_{stage}{suffix}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _deviation(agent_paths: dict, reference_paths: dict) -> float | None:
    shared = sorted(set(agent_paths) & set(reference_paths))
    if not shared:
        return None
    distances = []
    for name in shared:
        agent = np.asarray(agent_paths[name], dtype=np.float64)
        reference = np.asarray(reference_paths[name], dtype=np.float64)
        if len(agent) == 0 or len(reference) == 0:
            continue
        nearest = np.linalg.norm(agent[:, None, :] - reference[None, :, :], axis=2).min(axis=1)
        distances.append(float(nearest.mean()))
    if not distances:
        return None
    return float(np.mean(distances))


def _rows(agent: str, stage: str, mode: str = "") -> list[dict] | None:
    payload = _load(agent, stage, mode)
    reference = _load("reference", stage)
    if payload is None or reference is None:
        return None
    reference_by_index = {episode["index"]: episode for episode in reference["episodes"]}
    rows = []
    for episode in payload["episodes"]:
        other = reference_by_index.get(episode["index"], {})
        rows.append({
            "index": episode["index"],
            "success": float(episode["success"]),
            "zero_contact_success": float(episode["zero_contact_success"]),
            "ped_contacts": float(episode["ped_contacts"]),
            "time_to_goal_s": episode["time_to_goal_s"],
            "ped_deviation_m": _deviation(episode.get("paths", {}), other.get("paths", {})),
            "reason": episode["reason"],
            "steps": episode["steps"],
        })
    return rows


def _pair(left: list[dict], right: list[dict], metric: str) -> np.ndarray:
    right_by_index = {row["index"]: row for row in right}
    values = []
    for row in left:
        other = right_by_index.get(row["index"])
        if other is None:
            continue
        a, b = row[metric], other[metric]
        if a is None or b is None:
            continue
        if metric == "time_to_goal_s" and (row["success"] < 1 or other["success"] < 1):
            continue
        values.append((row["index"], float(a) - float(b)))
    return np.asarray(values, dtype=np.float64)


def _interval(samples: np.ndarray) -> dict:
    if len(samples) == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    return {
        "n": int(len(samples)),
        "mean": float(samples.mean()),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
    }


def _bootstrap(pairs_by_stage: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    stage_names = [name for name, pairs in pairs_by_stage.items() if len(pairs)]
    draws = []
    per_stage = {}
    for name in stage_names:
        values = pairs_by_stage[name][:, 1]
        choice = rng.integers(0, len(values), size=(DRAWS, len(values)))
        stage_draws = values[choice].mean(axis=1)
        per_stage[name] = _interval(stage_draws)
        per_stage[name]["point"] = float(values.mean())
        per_stage[name]["episodes"] = int(len(values))
        draws.append(stage_draws)
    pooled = np.mean(np.vstack(draws), axis=0) if draws else np.asarray([])
    result = {"pooled": _interval(pooled), "stages": per_stage}
    if draws:
        result["pooled"]["point"] = float(np.mean([per_stage[name]["point"] for name in stage_names]))
    return result


def main() -> None:
    rng = np.random.default_rng(SEED)
    report = {"stages": {}, "differences": {}, "ablation": {}, "residual": []}
    for stage in STAGES:
        report["stages"][stage] = {}
        for agent in AGENTS:
            rows = _rows(agent, stage)
            if rows is None:
                continue
            report["stages"][stage][agent] = {
                metric: None if any(row[metric] is None for row in rows) and metric == "ped_deviation_m"
                else float(np.mean([row[metric] for row in rows if row[metric] is not None]))
                for metric in METRICS
            }
            report["stages"][stage][agent]["episodes"] = len(rows)
            report["stages"][stage][agent]["reasons"] = [row["reason"] for row in rows]
    for candidate in ("AGENT_MF_ONLINE", "AGENT_MF_RESIDUAL"):
        report["differences"][candidate] = {}
        for baseline in ("AGENT_5", "AGENT_MF_ONLINE"):
            if candidate == baseline:
                continue
            report["differences"][candidate][baseline] = {}
            for metric in METRICS:
                pairs = {}
                for stage in STAGES:
                    left, right = _rows(candidate, stage), _rows(baseline, stage)
                    if left is None or right is None:
                        continue
                    pairs[stage] = _pair(left, right, metric)
                report["differences"][candidate][baseline][metric] = _bootstrap(pairs, rng)
    for mode in ("", "online", "zero"):
        label = mode or "residual"
        report["ablation"][label] = {}
        for stage in ("headon", "peds_5"):
            rows = _rows("AGENT_MF_RESIDUAL", stage, mode)
            if rows is None:
                continue
            report["ablation"][label][stage] = {
                metric: float(np.mean([row[metric] for row in rows if row[metric] is not None]))
                for metric in METRICS
            }
        native = _load("AGENT_MF_RESIDUAL", "headon") or _load("AGENT_MF_RESIDUAL", "peds_5")
        if native and "action_sensitivity" in native:
            report["ablation"]["action_sensitivity"] = native["action_sensitivity"]
    for stage in STAGES:
        payload = _load("AGENT_MF_RESIDUAL", stage)
        if payload and "residual_frozen" in payload:
            report["residual"].append({"stage": stage, **payload["residual_frozen"]})
            if "action_sensitivity" in payload:
                report["ablation"].setdefault("action_sensitivity_by_stage", {})[stage] = payload["action_sensitivity"]
    out = ROOT / "matched_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(out)


if __name__ == "__main__":
    main()
