"""Offline PedVO forecast comparison. Hyperparameters are fixed before evaluation.

Both networks are the existing CrowdResponseNet. The unconditioned variant
zeros the velocity channels inside that same module. Training uses seeds 5
and 6. Seed 7 is validation for the loss curve only. Seeds 8, 9, and 10 are
held out and are not used to stop or select anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.density import (
    rasterize_density,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.response_model import (
    CrowdResponseNet,
)
from rosnav_rl.observations.utils.semantic import get_relative_pos_to_robot

HORIZON = 10
SAMPLE_HZ = 10.0
DT = HORIZON / SAMPLE_HZ
EPOCHS = 40
BATCH = 64
LR = 1e-3
TORCH_SEED = 5
TRAIN_SEEDS = {5, 6}
VAL_SEEDS = {7}
TEST_SEEDS = {8, 9, 10}
LATERAL_ON = 0.30
LATERAL_OFF = 0.10
DATA = Path("/opt/arena_ws/data/mf_pedvo")
OUT = Path("/opt/arena_ws/data/mf_pedvo_forecast.json")
BOOTSTRAP = 1000


def relatives(robot: np.ndarray, yaw: np.ndarray, ped: np.ndarray) -> np.ndarray:
    out = np.empty_like(ped)
    ones = np.ones((ped.shape[1], 1), dtype=np.float64)
    for t in range(len(ped)):
        homo = np.concatenate([ped[t], ones], axis=1)
        pose = {"x": float(robot[t, 0]), "y": float(robot[t, 1]), "yaw": float(yaw[t])}
        out[t] = get_relative_pos_to_robot(pose, homo)
    return out


def load_episodes() -> list[dict]:
    episodes = []
    for path in sorted(DATA.glob("n*/*.npz")):
        count = int(path.parent.name[1:])
        seed_text, name, repeat_text = path.stem.split("_")
        blob = np.load(path)
        ped = blob["ped"]
        if len(ped) < HORIZON + 2:
            continue
        robot = blob["robot"]
        yaw = blob["yaw"]
        relative = relatives(robot, yaw, ped)
        density = rasterize_density(relative)[:, 0]
        center = int(np.argmin(np.abs(ped[0, :, 1] - robot[0, 1])))
        episodes.append(
            {
                "count": count,
                "seed": int(seed_text[4:]),
                "name": name,
                "repeat": int(repeat_text),
                "ped": ped,
                "vel": blob["vel"],
                "robot": robot,
                "yaw": yaw,
                "command": blob["command"],
                "relative": relative,
                "density": density,
                "center": center,
                "spawn_y": ped[0, :, 1].copy(),
            }
        )
    return episodes


def pairs_of(episode: dict) -> dict:
    density = episode["density"]
    ped = episode["ped"]
    vel = episode["vel"]
    robot = episode["robot"]
    yaw = episode["yaw"]
    index = np.arange(1, len(density) - HORIZON)
    future = ped[index] + vel[index] * DT
    extra = np.empty((len(index),) + density.shape[1:], dtype=np.float32)
    ones = np.ones((ped.shape[1], 1), dtype=np.float64)
    for row, t in enumerate(index):
        homo = np.concatenate([future[row], ones], axis=1)
        pose = {"x": float(robot[t + HORIZON, 0]), "y": float(robot[t + HORIZON, 1]), "yaw": float(yaw[t + HORIZON])}
        extra[row] = rasterize_density(get_relative_pos_to_robot(pose, homo))[0, 0]
    center = episode["center"]
    current_lat = np.abs(ped[index, center, 1] - episode["spawn_y"][center])
    future_lat = np.abs(ped[index + HORIZON, center, 1] - episode["spawn_y"][center])
    others = [i for i in range(ped.shape[1]) if i != center]
    if others:
        neighbor_lat = np.max(np.abs(ped[index + HORIZON][:, others, 1] - episode["spawn_y"][others]), axis=1)
    else:
        neighbor_lat = np.zeros(len(index))
    distance = np.linalg.norm(episode["relative"][index, center], axis=1)
    closest = np.linalg.norm(episode["relative"][index], axis=2).min(axis=1)
    return {
        "rho_t": density[index],
        "rho_prev": density[index - 1],
        "target": density[index + HORIZON],
        "action": episode["command"][index - 1],
        "extra": extra,
        "current_lat": current_lat,
        "future_lat": future_lat,
        "neighbor_lat": neighbor_lat,
        "distance": distance,
        "closest": closest,
        "count": np.full(len(index), episode["count"]),
        "name": np.array([episode["name"]] * len(index)),
    }


def stack(episodes: list[dict]) -> tuple[dict, list[int]]:
    parts = [pairs_of(episode) for episode in episodes]
    keys = [key for key in parts[0] if key != "name"]
    data = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    data["name"] = np.concatenate([part["name"] for part in parts])
    bounds = np.cumsum([0] + [len(part["distance"]) for part in parts])
    return data, bounds


def masks(data: dict) -> dict[str, np.ndarray]:
    distance = data["closest"]
    return {
        "all": np.ones(len(distance), dtype=bool),
        "robot_induced": (data["future_lat"] > LATERAL_ON) & (data["distance"] < 3.0),
        "anticipation": (data["current_lat"] < LATERAL_OFF) & (data["future_lat"] > LATERAL_ON) & (data["distance"] < 4.0),
        "neighbor_only": (data["neighbor_lat"] > LATERAL_ON) & (data["future_lat"] < LATERAL_OFF),
        "straight": (data["future_lat"] < LATERAL_OFF) & (data["neighbor_lat"] < LATERAL_OFF),
        "distance_0_1.5": distance < 1.5,
        "distance_1.5_3": (distance >= 1.5) & (distance < 3.0),
        "distance_3_6": (distance >= 3.0) & (distance < 6.0),
        "distance_beyond_6": distance >= 6.0,
        **{f"n{count}": data["count"] == count for count in (1, 5, 15, 20)},
        **{f"cmd_{name}": data["name"] == name for name in ("block", "leave", "approach", "slow", "cruise", "yaw")},
    }


def train_net(conditioned: bool, data: dict, val: dict) -> tuple[CrowdResponseNet, list[float], list[float]]:
    torch.manual_seed(TORCH_SEED)
    net = CrowdResponseNet(conditioned=conditioned)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    tensors = {
        key: torch.as_tensor(data[key][:, None] if key != "action" else data[key], dtype=torch.float32)
        for key in ("rho_t", "rho_prev", "action", "target")
    }
    val_t = {
        key: torch.as_tensor(val[key][:, None] if key != "action" else val[key], dtype=torch.float32)
        for key in ("rho_t", "rho_prev", "action", "target")
    }
    generator = torch.Generator().manual_seed(TORCH_SEED)
    train_curve, val_curve = [], []
    steps = int(np.ceil(len(data["action"]) / BATCH))
    for _epoch in range(EPOCHS):
        net.train()
        order = torch.randperm(len(data["action"]), generator=generator)
        total = 0.0
        for start in range(0, len(order), BATCH):
            choice = order[start : start + BATCH]
            prediction = net(tensors["rho_t"][choice], tensors["rho_prev"][choice], tensors["action"][choice])
            loss = torch.mean((prediction - tensors["target"][choice]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(choice)
        train_curve.append(total / len(data["action"]))
        net.eval()
        with torch.no_grad():
            held = net(val_t["rho_t"], val_t["rho_prev"], val_t["action"])
            val_curve.append(float(torch.mean((held - val_t["target"]) ** 2)))
        print(f"conditioned={conditioned} epoch {_epoch + 1} train {train_curve[-1]:.6f} val {val_curve[-1]:.6f}", flush=True)
    return net, train_curve, val_curve


def predict(net: CrowdResponseNet, data: dict, action: np.ndarray | None = None) -> np.ndarray:
    used = data["action"] if action is None else action
    with torch.no_grad():
        prediction = net(
            torch.as_tensor(data["rho_t"][:, None], dtype=torch.float32),
            torch.as_tensor(data["rho_prev"][:, None], dtype=torch.float32),
            torch.as_tensor(used, dtype=torch.float32),
        )
    return prediction[:, 0].numpy()


def episode_sse(prediction: np.ndarray, target: np.ndarray, bounds: list[int], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared = (prediction - target) ** 2
    sse = np.zeros(len(bounds) - 1)
    count = np.zeros(len(bounds) - 1)
    for episode, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        chosen = mask[start:end]
        if not np.any(chosen):
            continue
        block = squared[start:end][chosen]
        sse[episode] = float(block.sum())
        count[episode] = float(block.size)
    return sse, count


def pooled(sse: np.ndarray, count: np.ndarray, draw: np.ndarray | None = None) -> float:
    if draw is None:
        total_sse = float(sse.sum())
        total_count = float(count.sum())
    else:
        total_sse = float(sse[draw].sum())
        total_count = float(count[draw].sum())
    if total_count == 0:
        return float("nan")
    return total_sse / total_count


def bootstrap(reference: tuple[np.ndarray, np.ndarray], others: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    sse, count = reference
    present = count > 0
    base = pooled(sse, count)
    result = {"mse": base, "episodes": int(present.sum()), "elements": int(count.sum())}
    if present.sum() < 2:
        for name in others:
            result[name] = {"mean_delta": float("nan"), "ci95": [float("nan"), float("nan")]}
        return result
    rng = np.random.default_rng(0)
    usable = np.flatnonzero(present)
    draws = usable[rng.integers(0, len(usable), size=(BOOTSTRAP, len(usable)))]
    for name, (other_sse, other_count) in others.items():
        delta = np.empty(BOOTSTRAP)
        for row, draw in enumerate(draws):
            delta[row] = pooled(sse, count, draw) - pooled(other_sse, other_count, draw)
        result[name] = {
            "mean_delta": float(delta.mean()),
            "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        }
    return result


def com_shift(density: np.ndarray, origin: np.ndarray) -> np.ndarray:
    resolution = 20.0 / density.shape[-1]
    coords = (np.arange(density.shape[-1]) - density.shape[-1] // 2) * resolution
    mass = density.sum(axis=(1, 2))
    safe = np.maximum(mass, 1e-8)
    cy = (density * coords[None, :, None]).sum(axis=(1, 2)) / safe
    oy = (origin * coords[None, :, None]).sum(axis=(1, 2)) / np.maximum(origin.sum(axis=(1, 2)), 1e-8)
    return cy - oy


def main() -> None:
    episodes = load_episodes()
    groups = {
        "train": [ep for ep in episodes if ep["seed"] in TRAIN_SEEDS],
        "val": [ep for ep in episodes if ep["seed"] in VAL_SEEDS],
        "test": [ep for ep in episodes if ep["seed"] in TEST_SEEDS],
    }
    data = {}
    bounds = {}
    for split, chosen in groups.items():
        data[split], bounds[split] = stack(chosen)
        print(f"{split} episodes {len(chosen)} pairs {len(data[split]['distance'])}", flush=True)

    conditioned, train_c, val_c = train_net(True, data["train"], data["val"])
    unconditioned, train_u, val_u = train_net(False, data["train"], data["val"])
    torch.save(conditioned.state_dict(), DATA / "conditioned.pt")
    torch.save(unconditioned.state_dict(), DATA / "unconditioned.pt")

    test = data["test"]
    predictions = {
        "persistence": test["rho_t"],
        "motion_extrapolation": test["extra"],
        "conditioned": predict(conditioned, test),
        "unconditioned": predict(unconditioned, test),
        "conditioned_action_zeroed": predict(conditioned, test, np.zeros_like(test["action"])),
    }
    mean_density = data["train"]["target"].mean(axis=0, keepdims=True)
    predictions["train_mean_density"] = np.repeat(mean_density, len(test["target"]), axis=0)

    cells = {}
    for label, mask in masks(test).items():
        packed = {
            name: episode_sse(prediction, test["target"], bounds["test"], mask)
            for name, prediction in predictions.items()
        }
        cell = {"pairs": int(mask.sum())}
        for name, parts in packed.items():
            cell[name] = pooled(*parts)
        if mask.sum():
            cell["conditioned_vs_unconditioned"] = bootstrap(packed["conditioned"], {"delta": packed["unconditioned"]})["delta"]
            cell["conditioned_vs_extrapolation"] = bootstrap(packed["conditioned"], {"delta": packed["motion_extrapolation"]})["delta"]
            cell["unconditioned_vs_extrapolation"] = bootstrap(packed["unconditioned"], {"delta": packed["motion_extrapolation"]})["delta"]
            cell["extrapolation_vs_persistence"] = bootstrap(packed["motion_extrapolation"], {"delta": packed["persistence"]})["delta"]
        cells[label] = cell

    induced = masks(test)["robot_induced"]
    target_shift = com_shift(test["target"][induced], test["rho_t"][induced])
    pred_shift = com_shift(predictions["conditioned"][induced], test["rho_t"][induced])
    moving = np.abs(target_shift) > 0.05
    sensitivity_index = np.flatnonzero(test["density"].sum(axis=(1, 2)) > 1e-4)[:200] if False else np.flatnonzero(test["rho_t"].sum(axis=(1, 2)) > 1e-4)[:200]
    grid = np.array([[vx, wz] for vx in (0.0, 0.3, 0.5, 0.8) for wz in (0.0, 0.7, 1.0)], dtype=np.float32)
    base = predictions["conditioned"][sensitivity_index]
    changes = []
    subset = {key: test[key][sensitivity_index] for key in ("rho_t", "rho_prev", "action")}
    for action in grid:
        alternate = predict(conditioned, subset, np.repeat(action[None], len(sensitivity_index), axis=0))
        changes.append(float(np.mean(np.abs(alternate - base))))

    def counts(split: str) -> dict:
        chosen = groups[split]
        return {
            "episodes": len(chosen),
            "pairs": int(len(data[split]["distance"])),
            "by_count": {str(count): int(sum(ep["count"] == count for ep in chosen)) for count in (1, 5, 15, 20)},
            "by_command": {name: int(sum(ep["name"] == name for ep in chosen)) for name in ("block", "leave", "approach", "slow", "cruise", "yaw")},
        }

    result = {
        "planner": "pedvo",
        "horizon_steps": HORIZON,
        "sample_hz": SAMPLE_HZ,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "torch_seed": TORCH_SEED,
        "train_seeds": sorted(TRAIN_SEEDS),
        "validation_seeds": sorted(VAL_SEEDS),
        "test_seeds": sorted(TEST_SEEDS),
        "repeats_per_seed": 2,
        "parameter_count": sum(p.numel() for p in conditioned.parameters()),
        "dataset": {split: counts(split) for split in groups},
        "loss_curves": {
            "conditioned_train": train_c,
            "conditioned_val": val_c,
            "unconditioned_train": train_u,
            "unconditioned_val": val_u,
        },
        "held_out": cells,
        "lateral_direction": {
            "robot_induced_pairs": int(induced.sum()),
            "pairs_with_target_shift_over_5cm": int(moving.sum()),
            "conditioned_sign_agreement": float(np.mean(np.sign(pred_shift[moving]) == np.sign(target_shift[moving]))) if moving.any() else None,
            "mean_target_shift_m": float(np.mean(target_shift)) if induced.any() else None,
            "mean_conditioned_shift_m": float(np.mean(pred_shift)) if induced.any() else None,
        },
        "action_sensitivity": {
            "pairs": int(len(sensitivity_index)),
            "mean_abs_change_when_action_replaced": changes,
            "mean_abs_change_recorded_vs_zero": float(np.mean(np.abs(predictions["conditioned"] - predictions["conditioned_action_zeroed"]))),
            "grid": grid.tolist(),
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"dataset": result["dataset"], "held_out_all": cells["all"], "robot_induced": cells["robot_induced"]}, indent=2))


if __name__ == "__main__":
    main()
