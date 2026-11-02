"""Gated interaction residual on top of the online constant-velocity forecast.

The correction is trained on seeds 5 and 6. Seed 7 selects the checkpoint.
Seeds 8, 9, and 10 are evaluated once. Width, learning rate, epoch budget,
offset limit, and feature scales are fixed before that evaluation.

rho_hat = rho_cv + sum_i g_i * (blob(p_i + s_i) - blob(p_i))
g_i is zero when the pedestrian is padded. s_i is a body-frame offset of at
most one meter. Inputs are the current pedestrian state and the previous command.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mf_pedvo_forecast import (
    BOOTSTRAP,
    DATA,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    episode_sse,
    load_episodes,
    pooled,
    rasterize_density,
)
from mf_pedvo_online import integrate_unicycle, masks, to_robot
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.response_model import (
    CrowdResponseNet,
)

EPOCHS = 40
BATCH = 64
LR = 1e-3
TORCH_SEED = 5
MAX_OFFSET = 1.0
MAX_PEDS = 20
FEATURE_SCALE = np.array([5.0, 5.0, 1.5, 1.5, 1.0, 1.0, 1.5, 5.0, 2.0], dtype=np.float32)
TAUS = np.linspace(0.0, 2.0, 21)
OUT = Path("/opt/arena_ws/data/mf_pedvo_residual.json")
SIGMA = 0.5
RESOLUTION = 20.0 / 16.0
CELL_SCALE = RESOLUTION * RESOLUTION / (2.0 * math.pi * SIGMA * SIGMA)


class InteractionResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 3))
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        last.bias.data[0] = -2.0

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        gate = torch.sigmoid(raw[..., 0])
        lateral = torch.tanh(raw[..., 1]) * MAX_OFFSET
        forward = torch.tanh(raw[..., 2]) * MAX_OFFSET
        return gate, lateral, forward


def blobs(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    coords = (torch.arange(16, device=points.device, dtype=points.dtype) - 8.0) * RESOLUTION
    xx = coords.view(1, 1, 1, 16)
    yy = coords.view(1, 1, 16, 1)
    x = points[..., 0][:, :, None, None]
    y = points[..., 1][:, :, None, None]
    inside = (x >= -10.0) & (x < 10.0) & (y >= -10.0) & (y < 10.0) & valid[:, :, None, None]
    dx = xx - x
    dy = yy - y
    kernel = torch.exp(-0.5 * (dx * dx + dy * dy) / (SIGMA * SIGMA)) * CELL_SCALE
    return kernel * inside.to(kernel.dtype)


def mix_density(module, features, valid, ped_cv, n_fwd, n_lat, pose) -> torch.Tensor:
    gate, lateral, forward = module(features)
    gate = gate * valid.to(gate.dtype)
    delta = n_lat * lateral[..., None] + n_fwd * forward[..., None]
    cv_frame = torch_to_robot(ped_cv, pose)
    corrected = torch_to_robot(ped_cv + delta, pose)
    occupied = valid[:, :, None, None].to(cv_frame.dtype)
    mixed = (1.0 - gate[..., None, None]) * blobs(cv_frame, valid) + gate[..., None, None] * blobs(corrected, valid)
    return (mixed * occupied).sum(dim=1), gate


def torch_to_robot(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    dx = points[..., 0] - pose[:, None, 0]
    dy = points[..., 1] - pose[:, None, 1]
    cos_yaw = torch.cos(pose[:, 2])[:, None]
    sin_yaw = torch.sin(pose[:, 2])[:, None]
    return torch.stack((cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy), dim=-1)


def closest_approach(ped, vel, x, y, yaw, v, w):
    rx, ry, _ = integrate_unicycle(x[:, None], y[:, None], yaw[:, None], v[:, None], w[:, None], TAUS[None, :])
    future = ped[:, None, :, :] + vel[:, None, :, :] * TAUS[None, :, None, None]
    robot = np.stack((rx, ry), axis=-1)[:, :, None, :]
    distance = np.linalg.norm(future - robot, axis=-1)
    choice = distance.argmin(axis=1)
    return distance.min(axis=1), TAUS[choice]


def episode_batch(episode: dict) -> dict:
    ped = episode["ped"]
    vel = episode["vel"]
    robot = episode["robot"]
    yaw = episode["yaw"]
    command = episode["command"]
    index = np.arange(1, len(ped) - 10)
    ped_t = ped[index]
    vel_t = vel[index]
    action = command[index - 1]
    count = ped.shape[1]
    pred_x, pred_y, pred_yaw = integrate_unicycle(robot[index, 0], robot[index, 1], yaw[index], action[:, 0], action[:, 1], 1.0)
    cos_yaw = np.cos(yaw[index])[:, None]
    sin_yaw = np.sin(yaw[index])[:, None]
    speed_x = vel_t[..., 0]
    speed_y = vel_t[..., 1]
    vel_robot = np.stack((cos_yaw * speed_x + sin_yaw * speed_y, -sin_yaw * speed_x + cos_yaw * speed_y), axis=-1)
    robot_vel = np.stack((action[:, 0] * np.cos(yaw[index]), action[:, 0] * np.sin(yaw[index])), axis=1)
    relative_world = ped_t - robot[index, None, :]
    relative_vel = vel_t - robot_vel[:, None, :]
    distance = np.linalg.norm(relative_world, axis=-1)
    closing = -np.sum(relative_world * relative_vel, axis=-1) / np.maximum(distance, 1e-3)
    cpa, tcpa = closest_approach(ped_t, vel_t, robot[index, 0], robot[index, 1], yaw[index], action[:, 0], action[:, 1])
    relative = episode["relative"][index]
    raw = np.concatenate(
        (
            relative,
            vel_robot,
            np.repeat(action[:, None, :], count, axis=1),
            closing[..., None],
            cpa[..., None],
            tcpa[..., None],
        ),
        axis=-1,
    ).astype(np.float32)
    raw /= FEATURE_SCALE
    heading = np.linalg.norm(vel_t, axis=-1, keepdims=True)
    forward = np.divide(vel_t, np.maximum(heading, 1e-6))
    forward = np.where(heading >= 0.2, forward, np.array([1.0, 0.0]))
    lateral = np.stack((-forward[..., 1], forward[..., 0]), axis=-1)
    features = np.zeros((len(index), MAX_PEDS, 9), dtype=np.float32)
    valid = np.zeros((len(index), MAX_PEDS), dtype=bool)
    ped_cv = np.zeros((len(index), MAX_PEDS, 2), dtype=np.float32)
    n_fwd = np.zeros((len(index), MAX_PEDS, 2), dtype=np.float32)
    n_lat = np.zeros((len(index), MAX_PEDS, 2), dtype=np.float32)
    features[:, :count] = raw
    valid[:, :count] = True
    ped_cv[:, :count] = (ped_t + vel_t).astype(np.float32)
    n_fwd[:, :count] = forward.astype(np.float32)
    n_lat[:, :count] = lateral.astype(np.float32)
    center = episode["center"]
    current_lat = np.abs(ped_t[:, center, 1] - episode["spawn_y"][center])
    future_lat = np.abs(ped[index + 10, center, 1] - episode["spawn_y"][center])
    others = [i for i in range(count) if i != center]
    if others:
        neighbor_lat = np.max(np.abs(ped[index + 10][:, others, 1] - episode["spawn_y"][others]), axis=1)
    else:
        neighbor_lat = np.zeros(len(index))
    return {
        "features": features,
        "valid": valid,
        "ped_cv": ped_cv,
        "n_fwd": n_fwd,
        "n_lat": n_lat,
        "pose": np.stack((pred_x, pred_y, pred_yaw), axis=1).astype(np.float32),
        "target": episode["density"][index + 10],
        "persistence": episode["density"][index],
        "rho_t": episode["density"][index],
        "rho_prev": episode["density"][index - 1],
        "action": action.astype(np.float32),
        "current_lat": current_lat,
        "future_lat": future_lat,
        "neighbor_lat": neighbor_lat,
        "distance": np.linalg.norm(relative[:, center], axis=1),
        "count": np.full(len(index), episode["count"]),
        "name": np.array([episode["name"]] * len(index)),
    }


def stack(parts: list[dict]) -> tuple[dict, np.ndarray]:
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0] if key != "name"}
    data["name"] = np.concatenate([part["name"] for part in parts])
    bounds = np.cumsum([0] + [len(part["distance"]) for part in parts])
    return data, bounds


def tensors(data: dict) -> dict:
    out = {}
    for key in ("features", "ped_cv", "n_fwd", "n_lat", "pose", "target", "rho_t", "rho_prev", "action"):
        out[key] = torch.as_tensor(data[key], dtype=torch.float32)
    out["valid"] = torch.as_tensor(data["valid"])
    return out


def predict_residual(module, data: dict) -> tuple[np.ndarray, np.ndarray]:
    module.eval()
    held = tensors(data)
    densities = []
    gates = []
    with torch.no_grad():
        for start in range(0, len(data["distance"]), 256):
            end = start + 256
            density, gate = mix_density(
                module,
                held["features"][start:end],
                held["valid"][start:end],
                held["ped_cv"][start:end],
                held["n_fwd"][start:end],
                held["n_lat"][start:end],
                held["pose"][start:end],
            )
            densities.append(density.numpy())
            gates.append(gate.numpy())
    return np.concatenate(densities), np.concatenate(gates)


def online_density(data: dict) -> np.ndarray:
    frames = to_robot(data["ped_cv"], data["pose"][:, 0][:, None], data["pose"][:, 1][:, None], data["pose"][:, 2][:, None])
    # Padded slots sit at the origin. Drop them before rasterizing.
    frames = frames.copy()
    frames[~data["valid"]] = 100.0
    chunks = [rasterize_density(frames[start : start + 512])[:, 0] for start in range(0, len(frames), 512)]
    return np.concatenate(chunks)


def train(module, train_data, val_data) -> tuple[list[float], list[float], int]:
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    train_t = tensors(train_data)
    val_t = tensors(val_data)
    generator = torch.Generator().manual_seed(TORCH_SEED)
    train_curve, val_curve = [], []
    best_state = None
    best_val = math.inf
    best_epoch = 0
    for epoch in range(EPOCHS):
        module.train()
        order = torch.randperm(len(train_data["distance"]), generator=generator)
        total = 0.0
        for start in range(0, len(order), BATCH):
            choice = order[start : start + BATCH]
            prediction, _ = mix_density(
                module,
                train_t["features"][choice],
                train_t["valid"][choice],
                train_t["ped_cv"][choice],
                train_t["n_fwd"][choice],
                train_t["n_lat"][choice],
                train_t["pose"][choice],
            )
            loss = torch.mean((prediction - train_t["target"][choice]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(choice)
        train_curve.append(total / len(train_data["distance"]))
        module.eval()
        with torch.no_grad():
            val_prediction, _ = mix_density(
                module, val_t["features"], val_t["valid"], val_t["ped_cv"], val_t["n_fwd"], val_t["n_lat"], val_t["pose"]
            )
            val_loss = float(torch.mean((val_prediction - val_t["target"]) ** 2))
        val_curve.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}
        print(f"epoch {epoch + 1} train {train_curve[-1]:.6f} val {val_loss:.6f}", flush=True)
    module.load_state_dict(best_state)
    return train_curve, val_curve, best_epoch


def predict_cnn(data: dict) -> np.ndarray:
    net = CrowdResponseNet(conditioned=True)
    net.load_state_dict(torch.load(DATA / "conditioned.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        prediction = net(
            torch.as_tensor(data["rho_t"][:, None], dtype=torch.float32),
            torch.as_tensor(data["rho_prev"][:, None], dtype=torch.float32),
            torch.as_tensor(data["action"], dtype=torch.float32),
        )
    return prediction[:, 0].numpy()


def bootstrap_delta(reference, other) -> dict:
    sse, count = reference
    present = np.flatnonzero(count > 0)
    if len(present) < 2:
        return {"mean_delta": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(0)
    draws = present[rng.integers(0, len(present), size=(BOOTSTRAP, len(present)))]
    delta = np.empty(BOOTSTRAP)
    other_sse, other_count = other
    for row, draw in enumerate(draws):
        delta[row] = pooled(sse, count, draw) - pooled(other_sse, other_count, draw)
    return {"mean_delta": float(delta.mean()), "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))]}


def check_rasterizer(data: dict) -> float:
    module = InteractionResidual()
    module.eval()
    sample = {key: torch.as_tensor(data[key][:8], dtype=torch.float32) for key in ("features", "ped_cv", "n_fwd", "n_lat", "pose")}
    sample["valid"] = torch.as_tensor(data["valid"][:8])
    with torch.no_grad():
        density, _ = mix_density(module, sample["features"], sample["valid"], sample["ped_cv"], sample["n_fwd"], sample["n_lat"], sample["pose"])
    reference = online_density({key: data[key][:8] for key in ("ped_cv", "pose", "valid")})
    return float(np.max(np.abs(density.numpy() - reference)))


def latency_ms(function, repeats: int = 100) -> float:
    for _ in range(10):
        function()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats * 1000.0


def main() -> None:
    grouped = {"train": [], "val": [], "test": []}
    for episode in load_episodes():
        if episode["seed"] in TRAIN_SEEDS:
            grouped["train"].append(episode_batch(episode))
        elif episode["seed"] in VAL_SEEDS:
            grouped["val"].append(episode_batch(episode))
        elif episode["seed"] in TEST_SEEDS:
            grouped["test"].append(episode_batch(episode))
    data = {}
    bounds = {}
    for name, parts in grouped.items():
        data[name], bounds[name] = stack(parts)
        print(f"{name} episodes {len(parts)} pairs {len(data[name]['distance'])}", flush=True)
    mismatch = check_rasterizer(data["train"])
    print(f"rasterizer max abs error {mismatch:.3e}", flush=True)
    if mismatch > 1e-4:
        raise RuntimeError(f"torch rasterizer does not match the density grid: {mismatch}")

    torch.manual_seed(TORCH_SEED)
    module = InteractionResidual()
    train_curve, val_curve, best_epoch = train(module, data["train"], data["val"])
    torch.save(module.state_dict(), DATA / "interaction_residual.pt")

    test = data["test"]
    residual, gate = predict_residual(module, test)
    online = online_density(test)
    oracle_parts = []
    # Oracle uses the recorded future pose. Rebuild it from the source episodes so the
    # diagnostic stays independent of the predicted pose stored in the batch.
    from mf_pedvo_online import pairs_of as oracle_pairs

    source = [episode for episode in load_episodes() if episode["seed"] in TEST_SEEDS]
    for episode in source:
        oracle_parts.append(oracle_pairs(episode)["oracle"])
    oracle = np.concatenate(oracle_parts)
    cnn = predict_cnn(test)
    predictions = {
        "persistence": test["persistence"],
        "oracle_extrapolation": oracle,
        "online_extrapolation": online,
        "conditioned_cnn": cnn,
        "interaction_residual": residual,
    }
    cells = {}
    for label, mask in masks(test).items():
        packed = {name: episode_sse(prediction, test["target"], bounds["test"], mask) for name, prediction in predictions.items()}
        cell = {"pairs": int(mask.sum())}
        for name, parts_sse in packed.items():
            cell[name] = pooled(*parts_sse)
        if mask.any():
            cell["residual_vs_online"] = bootstrap_delta(packed["interaction_residual"], packed["online_extrapolation"])
            cell["residual_vs_cnn"] = bootstrap_delta(packed["interaction_residual"], packed["conditioned_cnn"])
            cell["online_vs_oracle"] = bootstrap_delta(packed["online_extrapolation"], packed["oracle_extrapolation"])
        cells[label] = cell

    valid = test["valid"]
    mean_gate = np.divide(gate.sum(axis=1), np.maximum(valid.sum(axis=1), 1))
    gate_report = {}
    for label, mask in masks(test).items():
        if label.startswith("n") or label.startswith("cmd_") or not mask.any():
            continue
        gate_report[label] = {"mean_gate": float(mean_gate[mask].mean()), "pairs": int(mask.sum())}

    def one_scene(count: int):
        episode = next(episode for episode in source if episode["count"] == count and episode["name"] == "block")
        ped = episode["ped"][20:21]
        vel = episode["vel"][20:21]
        robot = episode["robot"][20]
        yaw = episode["yaw"][20]
        action = episode["command"][19]

        def online_once():
            px, py, pyaw = integrate_unicycle(robot[0], robot[1], yaw, action[0], action[1], 1.0)
            points = to_robot(ped + vel, np.array([[px]]), np.array([[py]]), np.array([[pyaw]]))
            rasterize_density(points[0])

        batch = episode_batch(episode)
        row = 19
        feature_row = {key: torch.as_tensor(batch[key][row : row + 1], dtype=torch.float32) for key in ("features", "ped_cv", "n_fwd", "n_lat", "pose")}
        feature_row["valid"] = torch.as_tensor(batch["valid"][row : row + 1])

        def residual_once():
            closest_approach(ped, vel, np.array([robot[0]]), np.array([robot[1]]), np.array([yaw]), np.array([action[0]]), np.array([action[1]]))
            with torch.no_grad():
                mix_density(module, feature_row["features"], feature_row["valid"], feature_row["ped_cv"], feature_row["n_fwd"], feature_row["n_lat"], feature_row["pose"])

        rho = torch.as_tensor(batch["rho_t"][row : row + 1, None], dtype=torch.float32)
        prev = torch.as_tensor(batch["rho_prev"][row : row + 1, None], dtype=torch.float32)
        act = torch.as_tensor(batch["action"][row : row + 1], dtype=torch.float32)
        cnn_net = CrowdResponseNet(conditioned=True)
        cnn_net.load_state_dict(torch.load(DATA / "conditioned.pt", map_location="cpu"))
        cnn_net.eval()

        def cnn_once():
            with torch.no_grad():
                cnn_net(rho, prev, act)

        return {
            "online_extrapolation_ms": latency_ms(online_once),
            "interaction_residual_ms": latency_ms(residual_once),
            "conditioned_cnn_ms": latency_ms(cnn_once),
        }

    result = {
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "epochs": EPOCHS,
        "selected_epoch": best_epoch,
        "learning_rate": LR,
        "batch_size": BATCH,
        "max_offset_m": MAX_OFFSET,
        "train_seeds": sorted(TRAIN_SEEDS),
        "validation_seeds": sorted(VAL_SEEDS),
        "test_seeds": sorted(TEST_SEEDS),
        "rasterizer_max_abs_error": mismatch,
        "loss_curves": {"train": train_curve, "validation": val_curve},
        "dataset": {name: {"episodes": len(grouped[name]), "pairs": int(len(data[name]["distance"]))} for name in grouped},
        "held_out": cells,
        "mean_gate": gate_report,
        "latency_ms": {"n1": one_scene(1), "n20": one_scene(20)},
    }
    OUT.write_text(json.dumps(result, indent=2))
    summary = {
        label: {
            "pairs": cell["pairs"],
            "online": cell.get("online_extrapolation"),
            "residual": cell.get("interaction_residual"),
            "cnn": cell.get("conditioned_cnn"),
            "residual_vs_online": cell.get("residual_vs_online"),
        }
        for label, cell in cells.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
