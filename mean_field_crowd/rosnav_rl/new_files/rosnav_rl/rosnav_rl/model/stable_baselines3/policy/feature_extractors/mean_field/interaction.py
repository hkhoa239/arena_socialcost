"""Online PedVO forecast used by AGENT_MF_ONLINE and AGENT_MF_RESIDUAL.

The robot is at the origin of its current frame. Its pose one second ahead is
the unicycle integral of the previous command, held constant. Pedestrians move
at their current velocity. The residual adds a gated body-frame offset of at
most one meter. Nothing in this module reads a future pose or a future
pedestrian state.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

MAX_OFFSET = 1.0
HORIZON = 1.0
YAW_RATE_EPSILON = 1e-3
FEATURE_SCALE = (5.0, 5.0, 1.5, 1.5, 1.0, 1.0, 1.5, 5.0, 2.0)
APPROACH_TIMES = torch.linspace(0.0, 2.0, 21)
SIGMA = 0.5
RESOLUTION = 20.0 / 16.0
CELL_SCALE = RESOLUTION * RESOLUTION / (2.0 * math.pi * SIGMA * SIGMA)


def residual_checkpoint() -> Path:
    """Pretrained residual from the PedVO offline run."""
    candidates = (
        Path("/opt/arena_ws/data/mf_pedvo/interaction_residual.pt"),
        Path("/home/robotics/arena_ws/data/mf_pedvo/interaction_residual.pt"),
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


class InteractionResidual(nn.Module):
    """211-parameter gate and body-frame offset. State dict matches the offline run."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        gate = torch.sigmoid(raw[..., 0])
        lateral = torch.tanh(raw[..., 1]) * MAX_OFFSET
        forward = torch.tanh(raw[..., 2]) * MAX_OFFSET
        return gate, lateral, forward


def integrate_unicycle(x, y, yaw, v, w, tau):
    """Hold a commanded twist for ``tau`` seconds. States broadcast against ``tau``."""
    small = w.abs() < YAW_RATE_EPSILON
    yaw_next = yaw + w * tau
    safe_w = torch.where(small, torch.ones_like(w), w)
    v_over_w = v / safe_w
    straight_x = x + v * torch.cos(yaw) * tau
    straight_y = y + v * torch.sin(yaw) * tau
    arc_x = x + v_over_w * (torch.sin(yaw_next) - torch.sin(yaw))
    arc_y = y + v_over_w * (-torch.cos(yaw_next) + torch.cos(yaw))
    return (
        torch.where(small, straight_x, arc_x),
        torch.where(small, straight_y, arc_y),
        torch.where(small, yaw, yaw_next),
    )


def _to_frame(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    dx = points[..., 0] - pose[:, None, 0]
    dy = points[..., 1] - pose[:, None, 1]
    cos_yaw = torch.cos(pose[:, 2])[:, None]
    sin_yaw = torch.sin(pose[:, 2])[:, None]
    return torch.stack((cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy), dim=-1)


def _blobs(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
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


def _features(positions, velocities, command, slow_heading):
    batch, count, _ = positions.shape
    times = APPROACH_TIMES.to(device=positions.device, dtype=positions.dtype)
    linear = command[:, 0]
    angular = command[:, 1]
    zeros = torch.zeros_like(linear)
    robot_x, robot_y, _ = integrate_unicycle(
        zeros[:, None],
        zeros[:, None],
        zeros[:, None],
        linear[:, None],
        angular[:, None],
        times.view(1, -1),
    )
    future = positions[:, None, :, :] + velocities[:, None, :, :] * times.view(1, -1, 1, 1)
    robot = torch.stack((robot_x, robot_y), dim=-1)[:, :, None, :]
    distance = torch.linalg.norm(future - robot, dim=-1)
    nearest = distance.argmin(dim=1)
    closest = distance.min(dim=1).values
    time_to_closest = times[nearest]
    relative_velocity_x = velocities[..., 0] - linear[:, None]
    relative_velocity_y = velocities[..., 1]
    separation = torch.linalg.norm(positions, dim=-1).clamp_min(1e-3)
    closing = -(positions[..., 0] * relative_velocity_x + positions[..., 1] * relative_velocity_y) / separation
    command_per_ped = command[:, None, :].expand(batch, count, 2)
    raw = torch.cat(
        (
            positions,
            velocities,
            command_per_ped,
            closing[..., None],
            closest[..., None],
            time_to_closest[..., None],
        ),
        dim=-1,
    )
    scale = raw.new_tensor(FEATURE_SCALE)
    speed = torch.linalg.norm(velocities, dim=-1, keepdim=True)
    forward = velocities / speed.clamp_min(1e-6)
    if slow_heading is None:
        default_heading = forward.new_tensor((1.0, 0.0)).expand(batch, count, 2)
    else:
        default_heading = slow_heading[:, None, :].expand(batch, count, 2)
    forward = torch.where(speed >= 0.2, forward, default_heading)
    lateral = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
    return raw / scale, forward, lateral


def constant_velocity_points(positions: torch.Tensor, velocities: torch.Tensor, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pedestrian positions one second ahead, and the predicted robot pose, both in the current robot frame."""
    ped_cv = positions + velocities * HORIZON
    linear = command[:, 0]
    angular = command[:, 1]
    zeros = torch.zeros_like(linear)
    one = torch.ones_like(linear)
    pose = torch.stack(integrate_unicycle(zeros, zeros, zeros, linear, angular, one), dim=-1)
    return ped_cv, pose


def forecast_density(
    module: InteractionResidual | None,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    valid: torch.Tensor,
    command: torch.Tensor,
    apply_correction: bool,
    slow_heading: torch.Tensor | None = None,
) -> torch.Tensor:
    """16×16 density in the predicted robot frame.

    ``apply_correction`` is the only difference between the online baseline and
    the residual. Both call this function and share ``constant_velocity_points``.
    """
    positions = positions.to(dtype=torch.float32)
    velocities = velocities.to(dtype=torch.float32)
    command = command.to(dtype=torch.float32)
    if slow_heading is not None:
        slow_heading = slow_heading.to(dtype=torch.float32)
    ped_cv, pose = constant_velocity_points(positions, velocities, command)
    cv_frame = _to_frame(ped_cv, pose)
    if not apply_correction:
        return _blobs(cv_frame, valid).sum(dim=1)
    features, forward, lateral_axis = _features(positions, velocities, command, slow_heading)
    gate, lateral, along = module(features)
    gate = gate * valid.to(gate.dtype)
    delta = lateral_axis * lateral[..., None] + forward * along[..., None]
    corrected = _to_frame(ped_cv + delta, pose)
    mixed = (1.0 - gate[..., None, None]) * _blobs(cv_frame, valid) + gate[..., None, None] * _blobs(corrected, valid)
    return (mixed * valid[:, :, None, None].to(mixed.dtype)).sum(dim=1)
