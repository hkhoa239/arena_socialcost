"""One-second density targets inside a rollout, without crossing episodes.

``episode_starts[t, e]`` is 1 when the observation at step ``t`` is the first
observation of an episode, which is how Stable-Baselines3 fills the rollout
buffer. The target of step ``t`` is the current density at ``t + horizon``.
That pair is kept only when every observation from ``t + 1`` through
``t + horizon`` belongs to the same episode.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3.common.buffers import RolloutBuffer


def same_episode_forecast_mask(episode_starts: np.ndarray, horizon: int) -> np.ndarray:
    """Return a ``(n_steps, n_envs)`` mask of steps that have a same-episode future."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    starts = np.asarray(episode_starts)
    if starts.ndim != 2:
        raise ValueError(f"episode_starts must have shape (n_steps, n_envs), got {starts.shape}")
    steps = starts.shape[0]
    valid = np.zeros(starts.shape, dtype=bool)
    if steps <= horizon:
        return valid
    window = np.stack(
        [starts[offset : steps - horizon + offset] for offset in range(1, horizon + 1)],
        axis=0,
    )
    valid[:-horizon] = window.sum(axis=0) == 0
    return valid


def flatten_like_rollout(array: np.ndarray) -> np.ndarray:
    """Match ``RolloutBuffer.swap_and_flatten`` so minibatch indices line up.

    A two-dimensional ``(n_steps, n_envs)`` array, such as the validity mask,
    comes back with a trailing dimension of 1. That dimension is removed so a
    mask stays one Boolean per sample.
    """
    values = np.asarray(array)
    flat = RolloutBuffer.swap_and_flatten(values)
    if values.ndim == 2:
        flat = flat.reshape(-1)
    return flat


def masked_density_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error on the valid forecast pairs. Zero when none are valid."""
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target[valid])
