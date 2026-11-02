"""Tests for previous-command selection and same-episode forecast pairs."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

_ROOT = (
    Path(__file__).resolve().parents[1]
    / "rosnav_rl/model/stable_baselines3/policy/feature_extractors/mean_field"
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_action = _load("mean_field_action", "action.py")
_targets = _load("mean_field_targets", "targets.py")
_response = _load("crowd_response_net", "response_model.py")


def test_three_component_twist_keeps_yaw_rate_not_lateral_speed():
    twist = np.array([0.4, 0.9, -0.3], dtype=np.float32)
    command = _action.previous_command(twist)
    np.testing.assert_allclose(command, [0.4, -0.3])


def test_two_component_command_is_already_linear_and_yaw():
    command = torch.tensor([[0.2, -0.5], [0.0, 0.1]])
    selected = _action.previous_command(command)
    torch.testing.assert_close(selected, command)


def test_forecast_pairs_stop_at_an_episode_boundary_and_the_buffer_end():
    # steps: start, continue, continue, new episode, continue
    starts = np.array([[1], [0], [0], [1], [0]], dtype=np.float32)
    valid = _targets.same_episode_forecast_mask(starts, horizon=2)
    np.testing.assert_array_equal(valid[:, 0], [True, False, False, False, False])


def test_two_environments_are_masked_independently():
    starts = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    valid = _targets.same_episode_forecast_mask(starts, horizon=2)
    np.testing.assert_array_equal(valid[:, 0], [True, True, False, False])
    np.testing.assert_array_equal(valid[:, 1], [False, False, False, False])


def test_flatten_matches_the_rollout_buffer_order():
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    flat = _targets.flatten_like_rollout(values)
    # env 0 steps, then env 1 steps
    np.testing.assert_array_equal(flat, [0, 2, 4, 1, 3, 5])


def test_density_loss_updates_the_response_network():
    torch.manual_seed(0)
    net = _response.CrowdResponseNet()
    rho_t = torch.rand(4, 1, 16, 16)
    rho_prev = torch.rand(4, 1, 16, 16)
    velocity = torch.tensor([[0.3, -0.2]]).repeat(4, 1)
    target = torch.rand(4, 1, 16, 16)
    valid = torch.tensor([True, False, True, False])
    prediction = net(rho_t, rho_prev, velocity)
    loss = _targets.masked_density_loss(prediction, target, valid)
    loss.backward()
    gradients = [parameter.grad for parameter in net.parameters() if parameter.grad is not None]
    assert gradients
    assert any(torch.any(gradient != 0) for gradient in gradients)
