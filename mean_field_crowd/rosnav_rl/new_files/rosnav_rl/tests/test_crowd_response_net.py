"""Tests for the crowd response network."""

import importlib.util
from pathlib import Path

import torch

_PATH = (
    Path(__file__).resolve().parents[1]
    / "rosnav_rl/model/stable_baselines3/policy/feature_extractors/mean_field/response_model.py"
)
_spec = importlib.util.spec_from_file_location("crowd_response_net", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
CrowdResponseNet = _module.CrowdResponseNet


def _inputs(batch: int = 2):
    torch.manual_seed(0)
    rho_t = torch.rand(batch, 1, 16, 16)
    rho_prev = torch.rand(batch, 1, 16, 16)
    velocity = torch.tensor([[0.4, -0.2]]).repeat(batch, 1)
    return rho_t, rho_prev, velocity


def test_output_is_nonnegative_density():
    net = CrowdResponseNet()
    rho_t, rho_prev, velocity = _inputs()
    prediction = net(rho_t, rho_prev, velocity)
    assert prediction.shape == (2, 1, 16, 16)
    assert torch.all(prediction >= 0)


def test_unconditioned_network_ignores_velocity_and_matches_parameter_count():
    torch.manual_seed(1)
    conditioned = CrowdResponseNet(conditioned=True)
    torch.manual_seed(1)
    plain = CrowdResponseNet(conditioned=False)
    conditioned_count = sum(parameter.numel() for parameter in conditioned.parameters())
    plain_count = sum(parameter.numel() for parameter in plain.parameters())
    assert conditioned_count == plain_count

    rho_t, rho_prev, velocity = _inputs()
    other_velocity = torch.zeros_like(velocity)
    plain.load_state_dict(conditioned.state_dict())
    with torch.no_grad():
        from_velocity = plain(rho_t, rho_prev, velocity)
        from_zero = plain(rho_t, rho_prev, other_velocity)
        changed = conditioned(rho_t, rho_prev, velocity)
    torch.testing.assert_close(from_velocity, from_zero)
    assert not torch.allclose(changed, from_velocity)
