"""AGENT_5 ResNet with one extra channel: the online crowd forecast."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosnav_rl.spaces.observation_space.spaces.dynamics.basic_dynamics_spaces import (
    LastActionSpace,
)
from rosnav_rl.spaces.observation_space.spaces.environment.pedestrian_state_space import (
    ForecastCommandSpace,
    PedestrianKinematicsSpace,
)

from ..resnet.resnet import DRL_VO_ROSNAV_EXTRACTOR
from .interaction import InteractionResidual, forecast_density, residual_checkpoint


class ResidualForecastExtractor(DRL_VO_ROSNAV_EXTRACTOR):
    """Append the one-second density forecast as the sixth convolution channel.

    The first five channels are the AGENT_5 laser map and four pedestrian maps.
    ``apply_correction=False`` is online constant-velocity extrapolation.
    ``apply_correction=True`` adds the frozen residual. Both use
    ``forecast_density`` and the same unicycle baseline.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        stack_size: int = 1,
        apply_correction: bool = False,
        residual_checkpoint_path: str | None = None,
        **kwargs,
    ) -> None:
        self._apply_correction = apply_correction
        super().__init__(
            observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
            **kwargs,
        )
        self._residual_path = Path(residual_checkpoint_path) if residual_checkpoint_path else residual_checkpoint()
        if apply_correction and not self._residual_path.is_file():
            raise FileNotFoundError(f"interaction residual checkpoint not found: {self._residual_path}")
        self.residual = InteractionResidual()
        self._forecast_steps = 0
        self._load_residual()
        # SB3 orthogonal-initializes every Linear after this constructor returns.
        # The first forward loads the checkpoint again, after that pass.
        self._residual_ready = False

    def _load_residual(self) -> None:
        if self._apply_correction:
            self.residual.load_state_dict(torch.load(self._residual_path, map_location="cpu", weights_only=True))
        self.residual.requires_grad_(False)
        self.residual.eval()

    def _setup_network(self, inplanes: int = 64):
        super()._setup_network(inplanes)
        # Laser, the four AGENT_5 pedestrian maps, and the forecast.
        self.conv1 = nn.Conv2d(
            self.num_pedestrian_feature_maps + 2,
            inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.residual.eval()
        return self

    def _forecast(self, observations) -> tuple[torch.Tensor, torch.Tensor]:
        state = observations[PedestrianKinematicsSpace.name]
        command = observations[ForecastCommandSpace.name]
        positions = state[..., 0:2]
        velocities = state[..., 2:4]
        valid = state[..., 4] > 0.5
        twist = command[..., 0:2]
        slow_heading = command[..., 2:4]
        online = forecast_density(
            None, positions, velocities, valid, twist, apply_correction=False, slow_heading=slow_heading
        )
        if not self._apply_correction:
            return online, online
        corrected = forecast_density(
            self.residual,
            positions,
            velocities,
            valid,
            twist,
            apply_correction=True,
            slow_heading=slow_heading,
        )
        return online, corrected

    def forward(self, observations) -> torch.Tensor:
        if not self._residual_ready:
            self._load_residual()
            self._residual_ready = True
        inputs = self._get_input(observations)
        ped_map = inputs["ped_map"]
        if ped_map is None:
            raise RuntimeError("ResidualForecastExtractor requires the AGENT_5 pedestrian maps")
        if LastActionSpace.name not in observations:
            raise RuntimeError("ResidualForecastExtractor requires the previous command")
        mode = os.environ.get("MF_FORECAST_MODE", "")
        if mode == "online":
            self._apply_correction = False
        elif mode == "residual":
            self._apply_correction = True
        started = time.perf_counter()
        online, forecast = self._forecast(observations)
        if mode == "zero":
            forecast = torch.zeros_like(forecast)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._forecast_steps += 1
        if self._forecast_steps % 32 == 1:
            correction = float((forecast - online).abs().mean())
            density_sum = float(forecast.abs().sum() / forecast.shape[0])
            print(
                f"crowd forecast correction {correction:.6f} density sum {density_sum:.4f} "
                f"batch {forecast.shape[0]} latency {elapsed_ms:.2f} ms",
                flush=True,
            )
        upsampled = F.interpolate(
            forecast.detach().unsqueeze(1),
            size=ped_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        inputs["ped_map"] = torch.cat((ped_map, upsampled), dim=1)
        return self._forward_impl(**inputs)
