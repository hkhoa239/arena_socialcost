"""ResNet extractor with one extra channel: the predicted crowd density."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosnav_rl.spaces.observation_space.spaces.dynamics.basic_dynamics_spaces import (
    LastActionSpace,
)
from rosnav_rl.spaces.observation_space.spaces.environment.density_space import (
    DENSITY_CURRENT,
    DENSITY_PREVIOUS,
    PedestrianDensitySpace,
)

from ..resnet.resnet import DRL_VO_ROSNAV_EXTRACTOR
from .action import previous_command
from .response_model import CrowdResponseNet


class MeanFieldResNetExtractor(DRL_VO_ROSNAV_EXTRACTOR):
    """AGENT_5 extractor plus the one-second density forecast.

    The forecast uses the density observation and the previous command
    ``[linear.x, angular.z]``. Its 16×16 output is upsampled to the pedestrian
    map and appended as one more channel. ``conv1`` therefore reads the laser,
    the existing pedestrian maps, and this channel.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        stack_size: int = 1,
        conditioned: bool = True,
        **kwargs,
    ) -> None:
        self._conditioned = conditioned
        super().__init__(
            observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
            **kwargs,
        )

    def _setup_network(self, inplanes: int = 64):
        super()._setup_network(inplanes)
        # The parent convolution is laser + pedestrian maps. This one adds the forecast.
        self.conv1 = nn.Conv2d(
            self.num_pedestrian_feature_maps + 2,
            inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.response_net = CrowdResponseNet(conditioned=self._conditioned)

    def forward(self, observations) -> torch.Tensor:
        inputs = self._get_input(observations)
        ped_map = inputs["ped_map"]
        if ped_map is None:
            raise RuntimeError("MeanFieldResNetExtractor requires the AGENT_5 pedestrian maps")
        density = observations[PedestrianDensitySpace.name]
        velocity = previous_command(observations[LastActionSpace.name])
        prediction = self.response_net(
            density[:, DENSITY_CURRENT : DENSITY_CURRENT + 1],
            density[:, DENSITY_PREVIOUS : DENSITY_PREVIOUS + 1],
            velocity,
        )
        upsampled = F.interpolate(
            prediction,
            size=ped_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        inputs["ped_map"] = torch.cat((ped_map, upsampled), dim=1)
        return self._forward_impl(**inputs)
