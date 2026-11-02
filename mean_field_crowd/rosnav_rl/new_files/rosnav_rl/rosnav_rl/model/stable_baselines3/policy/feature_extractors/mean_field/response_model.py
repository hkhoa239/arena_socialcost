"""One-second crowd-density forecast.

The two variants share this module. ``conditioned=False`` still runs the
four-channel network, with the velocity channels set to zero, so both variants
have the same number of parameters.

``velocity`` is the previous command ``[linear.x, angular.z]``, shaped
``(B, 2)``, in m/s and rad/s. Training and inference both use that previous
command. ``conditioned=False`` zeros it and keeps the same weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrowdResponseNet(nn.Module):
    """Predict a 16×16 density from the current density, the previous density, and a velocity."""

    def __init__(self, conditioned: bool = True) -> None:
        super().__init__()
        self.conditioned = conditioned
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Softplus(),
        )

    def forward(
        self,
        rho_t: torch.Tensor,
        rho_prev: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = rho_t.shape
        if velocity.shape != (batch, 2):
            raise ValueError(
                f"velocity must have shape {(batch, 2)}, got {tuple(velocity.shape)}"
            )
        if not self.conditioned:
            velocity = torch.zeros_like(velocity)
        vel_maps = velocity.reshape(batch, 2, 1, 1).expand(-1, -1, height, width)
        features = torch.cat([rho_t, rho_prev, vel_maps], dim=1)
        return self.network(features)
