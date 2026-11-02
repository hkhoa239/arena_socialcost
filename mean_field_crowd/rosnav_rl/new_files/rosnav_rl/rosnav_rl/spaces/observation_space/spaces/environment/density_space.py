"""Pedestrian density observation.

The gym value is raw density, not a normalized feature map. Channel 0 is the
current frame and channel 1 is the previous frame of the same episode. The
history lives here, at collection time, because a PPO minibatch does not visit
timesteps in order.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)

DENSITY_CURRENT = 0
DENSITY_PREVIOUS = 1


def _as_positions(positions: np.ndarray) -> np.ndarray:
    """Return ``(N, 2)`` robot-frame meters. An empty detection is ``(0, 2)``."""
    array = np.asarray(positions, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[-1] != 2:
        raise ValueError(
            f"pedestrian_relative_locations must have shape (N, 2), got {array.shape}"
        )
    return array


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianDensitySpace(BaseObservationSpace):
    """Density of pedestrians in the robot frame, plus the previous frame.

    ``requires`` reads ``pedestrian_relative_locations`` from
    ``PedestrianRelativeLocationGenerator``. ``roi_in_m`` is the full width of
    the window, the same meaning as on the 80×80 feature maps. Shared agent
    kwargs such as ``feature_map_size`` are ignored so this grid stays 16×16
    when an agent also requests the 80×80 maps.

    The encoded value is not passed through the observation normalizer. The
    rollout buffer needs the same non-negative density the response network
    predicts.
    """

    name = "PedestrianDensitySpace"
    requires = {"pedestrian_relative_locations": np.ndarray}

    def __init__(
        self,
        grid_size: int = 16,
        roi_in_m: float = 20.0,
        sigma: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        self._grid_size = grid_size
        self._roi_in_m = roi_in_m
        self._sigma = sigma
        self._previous = np.zeros((grid_size, grid_size), dtype=np.float32)
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=np.float32(64.0),
            shape=(2, self._grid_size, self._grid_size),
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._previous[...] = 0.0

    def encode_observation(
        self,
        pedestrian_relative_locations: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        # Imported here so this space can be registered while the extractor package is still loading.
        from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.density import (
            next_density_observation,
        )

        stacked, current = next_density_observation(
            self._previous,
            _as_positions(pedestrian_relative_locations),
            grid_size=self._grid_size,
            roi_m=self._roi_in_m,
            sigma=self._sigma,
        )
        self._previous = current
        return stacked
