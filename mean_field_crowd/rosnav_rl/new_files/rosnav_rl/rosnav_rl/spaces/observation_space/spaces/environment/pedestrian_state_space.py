"""Raw pedestrian state for the frozen crowd forecast.

The 80×80 velocity maps cannot be inverted back to the positions and
velocities the residual was trained on. This space keeps those values in the
robot frame, unnormalized, and pads a variable crowd to a fixed length.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)

MAX_PEDESTRIANS = 20


def _positions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[-1] != 2:
        raise ValueError(f"pedestrian positions must have shape (N, 2), got {array.shape}")
    return array


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class PedestrianKinematicsSpace(BaseObservationSpace):
    """Robot-frame position, robot-frame velocity, and a validity flag.

    Columns are ``x, y, vx, vy, valid``. Empty detections and padded rows are
    valid=0. At most 20 pedestrians are kept, nearest first. The values stay
    in meters and meters per second. The observation normalizer is not applied.
    """

    name = "PedestrianKinematicsSpace"
    requires = {
        "pedestrian_relative_locations": np.ndarray,
        "pedestrian_vel_x": np.ndarray,
        "pedestrian_vel_y": np.ndarray,
    }

    def __init__(self, max_pedestrians: int = MAX_PEDESTRIANS, *args, **kwargs) -> None:
        self._max_pedestrians = max_pedestrians
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        low = np.array([-50.0, -50.0, -10.0, -10.0, 0.0], dtype=np.float32)
        high = np.array([50.0, 50.0, 10.0, 10.0, 1.0], dtype=np.float32)
        return spaces.Box(
            low=np.broadcast_to(low, (self._max_pedestrians, 5)).copy(),
            high=np.broadcast_to(high, (self._max_pedestrians, 5)).copy(),
            dtype=np.float32,
        )

    def encode_observation(
        self,
        pedestrian_relative_locations: np.ndarray,
        pedestrian_vel_x: np.ndarray,
        pedestrian_vel_y: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        positions = _positions(pedestrian_relative_locations)
        speed_x = np.asarray(pedestrian_vel_x, dtype=np.float64).reshape(-1)
        speed_y = np.asarray(pedestrian_vel_y, dtype=np.float64).reshape(-1)
        count = min(len(positions), len(speed_x), len(speed_y))
        encoded = np.zeros((self._max_pedestrians, 5), dtype=np.float32)
        if count == 0:
            return encoded
        positions = positions[:count]
        order = np.argsort(np.linalg.norm(positions, axis=1))[: self._max_pedestrians]
        encoded[: len(order), 0:2] = positions[order]
        encoded[: len(order), 2] = speed_x[order]
        encoded[: len(order), 3] = speed_y[order]
        encoded[: len(order), 4] = 1.0
        return encoded


@SpaceFactory.register(auto_name=True, category=SpaceCategory.DYNAMICS)
class ForecastCommandSpace(BaseObservationSpace):
    """Previous command in physical units, plus the map-forward axis in the robot frame.

    The first two values are ``[linear.x, angular.z]``. ``LastActionSpace`` is
    normalized for the policy head, and the forecast was trained on meters per
    second and radians per second. The last two values are map +x expressed in
    the robot frame. The residual uses that axis only for a pedestrian slower
    than 0.2 m/s. This observation is not normalized.
    """

    name = "ForecastCommandSpace"
    requires = {"last_action": np.ndarray, "robot_pose": dict}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        low = np.array([-5.0, -5.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([5.0, 5.0, 1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def encode_observation(self, last_action: np.ndarray, robot_pose: dict, *args, **kwargs) -> np.ndarray:
        # Imported here so observation-space loading does not pull the ResNet extractor back in.
        from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.action import (
            previous_command,
        )

        command = np.asarray(previous_command(np.asarray(last_action, dtype=np.float32)), dtype=np.float32)
        yaw = float(robot_pose["yaw"])
        # Map +x, expressed in the robot frame. The residual uses this only when a pedestrian is slower than 0.2 m/s.
        heading = np.array([np.cos(yaw), -np.sin(yaw)], dtype=np.float32)
        return np.concatenate((command.reshape(2), heading))
