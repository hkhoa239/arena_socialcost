"""Previous velocity command used to condition the crowd forecast.

The forecast is :math:`F(\\rho_t, \\rho_{t-1}, a_{t-1})` at training and at
inference. ``a_{t-1}`` is the command the robot already sent, ``[linear.x,
angular.z]`` in m/s and rad/s. A twist stored by ``TwistCollector`` is
``[linear.x, linear.y, angular.z]``; the middle value is lateral speed and is
not the yaw rate.
"""

from __future__ import annotations

import numpy as np


def previous_command(last_action: np.ndarray) -> np.ndarray:
    """Return ``[linear.x, angular.z]`` from the last commanded twist.

    ``(..., 3)`` is ``[linear.x, linear.y, angular.z]``. ``(..., 2)`` is already
    ``[linear.x, angular.z]``.
    """
    width = last_action.shape[-1]
    if width == 3:
        index = (0, 2)
    elif width == 2:
        index = (0, 1)
    else:
        raise ValueError(
            "last action must have 2 or 3 components, "
            f"[linear.x, angular.z] or [linear.x, linear.y, angular.z], got {width}"
        )
    if isinstance(last_action, np.ndarray):
        return last_action[..., index]
    return last_action[..., list(index)]
