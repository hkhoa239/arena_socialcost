"""Current-distance penalty: zero outside the margin, proportional inside, silent with nobody present."""

import numpy as np
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.reward.reward_function import RewardFunction

TERMS = {"pedestrian_proximity": {"margin": 1.0, "factor": -0.5}}


def _reward(positions: np.ndarray) -> float:
    parameters = AgentParameters(safety_distance=1.0)
    reward_function = RewardFunction(function_dict=TERMS)
    reward_function.get_reward(
        obs_dict={"pedestrian_relative_locations": positions, "laser_safety_violation": False},
        simulation_state_container=parameters,
    )
    return float(reward_function.state.reward_overview.get("RewardPedestrianProximity", 0.0))


def test_nobody_and_far_pedestrians_are_free():
    assert _reward(np.zeros((0, 2))) == 0.0
    assert _reward(np.array([[2.0, 0.0], [0.0, 3.0]])) == 0.0


def test_intrusion_is_charged_from_the_nearest_person():
    assert abs(_reward(np.array([[0.5, 0.0], [4.0, 0.0]])) - (-0.25)) < 1e-6


FORECAST = {"pedestrian_forecast_proximity": {"margin": 1.0, "factor": -0.5, "horizon": 1.0}}


def _forecast(positions: np.ndarray, velocities: np.ndarray) -> float:
    parameters = AgentParameters(safety_distance=1.0)
    reward_function = RewardFunction(function_dict=FORECAST)
    reward_function.get_reward(
        obs_dict={
            "pedestrian_relative_locations": positions,
            "pedestrian_relative_velocities": velocities,
            "laser_safety_violation": False,
        },
        simulation_state_container=parameters,
    )
    return float(reward_function.state.reward_overview.get("RewardPedestrianForecastProximity", 0.0))


def test_forecast_charges_a_person_who_will_enter_and_spares_one_who_walks_away():
    assert abs(_forecast(np.array([[1.5, 0.0]]), np.array([[-1.0, 0.0]])) - (-0.25)) < 1e-6
    assert _forecast(np.array([[0.5, 0.0]]), np.array([[1.0, 0.0]])) == 0.0
