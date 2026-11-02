"""Tests for the mean-field pedestrian density rasterizer.

Loaded from the file path so the suite can run without the ROS messages that
``import rosnav_rl`` pulls in (``nav2_msgs`` and the rest).
"""

import importlib.util
from pathlib import Path

import numpy as np

_DENSITY = (
    Path(__file__).resolve().parents[1]
    / "rosnav_rl/model/stable_baselines3/policy/feature_extractors/mean_field/density.py"
)
_spec = importlib.util.spec_from_file_location("mean_field_density", _DENSITY)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
rasterize_density = _module.rasterize_density
next_density_observation = _module.next_density_observation


def test_no_pedestrians_is_zero():
    density = rasterize_density(np.zeros((1, 0, 2)))
    assert density.shape == (1, 1, 16, 16)
    assert np.all(density == 0)


def test_pedestrian_at_origin_peaks_at_grid_center():
    density = rasterize_density(np.array([[0.0, 0.0]]))
    center = 16 // 2
    peak = np.unravel_index(density[0, 0].argmax(), density.shape[-2:])
    assert peak == (center, center)
    assert density[0, 0, center, center] > density[0, 0, center, center + 1]
    assert density[0, 0, center, center] > density[0, 0, center + 1, center]


def test_pedestrian_outside_roi_is_ignored():
    # roi_m=20 is the full width, so ±10 m is the window. The positive edge
    # is excluded, matching BaseFeatureMapSpace.
    outside = rasterize_density(np.array([[10.0, 0.0], [0.0, -11.0]]))
    assert np.all(outside == 0)

    inside = rasterize_density(np.array([[-10.0, 0.0]]))
    assert inside.sum() > 0


def test_multiple_pedestrians_add():
    first = np.array([[0.0, 0.0]])
    second = np.array([[2.0, -1.0]])
    both = np.concatenate([first, second], axis=0)
    summed = rasterize_density(first) + rasterize_density(second)
    np.testing.assert_allclose(rasterize_density(both), summed, rtol=1e-5, atol=1e-6)


def test_batch_shape_and_independence():
    positions = np.array(
        [
            [[0.0, 0.0], [3.0, 0.0]],
            [[-2.0, 1.5], [50.0, 50.0]],
        ]
    )
    valid = np.array([[True, True], [True, False]])
    density = rasterize_density(positions, valid=valid)
    assert density.shape == (2, 1, 16, 16)
    np.testing.assert_allclose(
        density[0],
        rasterize_density(positions[0])[0],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        density[1],
        rasterize_density(positions[1, :1])[0],
        rtol=1e-5,
        atol=1e-6,
    )


def test_density_observation_keeps_the_previous_frame():
    previous = np.zeros((16, 16), dtype=np.float32)
    first, current = next_density_observation(previous, np.array([[0.0, 0.0]]))
    assert first.shape == (2, 16, 16)
    assert np.all(first[1] == 0)
    np.testing.assert_array_equal(first[0], current)

    second, _ = next_density_observation(current, np.array([[2.0, 0.0]]))
    np.testing.assert_array_equal(second[1], first[0])
    assert not np.allclose(second[0], second[1])

    reset, _ = next_density_observation(np.zeros_like(current), np.array([[0.0, 0.0]]))
    assert np.all(reset[1] == 0)


def test_empty_detection_is_a_zero_frame():
    observation, current = next_density_observation(
        np.zeros((16, 16), dtype=np.float32),
        np.array([]),
    )
    assert observation.shape == (2, 16, 16)
    assert np.all(current == 0)


def test_origin_uses_the_same_cell_as_the_feature_map():
    # index = position * (size / roi) + size // 2, the BaseFeatureMapSpace formula.
    grid_size = 16
    roi_m = 20.0
    position = np.array([[1.3, -2.4]])
    expected = (
        position * (grid_size / roi_m) + (grid_size // 2)
    ).astype(np.intp)
    density = rasterize_density(position, grid_size=grid_size, roi_m=roi_m, sigma=0.25)
    peak = np.unravel_index(density[0, 0].argmax(), (grid_size, grid_size))
    assert peak == (int(expected[0, 1]), int(expected[0, 0]))
