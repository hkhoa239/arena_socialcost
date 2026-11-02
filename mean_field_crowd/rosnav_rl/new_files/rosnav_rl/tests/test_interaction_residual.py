"""Integration checks for the frozen PedVO residual forecast."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from rosnav_rl.model.stable_baselines3.policy.agent_factory import AgentFactory
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.action import (
    previous_command,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.interaction import (
    InteractionResidual,
    constant_velocity_points,
    forecast_density,
    residual_checkpoint,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field import interaction as interaction_module
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.residual_extractor import (
    ResidualForecastExtractor,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.resnet.resnet import (
    DRL_VO_ROSNAV_EXTRACTOR,
)
from rosnav_rl.observations.utils.semantic import get_relative_pos_to_robot, get_relative_vel_to_robot
from rosnav_rl.spaces.observation_space.spaces.environment.pedestrian_state_space import (
    ForecastCommandSpace,
    PedestrianKinematicsSpace,
)

DATA_ROOTS = (
    Path("/opt/arena_ws/data"),
    Path("/home/robotics/arena_ws/data"),
)
for root in DATA_ROOTS:
    if (root / "mf_pedvo_residual.py").is_file() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _space_kwargs():
    return {
        "roi_in_m": 20,
        "feature_map_size": 80,
        "laser_stack_size": 10,
        "normalize": True,
        "normalizer": "max_abs",
        "goal_max_dist": 10,
        "subgoal_max_dist": 10,
    }


def _observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "StackedLaserMapSpace": spaces.Box(-1, 1, shape=(1, 80, 80), dtype=np.float32),
            "PedestrianVelXSpace": spaces.Box(-1, 1, shape=(80, 80), dtype=np.float32),
            "PedestrianVelYSpace": spaces.Box(-1, 1, shape=(80, 80), dtype=np.float32),
            "PedestrianTypeSpace": spaces.Box(-1, 1, shape=(80, 80), dtype=np.float32),
            "PedestrianSocialStateSpace": spaces.Box(-1, 1, shape=(80, 80), dtype=np.float32),
            "DistAngleToSubgoalSpace": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            "LastActionSpace": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            "PedestrianKinematicsSpace": spaces.Box(-50, 50, shape=(20, 5), dtype=np.float32),
            "ForecastCommandSpace": spaces.Box(-5, 5, shape=(4,), dtype=np.float32),
        }
    )


def _observations(batch: int = 2, with_pedestrian: bool = True) -> dict[str, torch.Tensor]:
    observations = {
        "StackedLaserMapSpace": torch.zeros(batch, 1, 80, 80),
        "PedestrianVelXSpace": torch.zeros(batch, 80, 80),
        "PedestrianVelYSpace": torch.zeros(batch, 80, 80),
        "PedestrianTypeSpace": torch.zeros(batch, 80, 80),
        "PedestrianSocialStateSpace": torch.zeros(batch, 80, 80),
        "DistAngleToSubgoalSpace": torch.zeros(batch, 2),
        "LastActionSpace": torch.tensor([[0.4, 0.0, -0.2]]).expand(batch, 3).clone(),
        "PedestrianKinematicsSpace": torch.zeros(batch, 20, 5),
        "ForecastCommandSpace": torch.tensor([[0.4, -0.2, 1.0, 0.0]]).expand(batch, 4).clone(),
    }
    if with_pedestrian:
        observations["PedestrianKinematicsSpace"][0, 0] = torch.tensor([4.0, 0.2, -1.1, 0.05, 1.0])
    return observations


def _episode():
    import mf_pedvo_forecast as forecast
    import mf_pedvo_residual as residual

    path = next(Path(forecast.DATA).glob("n1/seed8_block_*.npz"))
    blob = np.load(path)
    ped = blob["ped"]
    robot = blob["robot"]
    yaw = blob["yaw"]
    relative = forecast.relatives(robot, yaw, ped)
    episode = {
        "count": 1,
        "seed": 8,
        "name": "block",
        "repeat": 0,
        "ped": ped,
        "vel": blob["vel"],
        "robot": robot,
        "yaw": yaw,
        "command": blob["command"],
        "relative": relative,
        "density": forecast.rasterize_density(relative)[:, 0],
        "center": 0,
        "spawn_y": ped[0, :, 1].copy(),
    }
    return residual, residual.episode_batch(episode)


def test_robot_frame_coordinates_and_velocities_are_preserved():
    pose = {"x": 17.3, "y": 7.3, "yaw": 0.4}
    world = np.array([[21.3, 7.3, 1.0], [18.0, 8.1, 1.0]])
    relative = get_relative_pos_to_robot(pose, world)
    velocity = get_relative_vel_to_robot(pose, np.array([[-1.1, 0.15], [0.2, 0.9]]))
    yaw = pose["yaw"]
    expected = np.column_stack(
        (
            np.cos(yaw) * (world[:, 0] - pose["x"]) + np.sin(yaw) * (world[:, 1] - pose["y"]),
            -np.sin(yaw) * (world[:, 0] - pose["x"]) + np.cos(yaw) * (world[:, 1] - pose["y"]),
        )
    )
    np.testing.assert_allclose(relative, expected, atol=1e-6)
    space = PedestrianKinematicsSpace(**_space_kwargs())
    encoded = space.encode_observation(relative, velocity[:, 0], velocity[:, 1])
    assert encoded.shape == (20, 5)
    order = np.argsort(np.linalg.norm(relative, axis=1))
    np.testing.assert_allclose(encoded[:2, 0:2], relative[order], atol=1e-6)
    np.testing.assert_allclose(encoded[:2, 2:4], velocity[order], atol=1e-6)
    assert np.all(encoded[:2, 4] == 1.0)
    assert np.all(encoded[2:] == 0.0)


def test_variable_empty_and_reset_observations():
    space = PedestrianKinematicsSpace(**_space_kwargs())
    empty = space.encode_observation(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    assert empty.shape == (20, 5)
    assert np.all(empty == 0.0)
    positions = np.array([[i, 0.0] for i in range(25)], dtype=np.float32)
    speeds = np.linspace(-1.0, 1.0, 25)
    encoded = space.encode_observation(positions, speeds, np.zeros(25))
    assert encoded[:, 4].sum() == 20
    assert np.linalg.norm(encoded[0, 0:2]) <= np.linalg.norm(encoded[1, 0:2])
    space.reset()
    again = space.encode_observation(positions, speeds, np.zeros(25))
    np.testing.assert_array_equal(encoded, again)


def test_previous_command_is_linear_x_and_angular_z():
    twist = np.array([0.45, 0.2, -0.3], dtype=np.float32)
    np.testing.assert_allclose(previous_command(twist), [0.45, -0.3])
    np.testing.assert_allclose(previous_command(np.array([0.45, -0.3], dtype=np.float32)), [0.45, -0.3])
    command_space = ForecastCommandSpace(**_space_kwargs())
    encoded = command_space.encode_observation(twist, robot_pose={"x": 1.0, "y": 2.0, "yaw": np.pi / 2})
    np.testing.assert_allclose(encoded[:2], [0.45, -0.3])
    np.testing.assert_allclose(encoded[2:], [0.0, -1.0], atol=1e-6)
    assert command_space.encode_observation.__func__ is not None
    assert "apply_normalization" not in type(command_space.encode_observation).__name__


def test_forecast_uses_no_future_information():
    parameters = inspect.signature(forecast_density).parameters
    assert set(parameters) == {"module", "positions", "velocities", "valid", "command", "apply_correction", "slow_heading"}
    positions = torch.tensor([[[3.0, 0.1]]])
    velocities = torch.tensor([[[-1.0, 0.0]]])
    valid = torch.tensor([[True]])
    command = torch.tensor([[0.0, 0.0]])
    first = forecast_density(None, positions, velocities, valid, command, False)
    second = forecast_density(None, positions.clone(), velocities.clone(), valid, command.clone(), False)
    torch.testing.assert_close(first, second)
    pose = constant_velocity_points(positions, velocities, command)[1]
    torch.testing.assert_close(pose, torch.zeros(1, 3))


def test_online_and_residual_share_the_kinematic_baseline():
    positions = torch.zeros(1, 2, 2)
    positions[0, 0] = torch.tensor([3.0, -0.4])
    velocities = torch.zeros(1, 2, 2)
    velocities[0, 0] = torch.tensor([-0.8, 0.1])
    valid = torch.tensor([[True, False]])
    command = torch.tensor([[0.3, 0.4]])
    calls = []
    real = interaction_module.constant_velocity_points

    def record(*args, **kwargs):
        calls.append(args[0].detach().clone())
        return real(*args, **kwargs)

    interaction_module.constant_velocity_points = record
    try:
        online = forecast_density(None, positions, velocities, valid, command, False)
        module = InteractionResidual()
        module.net[-1].weight.data.zero_()
        module.net[-1].bias.data.zero_()
        module.net[-1].bias.data[0] = -20.0
        corrected = forecast_density(module, positions, velocities, valid, command, True)
    finally:
        interaction_module.constant_velocity_points = real
    assert len(calls) == 2
    torch.testing.assert_close(calls[0], calls[1])
    torch.testing.assert_close(online, corrected, atol=1e-5, rtol=1e-5)
    assert online.shape == (1, 16, 16)
    assert torch.all(online >= 0)


def test_empty_crowd_forecast_is_zero():
    density = forecast_density(
        None,
        torch.zeros(2, 20, 2),
        torch.zeros(2, 20, 2),
        torch.zeros(2, 20, dtype=torch.bool),
        torch.zeros(2, 2),
        False,
    )
    assert density.shape == (2, 16, 16)
    assert torch.count_nonzero(density) == 0


def test_upsample_is_the_sixth_channel_and_policy_gets_gradients():
    observation_space = _observation_space()
    extractor = ResidualForecastExtractor(
        observation_space,
        features_dim=512,
        width_per_group=64,
        apply_correction=False,
    )
    assert extractor.conv1.in_channels == 6
    captured = {}

    def remember(_module, inputs, _output):
        captured["channels"] = inputs[0].shape

    extractor.conv1.register_forward_hook(remember)
    observations = _observations()
    features = extractor(observations)
    assert features.shape == (2, 512)
    assert captured["channels"] == (2, 6, 80, 80)
    online, _ = extractor._forecast(observations)
    assert online.shape == (2, 16, 16)
    upsampled = F.interpolate(online.unsqueeze(1), size=(80, 80), mode="bilinear", align_corners=False)
    assert upsampled.shape == (2, 1, 80, 80)

    policy = MultiInputActorCriticPolicy(
        observation_space=observation_space,
        action_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        lr_schedule=lambda _progress: 1e-3,
        features_extractor_class=ResidualForecastExtractor,
        features_extractor_kwargs={"features_dim": 512, "width_per_group": 64, "apply_correction": False},
        net_arch=dict(pi=[256, 128], vf=[256, 64]),
        activation_fn=nn.ReLU,
    )
    values = policy.predict_values(observations)
    distribution = policy.get_distribution(observations)
    loss = values.sum() + distribution.distribution.mean.sum()
    loss.backward()
    assert torch.isfinite(values).all()
    assert policy.features_extractor.conv1.weight.grad is not None
    assert policy.features_extractor.conv1.weight.grad.abs().sum() > 0
    assert policy.action_net.weight.grad is not None
    assert policy.action_net.weight.grad.abs().sum() > 0
    assert policy.features_extractor.residual.training is False


def test_residual_checkpoint_stays_frozen_and_reloads():
    path = residual_checkpoint()
    assert path.is_file()
    extractor = ResidualForecastExtractor(
        _observation_space(),
        features_dim=512,
        width_per_group=64,
        apply_correction=True,
        residual_checkpoint_path=str(path),
    )
    assert sum(parameter.numel() for parameter in extractor.residual.parameters()) == 211
    before = {name: value.detach().clone() for name, value in extractor.residual.state_dict().items()}
    extractor.train()
    assert extractor.residual.training is False
    assert extractor.conv1.training is True
    observations = _observations()
    features = extractor(observations)
    features.sum().backward()
    optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-2)
    optimizer.step()
    for name, value in extractor.residual.state_dict().items():
        assert value.requires_grad is False
        torch.testing.assert_close(value, before[name])
    assert extractor.conv1.weight.grad is not None

    reloaded = ResidualForecastExtractor(
        _observation_space(),
        features_dim=512,
        width_per_group=64,
        apply_correction=True,
        residual_checkpoint_path=str(path),
    )
    reloaded.load_state_dict(extractor.state_dict())
    reloaded.train()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(observations), extractor(observations))
    for name, value in reloaded.residual.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_residual_matches_offline_inference_on_a_held_out_episode():
    path = residual_checkpoint()
    if not path.is_file():
        pytest.skip("residual checkpoint is not available")
    try:
        residual, batch = _episode()
    except Exception as error:
        pytest.skip(f"held-out episode is not available: {error}")
    module = residual.InteractionResidual()
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    module.eval()
    packed = residual.tensors(batch)
    with torch.no_grad():
        offline, _gate = residual.mix_density(
            module,
            packed["features"],
            torch.as_tensor(batch["valid"]),
            packed["ped_cv"],
            packed["n_fwd"],
            packed["n_lat"],
            packed["pose"],
        )
    index = np.arange(1, len(batch["action"]) + 1)
    # episode_batch already selected the horizon pairs. Rebuild the robot-frame inputs in that same order.
    episode_path = next(Path("/opt/arena_ws/data/mf_pedvo").glob("n1/seed8_block_*.npz"), None)
    host_path = next(Path("/home/robotics/arena_ws/data/mf_pedvo").glob("n1/seed8_block_*.npz"), None)
    source = np.load(episode_path or host_path)
    import mf_pedvo_forecast as forecast

    relative = forecast.relatives(source["robot"], source["yaw"], source["ped"])
    pair = np.arange(1, len(relative) - 10)
    yaw = source["yaw"][pair]
    velocity = source["vel"][pair]
    cos_yaw = np.cos(yaw)[:, None]
    sin_yaw = np.sin(yaw)[:, None]
    robot_velocity = np.stack(
        (
            cos_yaw * velocity[..., 0] + sin_yaw * velocity[..., 1],
            -sin_yaw * velocity[..., 0] + cos_yaw * velocity[..., 1],
        ),
        axis=-1,
    )
    positions = torch.as_tensor(relative[pair], dtype=torch.float32)
    velocities = torch.as_tensor(robot_velocity, dtype=torch.float32)
    command = torch.as_tensor(source["command"][pair - 1], dtype=torch.float32)
    valid = torch.ones(positions.shape[:2], dtype=torch.bool)
    heading = torch.stack((torch.cos(torch.as_tensor(yaw)), -torch.sin(torch.as_tensor(yaw))), dim=-1)
    local = InteractionResidual()
    local.load_state_dict(module.state_dict())
    local.eval()
    with torch.no_grad():
        online = forecast_density(None, positions, velocities, valid, command, False, heading)
        corrected = forecast_density(local, positions, velocities, valid, command, True, heading)
    torch.testing.assert_close(corrected, offline.float(), atol=1e-4, rtol=1e-4)
    assert (corrected - online).abs().max() > 1e-5
    assert index[0] == 1


def test_mirror_reflects_kinematics_and_the_raw_command():
    from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
    from rosnav_rl.cfg.agent import AgentConfig
    from rosnav_rl.cfg.parameters import AgentParameters
    from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
    from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
    from rosnav_rl.rl_agent import RL_Agent

    beams = 720
    spec = AgentConfig(
        name="mirror_forecast",
        action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
        parameters=AgentParameters(
            laser_num_beams=beams,
            laser_angle_min=-np.pi,
            laser_angle_increment=2 * np.pi / beams,
            robot_radius=0.4,
            min_translational_vel=0.0,
            max_translational_vel=0.0,
        ),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_MF_ONLINE", parameters=PPO_Algorithm_Cfg(device="cpu", seed=5))
        ),
        mirror="canonical",
    )
    manager = RL_Agent(spec).space_manager
    scene = {
        "front_laser": np.linspace(1.0, 9.0, beams, dtype=np.float32),
        "is_terminal": False,
        "pedestrian_relative_locations": np.array([[2.0, -1.0]], dtype=np.float32),
        "pedestrian_relative_velocities": np.array([[-0.4, 0.3]], dtype=np.float32),
        "pedestrian_vel_x": np.array([-0.4], dtype=np.float32),
        "pedestrian_vel_y": np.array([0.3], dtype=np.float32),
        "pedestrian_types": np.zeros(1, dtype=np.int32),
        "pedestrian_social_states": np.zeros(1, dtype=np.int32),
        "dist_angle_to_subgoal": np.array([3.0, 0.2], dtype=np.float32),
        "last_action": np.array([0.4, -0.25], dtype=np.float32),
        "robot_pose": {"x": 17.3, "y": 7.3, "yaw": 0.3},
    }
    encoded = manager.encode_observation(scene)
    kinematics = encoded["PedestrianKinematicsSpace"]
    np.testing.assert_allclose(kinematics[0, 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(kinematics[0, 1], 1.0, atol=1e-5)
    np.testing.assert_allclose(kinematics[0, 3], -0.3, atol=1e-5)
    command = encoded["ForecastCommandSpace"]
    np.testing.assert_allclose(command[0], 0.4, atol=1e-6)
    np.testing.assert_allclose(command[1], 0.25, atol=1e-6)
    np.testing.assert_allclose(command[2], np.cos(0.3), atol=1e-6)
    np.testing.assert_allclose(command[3], np.sin(0.3), atol=1e-6)


def test_pretrained_residual_survives_the_policy_weight_initialization():
    path = residual_checkpoint()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    extractor = ResidualForecastExtractor(
        _observation_space(),
        features_dim=512,
        width_per_group=64,
        apply_correction=True,
        residual_checkpoint_path=str(path),
    )
    for module in extractor.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.orthogonal_(module.weight)
            module.bias.data.zero_()
    extractor(_observations())
    for key, value in checkpoint.items():
        torch.testing.assert_close(extractor.residual.state_dict()[key], value)
    assert extractor._residual_ready is True


def test_agent_registrations_leave_agent_5_unchanged():
    import rosnav_rl.model.stable_baselines3.policy.sb3_policy.paper as paper

    agent_5 = AgentFactory.instantiate("AGENT_5")
    online = AgentFactory.instantiate("AGENT_MF_ONLINE")
    residual = AgentFactory.instantiate("AGENT_MF_RESIDUAL")
    assert agent_5.features_extractor_class is DRL_VO_ROSNAV_EXTRACTOR
    assert agent_5.algorithm_class is PPO
    assert "PedestrianKinematicsSpace" not in [space.__name__ for space in agent_5.observation_spaces]
    assert online.algorithm_class is PPO
    assert residual.algorithm_class is PPO
    assert online.net_arch == agent_5.net_arch
    assert residual.net_arch == agent_5.net_arch
    assert online.activation_fn is agent_5.activation_fn
    assert online.observation_space_kwargs == agent_5.observation_space_kwargs
    assert online.features_extractor_kwargs["apply_correction"] is False
    assert residual.features_extractor_kwargs["apply_correction"] is True
    assert [space.__name__ for space in online.observation_spaces] == [
        *[space.__name__ for space in agent_5.observation_spaces],
        "PedestrianKinematicsSpace",
        "ForecastCommandSpace",
    ]
    assert residual.observation_spaces == online.observation_spaces
    assert paper.AGENT_MF.algorithm_class.__name__ == "MeanFieldPPO"
    assert paper.AGENT_MF_FORECAST.features_extractor_kwargs["conditioned"] is False
