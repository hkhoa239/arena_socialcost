"""Forward-only normalized action space and the per-episode scenario mixture."""

from __future__ import annotations

import numpy as np
import pytest
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.utils.curriculum.curriculum_base import CurriculumBase
from rosnav_rl.utils.stable_baselines3.tick_callbacks import EpisodeMixtureCurriculum


def test_normalized_space_maps_the_unit_box_onto_the_ranges():
    forward = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5), normalized=True)
    box = forward.get_gym_space()
    assert box.low.tolist() == [-1.0, -1.0] and box.high.tolist() == [1.0, 1.0]
    assert forward.decode(np.array([-1.0, -1.0])).tolist() == pytest.approx([0.0, 0.0, -1.5])
    assert forward.decode(np.array([0.0, 0.0])).tolist() == pytest.approx([0.25, 0.0, 0.0])
    assert forward.decode(np.array([1.0, 1.0])).tolist() == pytest.approx([0.5, 0.0, 1.5])
    # Out-of-box samples execute as the box edge, never as reverse.
    assert forward.decode(np.array([-3.0, 0.0]))[0] == pytest.approx(0.0)


def test_unrestricted_normalized_space_differs_only_in_the_linear_range():
    forward = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5), normalized=True)
    free = DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.5, 1.5), normalized=True)
    assert forward.get_gym_space() == free.get_gym_space()
    u = np.array([0.3, -0.4])
    assert forward.decode(u)[2] == pytest.approx(free.decode(u)[2])
    assert free.decode(np.array([-1.0, 0.0]))[0] == pytest.approx(-0.5)


def test_default_space_is_unchanged():
    native = DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.5, 1.5))
    assert native.get_gym_space().low.tolist() == [-0.5, -1.5]
    assert native.decode(np.array([0.2, -0.7])).tolist() == pytest.approx([0.2, 0.0, -0.7])


STAGES = [{"task.random.dynamic.n": [0, 0]}, {"task.random.dynamic.n": [1, 1]}, {"task.random.dynamic.n": [3, 5]}, {"task.random.dynamic.n": [10, 15]}]


@pytest.fixture
def mixture(monkeypatch):
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(CurriculumBase, "_init_parameter_clients", lambda self: {f"/arena/env_{i}/tg": None for i in range(self.num_envs)})
    monkeypatch.setattr(CurriculumBase, "_init_task_mode_clients", lambda self: {})
    monkeypatch.setattr(CurriculumBase, "_apply_curriculum", lambda self: True)
    monkeypatch.setattr(CurriculumBase, "_set_parameters_batch", lambda self, node, params: sent.append((node, params)) or True)

    def build(seed: int = 5) -> tuple[EpisodeMixtureCurriculum, list]:
        sent.clear()
        cb = EpisodeMixtureCurriculum(node=None, train_stages=STAGES, weights=[0.25, 0.25, 0.35, 0.15], seed=seed, num_envs=2, parameter_node_template="/arena/env_{i}/tg")
        return cb, sent

    return build


def _run(cb, episodes: int) -> list[int]:
    cb._on_training_start()
    for _ in range(episodes):
        cb.locals = {"dones": np.array([True, False])}
        cb._on_step()
    return cb.started[0]


def test_mixture_targets_only_the_finished_env_and_follows_the_weights(mixture):
    cb, sent = mixture()
    counts = _run(cb, 4000)
    nodes = [node for node, _ in sent[2:]]
    assert set(nodes) == {"/arena/env_0/tg"}
    freq = np.array(counts) / sum(counts)
    assert freq == pytest.approx([0.25, 0.25, 0.35, 0.15], abs=0.03)
    # The stage sent is the stage the next-but-one episode is labelled with.
    assert sent[-1][1] == STAGES[cb.queued[0]]


def test_mixture_draws_are_reproducible_and_seed_dependent(mixture):
    first, _ = mixture(5)
    a = _run(first, 200)
    again, _ = mixture(5)
    b = _run(again, 200)
    other, _ = mixture(6)
    c = _run(other, 200)
    assert a == b
    assert a != c


def test_mixture_rejects_a_wrong_weight_count(mixture, monkeypatch):
    with pytest.raises(ValueError):
        EpisodeMixtureCurriculum(node=None, train_stages=STAGES, weights=[1.0, 1.0], seed=0, num_envs=1, parameter_node_template="/x")
