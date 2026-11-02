"""Scenario definitions of the mean-field navigation benchmark.

One goal tolerance (1.0 m) for every scenario, so success rates compare across crowd sizes.
Random scenarios draw robot start, goal, static obstacles and pedestrians from the task
generator with a seed derived from (run_seed, world, episode_id); scenario-file scenarios
fix the layout and only PedVO's behaviour varies.

Seed families are disjoint by construction:
  training      env_seed 5/6/7 -> run_seed "<env_seed>-<env index>"
  validation    run_seed "bench-val-<stage>"
  test          run_seed "bench-test-<stage>"
  forecasting   run_seed "bench-fc-<stage>"
"""

from __future__ import annotations

TOLERANCE = 1.0
PED = ["Common/Human/arenian"]
STATIC_FEW = ["Hospital/SM_Chair_01a", "Hospital/SM_BedSideTable_01b"]
STATIC_MORE = ["Hospital/SM_Chair_01a", "Hospital/SM_BedSideTable_01b", "Hospital/SM_TrashCan"]


def random_stage(dynamic: tuple[int, int], static: tuple[int, int], models: list[str], steps: int = 600) -> dict:
    return {
        "steps": steps,
        "args": [
            "task.robots:=random",
            "task.obstacles:=random",
            "task.modules:=",
            f"task.random.static.n:=[{static[0]},{static[1]}]",
            *([f"task.random.static.models:=[{','.join(models)}]"] if static[1] > 0 else []),
            "task.random.interactive.n:=[0,0]",
            f"task.random.dynamic.n:=[{dynamic[0]},{dynamic[1]}]",
            *([f"task.random.dynamic.models:=[{','.join(PED)}]"] if dynamic[1] > 0 else []),
            f"task.episode.goto_pose.tolerance.radius:={TOLERANCE}",
        ],
    }


def scenario_stage(name: str, steps: int = 450) -> dict:
    return {
        "steps": steps,
        "args": [
            "task.robots:=scenario",
            "task.obstacles:=scenario",
            "task.modules:=",
            f"task.scenario.file:={name}",
            f"task.episode.goto_pose.tolerance.radius:={TOLERANCE}",
        ],
    }


# Navigation scenarios (Phase 2 table plus the two interaction scenarios).
NAV = {
    "b_empty": random_stage((0, 0), (0, 0), []),
    "b_static": random_stage((0, 0), (2, 4), STATIC_FEW),
    "b_moderate": random_stage((3, 5), (2, 4), STATIC_FEW),
    "b_dense": random_stage((10, 15), (3, 5), STATIC_MORE),
    "b_stress": random_stage((20, 20), (3, 5), STATIC_MORE),
    "b_headon": scenario_stage("mf_headon_open"),
    "b_crossing": scenario_stage("mf_crossing"),
    "b_headon_wall": scenario_stage("mf_headon", steps=450),
}

# Forecasting data (Phase 1): crowd size sweep, static obstacles, interaction geometries.
FORECAST = {
    "fc_n1": random_stage((1, 1), (0, 0), [], steps=400),
    "fc_n5": random_stage((5, 5), (0, 0), [], steps=400),
    "fc_n10": random_stage((10, 10), (0, 0), [], steps=400),
    "fc_n15": random_stage((15, 15), (0, 0), [], steps=400),
    "fc_n20": random_stage((20, 20), (0, 0), [], steps=400),
    "fc_n10_static": random_stage((10, 10), (3, 5), STATIC_MORE, steps=400),
    "fc_headon": scenario_stage("mf_headon_open"),
    "fc_crossing": scenario_stage("mf_crossing"),
    "fc_parallel": scenario_stage("mf_parallel"),
}

# Training curriculum stages in mixture order (Phase 2 sampling 0.25 / 0.25 / 0.35 / 0.15).
TRAIN_STAGES = [
    {"task.random.static.n": [0, 0], "task.random.dynamic.n": [0, 0], "task.random.static.models": [], "task.random.dynamic.models": []},
    {"task.random.static.n": [2, 4], "task.random.dynamic.n": [0, 0], "task.random.static.models": STATIC_FEW, "task.random.dynamic.models": []},
    {"task.random.static.n": [2, 4], "task.random.dynamic.n": [3, 5], "task.random.static.models": STATIC_FEW, "task.random.dynamic.models": PED},
    {"task.random.static.n": [3, 5], "task.random.dynamic.n": [10, 15], "task.random.static.models": STATIC_MORE, "task.random.dynamic.models": PED},
]
TRAIN_WEIGHTS = [0.25, 0.25, 0.35, 0.15]

STAGES = {**NAV, **FORECAST}
