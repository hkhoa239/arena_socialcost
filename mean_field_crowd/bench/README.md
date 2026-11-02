# Mean-field crowd navigation benchmark

Compares `AGENT_5` (no forecast), `AGENT_MF_ONLINE` (online constant-velocity crowd density forecast) and `AGENT_MF_RESIDUAL` (online forecast plus the frozen 211-parameter interaction residual) under identical PPO, reward, action space, scenarios and seeds. A fourth arm, unrestricted `AGENT_5`, measures the forward-only constraint on its own.

Results live in `results/REPORT.md`. It is labelled PARTIAL until every planned run is done.

## Commands

All commands run inside the container, after `cd /opt/arena_ws && source /opt/arena_ws/source`.

| Step | Command |
|---|---|
| Generate the 12 training configs | `python3 data/mf_bench/prepare_configs.py cuda 5 6 7` |
| Phase 1 forecasting (collection and benchmark) | `bash data/mf_bench/run_bench.sh forecast` |
| Train, validate, test and ablate seeds | `bash data/mf_bench/run_bench.sh seeds 5 6 7` |
| Pilot that is running now (forecast, then seed 5) | `bash data/mf_bench/run_pilot.sh` |
| Report from whatever exists | `python3 data/mf_bench/report.py 5` (or `5 6 7`) |
| Unit tests | `cd src/Arena/arena_training/deps/rosnav_rl/rosnav_rl && python -m pytest tests/test_bench_infra.py -q` |

To launch detached from the host, so the run survives Cursor closing:

```bash
docker exec -d -u arena -w /opt/arena_ws arena-arena_ws-arena-1 bash /opt/arena_ws/data/mf_bench/run_pilot.sh
```

Every step writes its own output file and is skipped when that file exists. A stopped pipeline therefore resumes when relaunched, and it never overwrites checkpoints or results.

Monitoring:

- `STATUS.txt` has one line per step.
- `logs/` holds the pipeline log, training logs and GPU samples.
- `eval/<RUN>/` holds the per-episode traces.

## Arms

All arms come from `agent_5_nav_progress.yaml`, which pays the approach reward on the final goal and uses the PPO settings of the pilots.

| Run | Architecture | Linear velocity |
|---|---|---|
| `BENCH_A5_FWD_S<seed>` | AGENT_5 | [0, 0.5] m/s, forward only |
| `BENCH_ONLINE_FWD_S<seed>` | AGENT_MF_ONLINE | [0, 0.5] |
| `BENCH_RESIDUAL_FWD_S<seed>` | AGENT_MF_RESIDUAL | [0, 0.5] |
| `BENCH_A5_FREE_S<seed>` | AGENT_5 | [−0.5, 0.5], the unrestricted control |

All four arms share the following:

- **Action space:** the policy acts in [−1, 1]² (`action_space.normalized: true`), and the decoder maps each component affinely onto the physical range. ω ∈ [−1.5, 1.5] rad/s.
- **Budget:** a 100k-step linear LR schedule. Checkpoints are written at 20k, 40k, 60k and 80k plus `last_model`, all from the same run.
- **Seeds:** the PPO seed, env seed and mixture seed all equal the run seed.
- **Training scenarios:** each episode is drawn from empty, static, moderate and dense with weights 0.25, 0.25, 0.35 and 0.15 (`staged.mixture_weights`).

The YAMLs differ only in name, architecture, linear range and seed; `prepare_configs.py` checks this.

**Why forward-only is normalized.** SB3 samples a Gaussian, stores the unclipped action and its log-probability, and sends a clipped copy to the env. A native box of [0, 0.5] would send about half of all early samples to exactly v = 0. With the shared [−1, 1]² box, both variants clip by the same mechanism and at the same rate (about 0.33 per dimension at initialisation). The rate is logged per rollout as `action/clip_rate_*`, so the remaining mismatch is measured rather than hidden.

## Scenarios (`stages.py`)

All scenarios use a 1.0 m goal tolerance.

| Stage | Pedestrians | Static obstacles | Episodes (test) | Limit |
|---|---:|---:|---:|---:|
| `b_empty` | 0 | 0 | 25 | 600 steps |
| `b_static` | 0 | 2–4 | 25 | 600 |
| `b_moderate` | 3–5 | 2–4 | 25 | 600 |
| `b_dense` | 10–15 | 3–5 | 25 | 600 |
| `b_stress` | 20 | 3–5 | 25 | 600 (evaluation only) |
| `b_headon` | 1, walking straight at the robot in open space | 0 | 10 | 450 |
| `b_crossing` | 1, crossing the robot's path | 0 | 10 | 450 |

The three seed families are disjoint by construction:

- training: `<env_seed>-<i>`
- validation: `bench-val-<stage>`
- test: `bench-test-<stage>`
- forecasting: `bench-fc-<stage>`

The forecasting data is never used for training or navigation evaluation.

## Phase 0 audit (28 Oct)

| Check | Result |
|---|---|
| MF agents use standard PPO | Yes. `AGENT_MF_ONLINE` and `AGENT_MF_RESIDUAL` inherit `algorithm_class = PPO` from `AGENT_5`. Only the older `AGENT_MF` and `AGENT_MF_FORECAST` use `MeanFieldPPO`. |
| Observation spaces | AGENT_5 spaces, plus `PedestrianKinematicsSpace` and `ForecastCommandSpace` for both MF arms. The forecast is a sixth ResNet channel. AGENT_5's class body is unchanged. |
| Residual frozen and unchanged | Yes. It is `requires_grad_(False)`, kept in eval mode, and reloaded from file on the first forward pass, after SB3's orthogonal init. After PPO updates, all 12 residual tensors in the saved model are bit-identical to `interaction_residual.pt` (`check_residual.py`). |
| `MF_FORECAST_MODE` | Read on every forward pass, so it applies to all extractor instances. An ONLINE model never loads the residual weights; setting `MF_FORECAST_MODE=residual` on it would apply random weights. The ablation only switches RESIDUAL models. |
| Eval runtime | `eval_one.sh` launches with `robot.mobile:=rosnav_rl` and aborts if a nav2 `controller_server` is running. The earlier eval scripts omitted the argument and ran nav2, with the goal observation stuck at (0, 0). |
| Goal identity | The observed goal equals the task generator's realized goal from the first step after reset (`goal_mismatch_steps` = 0). |
| Success | Counted only when `done_reason` is SUCCESS **and** `RewardGoalReached` = +20. |
| Termination | COLLISION has two triggers: a collision-monitor contact, or a laser fallback at min laser ≤ 0.317 m. The traces separate them (`contact_events`, `collision_monitor`). STEP_LIMIT is the episode limit. |
| Seed reproducibility | Over 310 recorded episodes on 40 matched keys, the episode seed and goal are identical across agents. Start poses differ by ≤ 0.07 m of teleport jitter. |
| World | `map_empty` contains a table (`table_0` at scenario (7, 5), map ≈ (12.3, 10.3)) that is absent from the occupancy image. The 2D lidar sees only part of it. The benchmark keeps the world unchanged; episodes involving the table appear in the failure analysis. |
| Configs | The three 4M configs (`sb_agent_5`, `sb_agent_mf_online`, `sb_agent_mf_residual`) differ only in name and architecture. |

## Package changes (uncommitted, `arena_training` submodule)

| File | Change |
|---|---|
| `rosnav_rl/cfg/action_spaces.py` | `DifferentialDriveActionSpace.normalized` (defaults to false, so existing configs are unchanged) |
| `rosnav_rl/utils/stable_baselines3/tick_callbacks.py` | `EpisodeMixtureCurriculum` and `ActionStatsCallback` |
| `arena_training/.../cfg/arena_cfg/task.py` | `staged.mixture_weights` and `staged.mixture_seed` |
| `arena_training/.../cfg/sb3_cfg/callbacks.py` | `callbacks.action_stats` |
| `arena_training/.../trainer/sb3_trainer.py` | wiring for the two callbacks |
| `tests/test_bench_infra.py` | 7 unit tests |

The existing `rosnav_rl` suite (83 tests) and `arena_training` suite (56 tests) pass.

New scenario files are `mf_headon_open`, `mf_crossing` and `mf_parallel` in `arena_simulation_setup/worlds/map_empty/scenarios/`. They were also copied into `install/.../share`.
