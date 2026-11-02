# Gate 1: episode reproducibility

Scenario `socialcost_repro` (map_empty): 8 m robot route, a TALK_TO pair beside it, a GROUP_WALK of three crossing head-on. `run_seed:=7` for every launch. Isaac Sim 6.1, ROS domain 7, Nav2 RPP with the social layer overlay.

Times are simulator seconds relative to `EpisodeRecord.start_time`. Episode *k* of every launch gets the same seed (Arena derives it from `run_seed` and the episode index).

## 1. Free-running launches, every episode probed (`scripts/repro_probe.py`, 5 launches x 3 episodes)

| Episode in launch | goal dispatched | first non-zero command | robot first moves | robot RMSE / max (m) | pedestrian RMSE / max (m) | interaction activation spread |
|---|---|---|---|---|---|---|
| 1 | 0.017–0.033 | 0.02, 0.18, 0.25, **5.20, 5.23** | 0.38, 0.47, **4.58, 5.45, 5.47** | 1.146 / 1.508 | 0.010 / 0.070 | 0.000 s |
| 2 | 0.017–0.033 | 0.000–0.017 | 0.200–0.217 | 0.006 / 0.015 | 0.0005 / 0.004 | 0.000 s |
| 3 | 0.017–0.033 | 0.000–0.017 | 0.200–0.217 | 0.042 / 0.101 | 0.053 / 0.267 | 0.000 s |

- **Pedestrians are deterministic** in episode time: activations at identical times (TALK_TO at 0.483 s, GROUP_WALK at 1.233 s in every launch), positions identical to millimetres whenever the robot is.
- **The multi-second robot offset occurs only in the first episode after a launch** (3 of 5 launches). The goal is always dispatched at once; the delay sits in Nav2 / ros2_control bring-up: either the controller emits its first command 5.2 s late (launches 2, 5) or it commands at 0.02 s but the velocity controller does not move the robot until 4.6 s (launch 3). At 0.29x real time a few wall seconds of node activation become seconds of simulator time.
- Episode 3's larger pedestrian spread (27 cm) is closed-loop: walkers react to a robot a few cm off, so differences grow; it is not simulator nondeterminism.

## 2. Lockstep (`lockstep:=true lockstep.paused:=false`)

Launch 1 ran normally (motion at 0.20–0.27 s). Launch 2 produced **no episode in 24 min** and was killed. Lockstep is therefore not used: one hang in two launches is not acceptable for batch collection.

## 3. Protocol adopted: warm-up episode

`socialcost/warmup.py`, enabled by `warmup_episode: true` in a dataset config:

1. launch, wait for robot TF;
2. wait until the robot has moved 5 cm in episode 1 (planner, controller and velocity controller are live);
3. `lifecycle/reset_episode` with `seed: -1` (Arena's derived seed for the next index);
4. record from episode 2 only (`min_episode_id`), window = episode start + 8 s, 5 Hz grid in simulator time.

## 4. Acceptance run: 5 launches recorded through the full pipeline (`datasets/repro_gate1`)

| Quantity | Value |
|---|---|
| episode seeds | identical (1416656050948784524) |
| robot position, same slot, RMSE / max | **0.012 m / 0.050 m** |
| robot heading max deviation | 2.9° |
| pedestrian position RMSE / max | **0.006 m / 0.041 m** |
| label MAE between launches (cost units, 0–252) | 0.41 |

Pair probe (8 episodes, 4 ON/OFF pairs, bodies-only Nav2): robot ON vs OFF per slot max 0.022–0.049 m, mean 0.007–0.015 m.

**Gate 1: PASS.** No unexplained multi-second offsets, deterministic initialisation, pedestrian trajectories repeat to centimetres, matched episodes share one simulator-time window.

## Corrections made on the way

- The recorder's start check first used odometry *displacement*; the odometry pose follows the reset teleport (0.45–1.55 s after the start), so it flagged teleports as late starts. It now uses the odometry *twist*. Gate-1 and probe episodes flagged by the old check were re-validated from their TF trajectories (note in each `episode_metadata.json`).
- Arena bug found: humansim publishes `agent_gestures` only on change and the bridge clears its gesture cache on every reset, so a held gesture (halt) disappeared from `arena_peds` after any reset. Fixed in `humansim/.../core/agent_manager.py` (`_force_agent_gestures()` on reset, beside `_force_agent_meta()`); verified: halt present in the episode after the warm-up reset.

Raw data: `outputs/reproducibility/{free,lockstep}_*.jsonl`, `report.json`, `gate1_recorded.json`.
