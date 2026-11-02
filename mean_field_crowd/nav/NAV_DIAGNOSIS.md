# AGENT_5 goal-reaching diagnosis

Status as of 2026-10-28 09:00 UTC. The detached pipeline `data/mf_nav/run_nav100k.sh` is still running. The 100k static-checkpoint evals at 60k, 80k and 100k are pending, as are the zero-shot evals of the 100k static model on the empty map, one pedestrian, and head-on. Progress is in `data/mf_nav/STATUS_100K.txt`. When it ends with `PIPELINE_DONE`, run `python3 data/mf_nav/analyze_trace.py` for the final tables.

## 1. Summary

1. **The 27 Oct matched evaluation (0/50 goals) was invalid.** The eval launcher started the simulator without `robot.mobile:=rosnav_rl`, so the robot ran the nav2 bringup instead of the RL one. The policy never received its goal and was steering toward the map origin for the whole episode. Training was not affected.
2. **With the launch fixed, the 20k checkpoints trained with the final-goal progress reward reach goals.** Results: empty 9/10, static 7/10, one pedestrian 8/10. The original subgoal reward gets 5/10, 0/10 and 0/10.
3. **Head-on fails for every policy, and the cause is not the pedestrian.** The progress policies drive in reverse. On head-on they back into an obstacle about 2 m behind the start within ~47 steps. There were 0 pedestrian contacts.
4. **Training is not reliable.** The 100k empty run uses the same reward, PPO settings and seed as the 9/10 run, only a longer budget. It never learned to reach goals: its final policy weaves with heavy turning, often drives away from the goal, and scores 0/10. The 100k static run did improve during training (11 goals in the last 24 episodes). Its matched evals at 60k–100k are pending.
5. **Current recommendation: PPO navigation is not ready for the Mean-Field comparison.** The bottleneck is now training stability and variance across runs, plus the reverse-driving gait. The reward and the environment signals check out.

## 2. Root causes and fixes

### 2.1 Eval runtime ran nav2 instead of the rosnav_rl bringup (verified, fixed in the eval path)

`arena train` includes `training.launch.py`, which defaults `robot.mobile:=rosnav_rl` and forwards it to `arena_runtime.launch.py`. The standalone eval scripts launched `arena_runtime.launch.py` directly without that argument, and the runtime's default mobile adapter is nav2. This affects `data/mf_nav/run_all.sh`, `data/mf_eval/run_matched.sh` and `data/mf_diag/*.sh`.

What happened as a result:

| Symptom | Mechanism |
|---|---|
| The goal observation was (0, 0) on every step | The rosnav_rl GotoPose handler, which publishes `<robot>/goal_pose`, did not exist. The env's `goal_pose` collector keeps its default `PoseStamped()` until a message arrives. |
| The policy and `approach_goal` targeted the map origin | Both read the same observed goal, so they agreed with each other but not with the task generator's goal. |
| `nav2 controller_server` and `bt_navigator` were running | Confirmed with `ps` inside the container during the eval. |
| Two ped1 episodes ended as "finished after 8.x s", `success: true`, no +20 | The task generator closed the episode on its own goal check (the GotoPose action succeeded). The reward never saw the robot inside the radius of the goal it was observing, (0, 0). |

Verification is on the same seed and checkpoint (`AGENT_5_NAV_PED1_PROGRESS`, ped1, run_seed `nav-ped1-5`):

| Launch | Goal obs vs record goal | Result (5 episodes) |
|---|---|---|
| without `robot.mobile:=rosnav_rl` | wrong on 100 % of steps, always (0, 0) | 0 goals, 4 collisions, 1 task-generator-only "success" |
| with `robot.mobile:=rosnav_rl` | equal from the first step | 3 goals with +20, 2 collisions |

The fix is in `data/mf_nav/trace_one.sh`, which launches with `robot.mobile:=rosnav_rl` and aborts if any `nav2_controller/controller_server` process is running. No package source was changed. The old launch scripts were left untouched as records of what ran. `CLAUDE.md` ("Navigation result, 27 Oct 2026") and `RESULTS.md` still state the invalid 0/50 conclusion and should be corrected.

### 2.2 COLLISION endings: contact or laser bumper (corrected 28 Oct)

`RewardCollision` ends an episode on either of two signals:

- `collision_monitor_state` from the task generator's `collision_tracker`, which fires when the Jackal polygon intersects the occupancy grid or a pedestrian disc of radius 0.3 m;
- its laser fallback, which fires when min laser ≤ robot radius + 0.05 m = 0.317 m.

An earlier version of this section said these endings were real contacts. That was not established: the traces here did not record which signal fired. The benchmark recorder (`trace_eval.py`, from 28 Oct) now stores every contact event of the episode and the per-step monitor value. In its first smoke runs, every COLLISION ending was the laser fallback (final min laser 0.30–0.31 m, no contact event, monitor never true). The head-on endings in section 5 also end at 0.29–0.32 m and were most likely laser-bumper stops. Pedestrian contact events were 0 on every matched episode.

### 2.3 Goal and termination consistency (verified, no further bug)

With the corrected launch, every matched episode passes all of these checks:

- **Goal identity.** The observed `goal_pose` equals the realized goto pose in the task generator's `EpisodeRecord` (`map_poses`) on every step, including the first step after reset (`goal_mismatch_steps = 0`).
- **Frames.** The robot pose comes from TF in the map frame, and the goal is in the map frame, so the robot-frame goal is computed consistently.
- **Tolerance.** The reward's `goal_radius` is re-synced on reset from the episode record's `tolerance_radius` (`ArenaBaseEnv._after_task_reset`) and matched it on every episode. At success the true distance is just below the tolerance (≈1.48 m against 1.5 m, ≈1.18 m against 1.2 m, ≈0.78 m against 0.8 m).
- **Seeds.** `seed_matches_formula` is true on every episode, and the seeds are identical across policies on a stage.
- **Training.** All SUCCESS endings in the training logs carry the +20, and no training episode was closed by the task generator alone.

### 2.4 Two arrival judges (latent, not observed with the correct launch)

Arrival is judged in two places:

- `RewardGoalReached`: observed distance below `goal_radius`, checked every control step.
- The GotoPose handler: TF distance within `pose_tolerance`, polled every 0.2 s of wall time. When it fires, the task generator ends the episode as SUCCESS and `base_env` sets `is_success` without the +20.

They use the same tolerance and the same map-frame pose, so they agree in practice. With the correct launch, task-generator-only successes were 0 out of 175 matched episodes. I left this unchanged. If it ever shows up, count it as a non-goal, as `analyze_trace.py` already does.

## 3. Reward: formulas

Notation:

| Symbol | Meaning |
|---|---|
| $p_t$ | robot position (map frame, TF) |
| $\theta_t$ | robot heading |
| $g$ | final goal |
| $d_t = \lVert g - p_t \rVert$ | distance to the final goal |
| $\ell_t$ | minimum laser range (360° lidar, 12 m max) |
| $r_{rob} = 0.267$ m | Jackal radius |
| $R$ | stage tolerance: 1.5, 1.2, 1.0 or 0.8 m |

The robot-frame goal, which is both observed and used by the reward, is

$$
\begin{bmatrix} x_g^{rob} \\ y_g^{rob} \end{bmatrix} = R(-\theta_t)\,(g - p_t).
$$

The reward has five terms. Only the approach term changed between the original and progress configs.

**Goal reached** (terminal):

$$
r^{goal}_t = 20 \cdot \mathbb{1}[d_t < R].
$$

**Approach**, with $\Delta_t = d_{t-1} - d_t$:

$$
r^{app}_t = \begin{cases} 0.3\,\Delta_t & \Delta_t > 0 \\ 0.4\,\Delta_t & \Delta_t \le 0 \end{cases}
$$

The term is skipped on a step where the target jumps more than 0.25 m in world coordinates.

- Original config: $d_t$ is the distance to the straight-line subgoal.
- Progress config: $d_t$ is the distance to the final goal $g$ (`approach_goal._follow_subgoal: false`).

**Safe distance:**

$$
r^{safe}_t = -0.1 \cdot \mathbb{1}[\ell_t < 1.0], \qquad
r^{fsafe}_t = -0.2\,(1.0 + r_{rob} - \ell_t)\cdot \mathbb{1}[\ell_t < 1.0].
$$

**Collision** (terminal): $-15$.

**Pedestrian proximity** (RISK arm only): $-0.5\,(1.0 - \delta_t)$ while the nearest person is at distance $\delta_t < 1.0$ m.

### Why the subgoal version does not pay for progress

The subgoal lies on the robot-to-goal line at a speed-dependent lookahead:

$$
s_t = p_t + L(v_{t-1})\,\frac{g - p_t}{\lVert g - p_t\rVert}, \qquad
L(v) = \mathrm{clip}(1.5\,v,\ 0.5,\ 2.5)\ \text{m}.
$$

So $\lVert s_t - p_t \rVert = L(v_{t-1})$ wherever $d_t > L$. At the Jackal's 0.5 m/s cap, $L \le 0.75$ m, which is less than every tolerance $R$. The subgoal therefore never reaches the goal before the episode ends, and

$$
\Delta_t^{sub} = L(v_{t-2}) - L(v_{t-1}).
$$

This depends only on speed changes, not on the distance covered. Speeding up is charged at 0.4 × 1.5 × Δv and slowing down is paid. Over an episode the sum is roughly constant whatever the robot does. The final-goal version pays $0.3$ per metre actually gained.

### Cost of a detour

Going $L$ m off course and back costs $-0.4L + 0.3L = -0.1L$. For example, a 3 m detour costs 0.3, against $-15$ for a collision and $+20$ for the goal.

### Discount

The goal bonus seen from the start of a $T$-step crossing is $20\,\gamma^{T}$:

| Crossing | $T$ | Discounted bonus |
|---|---|---|
| mean 13.7 m empty-map start–goal distance at 0.5 m/s | ≈ 275 | ≈ 1.3 |
| a 600-step timeout | 600 | 0.05 |

This is why the dense term matters.

### Eval metrics

| Metric | Definition |
|---|---|
| progress fraction | $(d_0 - d_{end})/d_0$ |
| reverse fraction | share of steps with commanded $v < -0.05$ m/s |
| inside 1 m | share of steps with $\ell_t < 1$ m |
| goal | `done_reason == SUCCESS` **and** the episode return contains `RewardGoalReached = +20` |

## 4. Implementation (new files, all in `data/mf_nav/`)

| File | Purpose |
|---|---|
| `trace_eval.py` | Matched rollout with the same stages, seeds and launch args as `eval_one.py`. The env runs in-process (`debug_mode`) so each step's `obs_dict` and reward breakdown are recorded. Per step it stores the observed and record goal, true and observed goal distance, commanded and measured velocity, min laser, and reward terms. Reset observations are kept apart from terminal ones because the vec env auto-resets. |
| `trace_one.sh` | Starts `arena_runtime` **with `robot.mobile:=rosnav_rl`**, refuses to run if nav2 is up, runs one trace, and stops the runtime. |
| `analyze_trace.py` | Matched table, failure-mode metrics, reward decomposition and consistency flags. Output goes to `trace/summary.json`. |
| `train_curve.py` | Training curve from an `arena train` log, binned by timesteps. |
| `prepare_100k.py` | 100k configs generated from `agent_5_nav_progress.yaml`. The only changes are the name, budget, device, stage, `checkpoint_ticks` (20k/40k/60k/80k), a pinned stage (no promotion or demotion), and the periodic eval moved past the budget. PPO, network, reward coefficients and seeds are unchanged. |
| `run_nav100k.sh` | Detached pipeline: corrected re-eval, then two 100k trainings, then checkpoint evals. |

No package source was edited. `AGENT_5`, the reward units, the PedVO parameters and the mean-field code are untouched.

## 5. Results

### 5.1 Corrected matched evaluation of the existing 20k checkpoints

All rows use 10 episodes per stage with the same seeds for every policy on a stage. Run seeds are `nav-<stage>-5`, and episodes 1–5 are the same seeds as the invalid 27 Oct eval. Every row passes the seed, goal-identity and tolerance checks.

| Policy (reward) | Stage | Goal | Collision | Timeout | Progress frac | Final dist (m) | mean abs v (m/s) | Reverse frac |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| AGENT_5_DIAG_EMPTY (subgoal) | empty | 5 | 1 | 4 | 0.64 | 4.84 | 0.35 | 0.77 |
| AGENT_5_NAV_PROGRESS (final goal) | empty | **9** | 0 | 1 | 0.86 | 1.58 | 0.50 | 0.00 |
| AGENT_5_NAV_STATIC (subgoal) | static | 0 | 0 | 10 | 0.00 | 10.08 | 0.35 | 0.63 |
| AGENT_5_NAV_STATIC_PROGRESS (final goal) | static | **7** | 1 | 2 | 0.68 | 3.22 | 0.34 | 0.93 |
| AGENT_5_NAV_PED1 (subgoal) | ped1 | 0 | 2 | 8 | −0.02 | 14.31 | 0.42 | 0.36 |
| AGENT_5_NAV_PED1_RISK (subgoal + proximity) | ped1 | 2 | 2 | 6 | 0.29 | 11.46 | 0.31 | 0.89 |
| AGENT_5_NAV_PED1_PROGRESS (final goal) | ped1 | **8** | 2 | 0 | 0.83 | 2.98 | 0.49 | 1.00 |
| AGENT_5_NAV_PED1 (subgoal) | headon | 0 | 0 | 10 | 0.06 | 5.65 | 0.43 | 0.49 |
| AGENT_5_NAV_PED1_RISK | headon | 0 | 0 | 10 | 0.58 | 2.53 | 0.27 | 0.74 |
| AGENT_5_NAV_PED1_PROGRESS | headon | 0 | 10 | 0 | −0.18 | 7.08 | 0.50 | 1.00 |

Pedestrian contacts were 0 on all 100 episodes.

Mean reward decomposition per episode:

| Policy | Stage | GoalReached | Approach | Safe | FactoredSafe | Collision | PedProx |
|---|---|--:|--:|--:|--:|--:|--:|
| AGENT_5_DIAG_EMPTY | empty | 10.00 | 0.46 | −0.24 | −0.33 | −1.50 | 0 |
| AGENT_5_NAV_PROGRESS | empty | 18.00 | 3.61 | 0.00 | 0.00 | 0.00 | 0 |
| AGENT_5_NAV_STATIC | static | 0.00 | −1.16 | 0.00 | 0.00 | 0.00 | 0 |
| AGENT_5_NAV_STATIC_PROGRESS | static | 14.00 | 2.13 | −2.38 | −2.89 | −1.50 | 0 |
| AGENT_5_NAV_PED1 | ped1 | 0.00 | 0.06 | −2.31 | −2.68 | −3.00 | 0 |
| AGENT_5_NAV_PED1_RISK | ped1 | 4.00 | −0.19 | −1.07 | −1.08 | −3.00 | −0.14 |
| AGENT_5_NAV_PED1_PROGRESS | ped1 | 16.00 | 3.49 | −0.16 | −0.20 | −3.00 | 0 |
| AGENT_5_NAV_PED1_PROGRESS | headon | 0.00 | −0.43 | −1.60 | −2.08 | −15.00 | 0 |
| AGENT_5_NAV_PED1_RISK | headon | 0.00 | −1.06 | −12.83 | −15.20 | 0.00 | −7.87 |

### 5.2 Failure modes

| Failure | Evidence | Class |
|---|---|---|
| The subgoal reward does not move toward the goal | Static and ped1 progress fraction ≈ 0. The robot drives at 0.35–0.45 m/s without closing distance. Approach sum ≤ 0. | poor goal tracking (reward) |
| Head-on, progress policy | Starts 6.0 m from the goal, facing it. It reverses at −0.5 m/s while turning (ω ≈ +0.5 rad/s) and moves to 7.1 m from the goal. Min laser goes from 1.98 m to 0.30 m and it hits an obstacle at step ~47 on all 10 episodes. Pedestrian contacts 0. | reverse gait in a constrained start |
| Head-on, proximity policy | Never collides but times out at 1.3–4.2 m. FactoredSafe −15.2 and PedProx −7.9 per episode: the robot hovers inside 1 m of obstacles and the person. | insufficient progress near people |
| Static progress, the 1 collision | Pinned against an obstacle (laser 0.31 m) with no progress for 276 steps. | insufficient avoidance |
| Static progress, 2 timeouts | One stalled for 428 steps 2.4 m from the goal. One was still closing (4.4 m left). | slow or stuck |
| Environment signals | Goal, tolerance, frames and seeds consistent. 0 task-generator-only successes. | none found (after the launch fix) |

**Reverse gait.** The progress-reward policies drive backward 93–100 % of the time, and the pedestrian ones always at −0.5 m/s. With a 360° lidar this works in open space. Nothing in the reward prefers forward motion, so the gait is a learned local optimum. A head-on start then needs a reversing turn into the wall behind. This is evidence for a forward-preference term (an existing unit, `reverse_drive`), but per the constraints none was added.

### 5.3 Controlled 100k training (final-goal progress reward, from scratch)

Setup: AGENT_5, PPO settings unchanged, seed 5 / env_seed 5, 2 envs, CUDA. Stage pinned: empty = stage 0 (R = 1.5 m), static = stage 1 (2–4 obstacles, R = 1.2 m). Both runs reached 100,352 steps. Each took about 2 h 38 min. There was no OOM.

**Training curves**, per 10k steps. These are training episodes, not a matched eval. Approach, Safe and Collision are mean episode sums.

AGENT_5_NAV100K_EMPTY:

| Steps | Ep | Goal | Coll | Timeout | Mean len | Approach | Safe | Collision |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0–10k | 18 | 1 | 3 | 14 | 532 | −0.58 | −0.57 | −2.50 |
| 10–20k | 17 | 1 | 1 | 15 | 586 | −0.52 | −0.50 | −0.88 |
| 20–30k | 18 | 1 | 0 | 17 | 580 | 0.47 | −1.95 | 0.00 |
| 30–40k | 17 | 3 | 0 | 14 | 588 | 1.46 | 0.00 | 0.00 |
| 40–50k | 19 | 3 | 3 | 13 | 525 | 1.07 | −1.28 | −2.37 |
| 50–60k | 17 | 2 | 1 | 14 | 568 | 1.80 | −1.06 | −0.88 |
| 60–70k | 18 | 2 | 1 | 15 | 550 | 0.90 | −0.42 | −0.83 |
| 70–80k | 17 | 0 | 1 | 16 | 596 | 0.20 | −0.35 | −0.88 |
| 80–90k | 17 | 0 | 0 | 17 | 600 | 0.83 | −2.79 | 0.00 |
| 90–100k | 18 | 2 | 2 | 14 | 537 | −0.06 | −1.37 | −1.67 |

AGENT_5_NAV100K_STATIC:

| Steps | Ep | Goal | Coll | Timeout | Mean len | Approach | Safe | Collision |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0–10k | 18 | 0 | 2 | 16 | 555 | −0.56 | −0.42 | −1.67 |
| 10–20k | 18 | 1 | 3 | 14 | 551 | 0.69 | −0.82 | −2.50 |
| 20–30k | 17 | 2 | 1 | 14 | 567 | 0.62 | −1.02 | −0.88 |
| 30–40k | 17 | 0 | 1 | 16 | 580 | 0.53 | −0.84 | −0.88 |
| 40–50k | 20 | 4 | 4 | 12 | 526 | 1.59 | −1.22 | −3.00 |
| 50–60k | 19 | 6 | 4 | 9 | 502 | 2.30 | −2.28 | −3.16 |
| 60–70k | 22 | 3 | 8 | 11 | 475 | 1.68 | −2.26 | −5.45 |
| 70–80k | 21 | 7 | 3 | 11 | 462 | 2.40 | −1.07 | −2.14 |
| 80–90k | 22 | 6 | 5 | 11 | 459 | 2.12 | −1.99 | −3.41 |
| 90–100k | 24 | 11 | 7 | 6 | 417 | 3.20 | −0.96 | −4.38 |

**Matched checkpoint evaluations** (10 episodes, same seeds as section 5.1):

| Run | Checkpoint | Stage | Goal | Collision | Timeout | Progress frac | mean abs v | mean abs ω (rad/s) |
|---|---|---|--:|--:|--:|--:|--:|--:|
| NAV100K_EMPTY | 20k | empty | 1 | 3 | 6 | 0.10 | 0.30 | 0.14 |
| NAV100K_EMPTY | 40k | empty | 1 | 7 | 2 | 0.12 | 0.44 | 0.45 |
| NAV100K_EMPTY | 60k | empty | 1 | 3 | 6 | 0.26 | 0.20 | 0.22 |
| NAV100K_EMPTY | 80k | empty | 0 | 1 | 9 | −0.02 | 0.19 | 0.33 |
| NAV100K_EMPTY | 100k | empty | 0 | 1 | 9 | −0.05 | 0.43 | 0.91 |
| NAV100K_STATIC | 20k | static | 4 | 3 | 3 | 0.52 | 0.37 | 0.12 |
| NAV100K_STATIC | 40k | static | 3 | 3 | 4 | 0.47 | 0.48 | 0.67 |
| NAV100K_STATIC | 60k / 80k / 100k | static | *pending* | | | | | |

**Reading the 100k results:**

- **The empty run failed to learn goal-reaching.** Training goals never exceeded 3 per 10k steps, and the 100k policy weaves (mean |ω| 0.9 rad/s, low net speed) and in 3 of 10 episodes ends farther from the goal than it started (see `viz/now/overview/AGENT_5_NAV100K_EMPTY_t100000_empty.png`), giving 0/10. The 20k run with the same reward and seed reached 9/10. The only difference is the budget, which stretches the linear LR decay (1e-3 → 1e-4) over 100k instead of 20k. A single-seed outcome is therefore not reproducible, and the 9/10 should not be treated as a reliable property of the reward.
- **The static run improves steadily in training.** Goals per 10k go from 0 to 11, episode length from 555 to 417, and the approach sum from −0.56 to +3.20. Collisions rise as well, to 7 of 24 in the last bin. The matched evals at 20k/40k (4/10, 3/10) are below the 20k static pilot (7/10). The 60k–100k evals will show whether training gains transfer.

## 6. Diagnosis and next step (provisional until the pipeline finishes)

- The **environment and termination signals are correct**. The only defect found was the eval launch (section 2.1).
- The **final-goal progress reward is the right single change**. The subgoal reward provably pays nothing for progress (section 3), and its policies make no net progress on static and ped1.
- The **current bottleneck is PPO training reliability, not the reward**:
  - outcomes differ widely between runs that differ only in budget (9/10 against 0/10 on empty);
  - a deterministic policy can degrade into weaving away from the goal;
  - all of this comes from one seed.

  The fixed hyperparameters plausibly contribute: `ent_coef 0`, LR 1e-3, and `target_kl 1.0`, which almost never stops an update. This is a hypothesis. They were not changed, per the constraints.
- A second, specific bottleneck is the **reverse-only gait**. It is harmless in open space and fatal at the head-on start.

Recommended order, one change at a time:

1. Wait for the 100k static evals at 60k–100k and the zero-shot ped1/head-on results.
2. Measure variance before changing anything: repeat the 100k static run on 2–3 more seeds with everything else identical. This is the minimum evidence needed to call goal-reaching "reliable" (target: ≥ 8/10 matched goals on empty and static across seeds).
3. Only if seeds stay unreliable, consider one controlled change, either to PPO exploration (`ent_coef`) or to the reward (forward preference via the existing `reverse_drive` unit, justified by the head-on traces). Do not do both at once.
4. Run single-PedVO training only after step 2 passes.

**Mean-Field readiness: not ready.** A Mean-Field comparison on top of a baseline that varies from 0/10 to 9/10 between runs would not be interpretable.

## 7. Files

| Path | Content |
|---|---|
| `data/mf_nav/trace/*.json` | per-step traces of every matched episode (corrected launch) |
| `data/mf_nav/trace/*_NAV2BUG.json` | the invalid launch, for comparison |
| `data/mf_nav/trace/*_fixcheck5.json` | the same checkpoint and seed with the fixed launch |
| `data/mf_nav/trace/summary.json` | output of `analyze_trace.py` |
| `data/mf_nav/STATUS_100K.txt`, `pipeline_100k.log` | pipeline progress |
| `data/mf_nav/train_AGENT_5_NAV100K_{EMPTY,STATIC}.log` | 100k training logs |
| `data/agents/AGENT_5_NAV100K_{EMPTY,STATIC}/` | `tick_{20,40,60,80}000.zip` and `last_model.zip` |
