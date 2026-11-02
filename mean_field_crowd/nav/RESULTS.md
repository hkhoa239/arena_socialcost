# Navigation reward pilot

This file is rewritten as the detached pipeline finishes each stage. Status tail:

```
26-10-27T17:45:31+00:00
EVAL_BEGIN AGENT_5_DIAG_EMPTY empty 2026-10-27T17:45:31+00:00
EVAL_EXIT AGENT_5_DIAG_EMPTY empty 0 2026-10-27T17:49:51+00:00
EVAL_BEGIN AGENT_5_NAV_PROGRESS empty 2026-10-27T17:49:52+00:00
EVAL_EXIT AGENT_5_NAV_PROGRESS empty 0 2026-10-27T17:51:56+00:00
TRAIN_BEGIN AGENT_5_NAV_STATIC device=cuda 2026-10-27T17:51:57+00:00
TRAIN_EXIT AGENT_5_NAV_STATIC 1 2026-10-27T18:29:21+00:00
TRAIN_BEGIN AGENT_5_NAV_STATIC_PROGRESS device=cuda 2026-10-27T18:29:21+00:00
TRAIN_EXIT AGENT_5_NAV_STATIC_PROGRESS 1 2026-10-27T19:06:50+00:00
EVAL_BEGIN AGENT_5_NAV_STATIC static 2026-10-27T19:06:50+00:00
EVAL_EXIT AGENT_5_NAV_STATIC static 0 2026-10-27T19:12:33+00:00
EVAL_BEGIN AGENT_5_NAV_STATIC_PROGRESS static 2026-10-27T19:12:33+00:00
EVAL_EXIT AGENT_5_NAV_STATIC_PROGRESS static 0 2026-10-27T19:16:41+00:00
TRAIN_BEGIN AGENT_5_NAV_PED1 device=cuda 2026-10-27T19:16:41+00:00
TRAIN_EXIT AGENT_5_NAV_PED1 1 2026-10-27T19:54:14+00:00
TRAIN_BEGIN AGENT_5_NAV_PED1_RISK device=cuda 2026-10-27T19:54:14+00:00
TRAIN_EXIT AGENT_5_NAV_PED1_RISK 1 2026-10-27T20:26:50+00:00
TRAIN_BEGIN AGENT_5_NAV_PED1_PROGRESS device=cuda 2026-10-27T20:26:50+00:00
TRAIN_EXIT AGENT_5_NAV_PED1_PROGRESS 1 2026-10-27T20:59:17+00:00
EVAL_BEGIN AGENT_5_NAV_PED1 ped1 2026-10-27T20:59:17+00:00
EVAL_EXIT AGENT_5_NAV_PED1 ped1 0 2026-10-27T21:03:30+00:00
EVAL_BEGIN AGENT_5_NAV_PED1_RISK ped1 2026-10-27T21:03:30+00:00
EVAL_EXIT AGENT_5_NAV_PED1_RISK ped1 0 2026-10-27T21:06:10+00:00
EVAL_BEGIN AGENT_5_NAV_PED1_PROGRESS ped1 2026-10-27T21:06:10+00:00
EVAL_EXIT AGENT_5_NAV_PED1_PROGRESS ped1 0 2026-10-27T21:08:10+00:00
EVAL_BEGIN AGENT_5_NAV_PED1 headon 2026-10-27T21:08:11+00:00
EVAL_EXIT AGENT_5_NAV_PED1 headon 0 2026-10-27T21:11:00+00:00
EVAL_BEGIN AGENT_5_NAV_PED1_RISK headon 2026-10-27T21:11:01+00:00
EVAL_EXIT AGENT_5_NAV_PED1_RISK headon 0 2026-10-27T21:12:56+00:00
EVAL_BEGIN AGENT_5_NAV_PED1_PROGRESS headon 2026-10-27T21:12:56+00:00
EVAL_EXIT AGENT_5_NAV_PED1_PROGRESS headon 0 2026-10-27T21:14:16+00:00
```

## Root cause

`approach_goal` was paid on the straight-line subgoal. That subgoal sits `clamp(speed * 1.5 s, 0.5 m, 2.5 m)` ahead of the robot. The Jackal command limit is 0.5 m/s, so the lookahead is at most 0.75 m. Stage success radii are 1.5, 1.2, 1.0, and 0.8 m. The episode therefore ends at a radius that is still outside the lookahead, and the distance to the subgoal does not fall while the robot closes on the goal. Speeding up lengthens the lookahead and is charged as a retreat.

The empty-map run of the original reward (20,000 steps, seed 5) ended 1 goal / 1 bumper / 32 timeouts. The one goal, episode 7 at step 426, had approach reward -0.729 and the +20 bonus. At that instant the robot was 1.50 m from the goal and the subgoal was 0.75 m ahead, the full-speed lookahead. Across the 33 logged returns the approach term averaged -1.61.

The policy observes the subgoal angle, which does point at the goal, but the reward does not pay for reducing goal distance. Laser safety is the other dense term. On the empty map most timeouts had no safety penalty, so walls are not why the robot misses the goal. Gamma 0.99 makes the sparse +20 small when the crossing takes a few hundred steps. A 60 s episode at 0.5 m/s can cover the 30 m by 23 m empty map (mean random start-goal distance about 14 m). The horizon is long enough if the robot drives toward the goal.

Current pedestrian proximity removes the head-on bumper and still does not reach the goal, because progress is still unpaid. The one-second proximity term keeps charging an oncoming person and the short run slowed to about 0.04 m/s.

## The one change

`approach_goal` follows the final goal (`_follow_subgoal: false`). Coefficients are unchanged:

```
goal_reached: +20 inside the stage tolerance
safe_distance: -0.1 when the nearest laser return is inside 1.0 m
factored_safe_distance: -0.2 * (1.0 + robot_radius - min_laser) on that same event
approach_goal: +0.3 per meter closer to the final goal, -0.4 per meter farther
collision: -15, episode ends
pedestrian_proximity, only on the third arm: -0.5 * (1.0 - distance) while the nearest person is inside 1.0 m
```

A detour that adds L meters of path costs about 0.1 L because the retreat factor is 0.4 and the approach factor is 0.3. A 3 m detour costs 0.3, against a collision charge of 15 and a goal bonus of 20. No future pedestrian state and no residual forecast enter the reward. PPO, the network, and the seeds are unchanged. The empty-map original run is the existing `AGENT_5_DIAG_EMPTY` 20,000-step result and is not retrained.

`run_seed` is now a task-generator launch parameter. Before this, the launch file dropped it, each process drew a uuid, and repeated evaluations were not the same crowd.

## Training counts

These are the training episodes, not a matched evaluation. Approach is the mean episode sum of `RewardApproachGoal`.

| Run | Episodes | Goals | Bumper | Timeout | Mean approach |
|---|---:|---:|---:|---:|---:|
| AGENT_5_DIAG_EMPTY | 34 | 1 | 1 | 32 | -1.610 |
| AGENT_5_NAV_PROGRESS | 36 | 3 | 4 | 29 | 0.583 |
| AGENT_5_NAV_STATIC | 34 | 0 | 1 | 33 | -1.541 |
| AGENT_5_NAV_STATIC_PROGRESS | 38 | 6 | 5 | 27 | 0.112 |
| AGENT_5_NAV_PED1 | 34 | 1 | 1 | 32 | -1.694 |
| AGENT_5_NAV_PED1_RISK | 34 | 1 | 0 | 33 | -1.514 |
| AGENT_5_NAV_PED1_PROGRESS | 42 | 11 | 8 | 23 | 1.931 |

The attached CPU attempt that was stopped at 12,288 steps is `progress_empty_cpu.log`. It is not a finished policy. At that point it had 3 goals, 2 bumpers, and 17 timeouts in 22 episodes, and the two finished goals had positive approach reward (+0.93 and +1.83).

## Matched evaluation

A goal here means the reward paid `RewardGoalReached` (+20) and the log reason is `DONE_REASONS.SUCCESS`. Every matched episode had `seed_matches_formula` true, and the five seeds on a stage are the same for every policy on that stage.

Two ped1 rows were stored as `success: true` because the task generator closed the episode (`finished after 8.2s` on the progress policy, `finished after 9.3s` on the proximity policy, both episode id 3, seed 8312821209066852145). The logged goal pose on both is `(0, 0, 0)`, the return has no +20, and the robot was still a full-speed lookahead from the subgoal. Those are not navigation goals. The table below counts them as non-goals. With that correction every paired goal-rate difference is 0.

| Policy | Stage | Episodes | Goals (+20) | Bumper | Timeout | Other end | Mean speed |
|---|---|---:|---:|---:|---:|---:|---:|
| AGENT_5_DIAG_EMPTY | empty | 5 | 0 | 2 | 3 | 0 | 0.342 |
| AGENT_5_NAV_PROGRESS | empty | 5 | 0 | 5 | 0 | 0 | 0.487 |
| AGENT_5_NAV_STATIC | static | 5 | 0 | 0 | 5 | 0 | 0.358 |
| AGENT_5_NAV_STATIC_PROGRESS | static | 5 | 0 | 3 | 2 | 0 | 0.381 |
| AGENT_5_NAV_PED1 | ped1 | 5 | 0 | 3 | 2 | 0 | 0.425 |
| AGENT_5_NAV_PED1_RISK | ped1 | 5 | 0 | 4 | 0 | 1 | 0.356 |
| AGENT_5_NAV_PED1_PROGRESS | ped1 | 5 | 0 | 4 | 0 | 1 | 0.492 |
| AGENT_5_NAV_PED1 | headon | 5 | 0 | 5 | 0 | 0 | 0.361 |
| AGENT_5_NAV_PED1_RISK | headon | 5 | 0 | 5 | 0 | 0 | 0.296 |
| AGENT_5_NAV_PED1_PROGRESS | headon | 5 | 0 | 5 | 0 | 0 | 0.500 |

Pedestrian contact events were 0 on every episode, including the laser-bumper endings.


## Mean-field comparison

Not started. The forecasting model, PedVO, and the AGENT_5 network were not changed. A mean-field run is justified only after the goal-tracking reward shows reliable collision-free goals on the matched empty, static, and single-pedestrian evaluations. Five episodes make the bootstrap interval wide; treat a small difference as unresolved.
