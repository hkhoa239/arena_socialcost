# SOCIALCOST dataset_v1: specification

Status: Stage A (specification and manifest). The manifest freezes after the Stage A subset passes validation. Once it is frozen, any regeneration requires `--force` and invalidates every recording.

Code: `src/deps/socialcost/socialcost/dv1/` (spec, geometry, scenes, scenario, manifest, post) and `src/deps/socialcost/scripts/dv1_*.py`. Data: `data/socialcost/dataset_v1/`. Status log: `data/socialcost/dataset_v1_STATUS.txt`.

## Recording

| Item | Value |
|---|---|
| Simulator | Arena + Isaac Sim 6.1 (PhysX), TurtleBot4 with OAK-D, ROS domain 7, container `socialcost-arena` |
| Robot controller | Nav2 Regulated Pure Pursuit, bodies-only (no social layer); the robot drives start to goal in every episode |
| Episode | 25 s recording window at 5 Hz = 125 samples, opening 8 s after the seeded episode start |
| Image | 320 x 240 RGB + planar depth (m), camera intrinsics and extrinsics per episode |
| Collection | `scripts/dv1_collect.py`. Persistent mode: one Isaac launch serves a group of episodes in one world (switch `task.scenario.file`, then `reset_episode` with the episode's seed; relaunch on a world change, after an error, or every 25 episodes). Isaac renders every physics step (default). Rendering every 2nd step (opt-in flag file `/opt/arena_ws/log/isaac_render_every`) raised the RTF from about 0.36 to 0.5 but broke RGB/depth pairing (132 of 183 frames without depth, 20 skipped slots), so it is not used. |
| Validity | at least 80 % of the expected samples, no TF failure, the robot upright (tilt < 5 deg) and moving (path >= 1 m), `combined_native` identical to `social_cost` |

## Worlds

| Key | World | Role | Geometry |
|---|---|---|---|
| A | `socialcost_office` (new, lightweight) | seen | 44 m corridor whose width steps 3 / 4 / 3 / 2.2 m; 2.5 m north and 3 m south side passages (junctions); three side rooms behind open doorways; light furniture |
| B | `hospital_1` | seen | 4 m central hallway, 3 m sub-hallway, waiting area |
| C | `airport` | seen | 45 x 30 m open hall, check-in desks with belt-barrier lanes, seat rows, planters |
| D | `reception` | held-out (Test B only) | 30 x 23 m reception hall with counter, waiting chairs, 3 m door to a second room |

`office_1` was the first choice for World A and was replaced. Under Isaac its roughly 700 objects took about 405 s to load per launch, and after the warm-up reset the TurtleBot came to rest tilted 17-20 deg and could not move. This was reproduced at the probe start and at two dataset sites. The replacement shares no geometry with B, C or D.

Each world has 6-14 robot routes (sites): corridor, junction (an L-shaped route through a turn), open, seat aisle and door. Every route was checked on the map with furniture footprints: clearance >= 0.4 m and an A* path exists. Map frame = world frame + (5 - map origin) per axis. This was measured in all five probe worlds: (5.25, 5.25), (5.25, 5.30), (5.25, 8.30) for office_1 and (5.25, 4.75) for socialcost_office.

## Families and quotas

| Family | Episodes | Paired | Pedestrians | Variation |
|---|---:|---|---|---|
| Conversation | 120 (60 pairs) | ON / OFF | 2-4 | static 30, approach-and-stop 15, lifecycle 15 pairs; dyad (0.9-2.0 m, face-to-face or L), triangle, circle; 30 % of dyads straddle the route |
| Stop gesture | 100 (50 pairs) | ON / OFF | 1-4 | frontal / 30 deg / 60 deg body orientation; gesture from start, on approach, active-then-off, pulse; 20 % with an occluding bystander |
| Wheelchair | 100 (50 pairs) | ON wheelchair / OFF walking twin of the same identity | 1-4 | stationary, slow forward (same / opposite), crossing, boundary wait, turning |
| Queue | 100 (50 pairs) | ON membership / OFF same poses | 3-8 | dense 0.6-0.9 / normal 0.9-1.3 / sparse 1.3-2.0 m; parallel, perpendicular, oblique; 20 advancing, 30 static |
| Group walking | 60 | - | 2-6 | same direction, opposite, perpendicular, passing, FOV enter / leave, corridor narrowing; requested formation side-by-side, single file, V, loose |
| Partial observability | 80 | - | 2-6 | 20 FOV exit, 20 FOV re-entry, 20 static occlusion (occluder object), 20 dynamic occlusion (walking group) |
| Controls | 40 | - | 0-4 | 8 each: empty, single static, single moving, two independent, neutral crowd |

Subtype mixes are fixed per unit (largest remainder) before geometry sampling. A plan that is geometrically infeasible after 300 attempts falls back to a random plan, and the fallback is recorded (`plan_fallback`). In the current manifest 5 group-walk "passing" plans fell back.

## Pedestrian behaviour and trajectories

The scene builders (`dv1/scenes.py`) place every person in the route's local frame and reject:
- clearance below 0.4 m to walls or furniture;
- centre distance below 0.55 m to another person;
- a person within 2 m of the robot start;
- unreachable waypoints.

Behaviour compiles to humansim behaviour trees. Hold, `go_to` waypoints, TALK_TO / GROUP_CONVERSATION with partner gaze, halt attention (raised hand), QUEUE_USE at a scenario desk, and GROUP_WALK leader and followers.

Bystander motion follows the LISN (`tvss_nav`, Social-Nav arena-simulation-setup) scenario patterns:

| Class | Pattern | Waypoints |
|---|---|---|
| static | stands, random heading | - |
| linear / parallel / crossing | single leg | one target |
| pacing | back and forth between 2-3 nearby points with dwells, looping (LISN `waypoint_mode` 2 with short legs) | 2-3 |
| traverse | single pass 8-15 m, routed on the map through corridors and doorways (LISN `waypoint_mode` 1) | A* path, Douglas-Peucker simplified, legs <= 4 m |
| patrol | loop through 2-3 anchor points around the scene with dwells (LISN `waypoint_mode` 2) | A* routed |

Intended waypoints are stored per person and executed trajectories are in the recordings. Reproducibility class per episode: fully reproducible (nobody moves, or static queues), approximately reproducible, or reactive (group walking and crossings).

## Counterfactual pairs

Both conditions share pair id, run seed, world, robot start and goal, objects, spawn poses and bodies. The exception is the wheelchair family, where ON uses the wheelchair variant of the OFF identity. OFF episodes of static conversations and of queues take their spawn poses from the ON episode's first recorded sample. This is needed because formations re-slot people. The derived poses are stored in `manifests/derived_spawns.json`, and the manifest itself is not edited.

Matched geometry is tested per person over the samples where that person is stationary in both conditions. Controlled people (primary, participants, queue members, targets) must agree within 2 cm and 2 deg. Bystander agreement is reported separately.

Observability class: conversation `visual_temporal`, stop gesture `visual` (confirmed per pair from rendered frames, raised-hand point depth-consistent in >= 3 samples), wheelchair `visual_appearance`, queue `hidden_state`.

## Splits

| Split | Episodes | Rule |
|---|---:|---|
| train | 360 | worlds A-C, known bodies |
| val | 60 | worlds A-C, known bodies |
| test_a | 60 | worlds A-C, known bodies, unseen instances |
| test_b | 60 | world D only, known bodies |
| test_c | 60 | worlds A-C, held-out bodies only |
| Test D | 63 pairs | index of the counterfactual pairs inside test_a / b / c, not counted again |

Every family is split 60 / 10 / 10 / 10 / 10 %. Both conditions of a pair share a split. Episode, pair, template and seed are unique per unit, so no key spans two splits.

## Bodies

The asset bucket holds 104 human bodies: 102 walking and 2 wheelchair users. The pool is 44 identities, 33 known and 11 held out (25 %). Held-out selection searched 20,000 candidate sets for the one whose gender / age / ethnicity / role / domain proportions best match the pool, rejecting any set with a tag value that appears only in the held-out set. Each wheelchair user's walking twin is in the same pool. Labels and behaviour never depend on these tags.

## Pedestrian counts

| Bin | Target | Planned |
|---|---:|---:|
| 0-1 | 10 % | 10.0 % |
| 2 | 25 % | 24.7 % |
| 3-4 | 30 % | 35.0 % |
| 5-6 | 25 % | 20.3 % |
| 7-10 | 10 % | 10.0 % |

Only queue, group walking and partial observability may exceed 4 people under the family ranges, so 5-6 falls short by design.

## Labels

| Channel | Version |
|---|---|
| `social_cost/` | arena-native/1.1, unchanged |
| `social_cost_membership/` | extended-membership/1.1, unchanged |
| `social_cost_extended/` | extended/2.0 (existing pipeline channel: membership + halt + mobility hypothesis) |
| `components/NNNNNN.npz` | personal_space, conversation (native interaction term), queue (membership-completed QUEUE_USE), gesture (halt-gesture/1.0), appearance_or_geometry_extension (mobility-aid/1.0-hypothesis, evaluated separately), combined_native (== social_cost, checked per sample), combined_extended (extended/2.1 = membership + halt, no mobility hypothesis), body_occupancy (asset bounding box, physical footprint) |

The composition operator is the per-cell maximum. GROUP_WALK keeps weight 0, so it has no group term.

## Visibility labels

Per pedestrian and sample (`dv1/visibility.jsonl`): `visible`, `outside_fov`, `partially_occluded`, `fully_occluded`, `visibility_fraction`, `visible_pixel_count` (estimate), `time_since_last_seen`, distance and bearing.

Method: 15 points on the camera-facing half of a body cylinder (asset height) are projected and tested against the rendered depth in a 3 x 3 window with 0.35 m tolerance. No instance segmentation is published by the Isaac camera, so the labels use geometry and depth consistency only. These are privileged labels and are not inputs.


## Validation rules added after the pilot

- Matched geometry is evaluated per controlled person over stationary samples, and only until the robot first passes within 1.0 m of that person in either condition. After that, a close pass can push people or make them step aside (P0024: 4-6 cm at 0.56 m), which is reactive behaviour that is recorded, not matched.
- Interactions that start after a wait or end mid-episode (lifecycle conversations) are dynamic pairs: the participants turn toward each other when talking starts, and the divergence is recorded.
- The gesture term is required only when the gesturing person is inside the 10 m robot-centred grid.
- FOV exit and re-entry are detected as seen, then outside the FOV, then seen again, with any unseen states in between.
- Empty episodes have no bodies and are exempt from the body-split rule.
