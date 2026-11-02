# dataset_v1 validation: stage `subset`

Created 2026-11-01T10:14:34.

Episodes checked: 16, passing every check: 16, not recorded / not done: 0.
Counterfactual pairs checked: 6, passing: 6. Leakage checks pass: True.

## Pairs

| pair | family | subtype | world | matched / stationary person-slots | pos max (m) | heading max (deg) | robot pos max (m) | observability | ok |
|---|---|---|---|---|---:|---:|---:|---|---|
| P0051 | conversation | static | airport | 250 / 250 (125 slots) | 0.0 | 0.001 | 1.623 | visually_observable | True |
| P0058 | conversation | approach_and_stop | socialcost_office | 1 / 245 (125 slots) | 0.0416 | 107.817 | 0.683 | visually_observable | True |
| P0098 | stop_gesture | frontal/pulse | hospital_1 | 125 / 125 (125 slots) | 0.0 | 0.0 | 0.130 | visually_observable | True |
| P0154 | wheelchair | slow_forward_opposite | reception | 0 / 15 (125 slots) | 0.1467 | 53.318 | 0.057 | visually_observable | True |
| P0173 | queue | normal/perpendicular/advancing | hospital_1 | 42 / 743 (125 slots) | 2.8335 | 178.741 | 2.960 | hidden_state | True |
| P0182 | queue | dense/perpendicular/advancing | socialcost_office | 4 / 418 (125 slots) | 2.9881 | 37.746 | 0.128 | hidden_state | True |

## Episodes

| episode | family | cond | world | split | ok | failing |
|---|---|---|---|---|---|---|
| 000101 | conversation | on | C | train | True |  |
| 000102 | conversation | off | C | train | True |  |
| 000115 | conversation | on | A | train | True |  |
| 000116 | conversation | off | A | train | True |  |
| 000195 | stop_gesture | on | B | test_a | True |  |
| 000196 | stop_gesture | off | B | test_a | True |  |
| 000307 | wheelchair | on | D | test_b | True |  |
| 000308 | wheelchair | off | D | test_b | True |  |
| 000345 | queue | on | B | test_c | True |  |
| 000346 | queue | off | B | test_c | True |  |
| 000363 | queue | on | A | train | True |  |
| 000364 | queue | off | A | train | True |  |
| 000425 | group_walking | na | C | test_c | True |  |
| 000526 | partial_observability | na | D | test_b | True |  |
| 000547 | partial_observability | na | C | test_a | True |  |
| 000561 | controls | na | D | test_b | True |  |
