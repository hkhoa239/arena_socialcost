# dataset_v1 validation: stage `pilot`

Created 2026-11-01T13:47:09.

Episodes checked: 120, passing every check: 120, not recorded / not done: 0.
Counterfactual pairs checked: 42, passing: 42. Leakage checks pass: True.

## Pairs

| pair | family | subtype | world | matched / stationary person-slots | pos max (m) | heading max (deg) | robot pos max (m) | observability | ok |
|---|---|---|---|---|---:|---:|---:|---|---|
| P0010 | conversation | static | socialcost_office | 401 / 401 (125 slots) | 0.0 | 0.0 | 0.089 | visually_observable | True |
| P0011 | conversation | static | reception | 339 / 339 (125 slots) | 0.0 | 0.0 | 0.141 | visually_observable | True |
| P0023 | conversation | lifecycle | hospital_1 | 26 / 233 (125 slots) | 0.0 | 12.703 | 0.063 | visually_observable | True |
| P0024 | conversation | static | socialcost_office | 113 / 113 (125 slots) | 0.0 | 0.003 | 1.609 | visually_observable | True |
| P0032 | conversation | lifecycle | hospital_1 | 48 / 216 (125 slots) | 0.0 | 20.833 | 0.147 | visually_observable | True |
| P0033 | conversation | approach_and_stop | airport | 0 / 250 (125 slots) | 0.0169 | 21.894 | 0.099 | visually_observable | True |
| P0039 | conversation | approach_and_stop | airport | 0 / 246 (125 slots) | 0.3535 | 134.186 | 2.078 | visually_observable | True |
| P0040 | conversation | static | hospital_1 | 432 / 432 (125 slots) | 0.0001 | 0.0 | 0.153 | visually_observable | True |
| P0045 | conversation | static | airport | 203 / 203 (125 slots) | 0.0 | 0.001 | 1.903 | visually_observable | True |
| P0051 | conversation | static | airport | 188 / 188 (125 slots) | 0.0 | 0.001 | 1.623 | visually_observable | True |
| P0057 | conversation | static | socialcost_office | 250 / 250 (125 slots) | 0.0 | 0.002 | 0.018 | visually_observable | True |
| P0058 | conversation | approach_and_stop | socialcost_office | 1 / 204 (125 slots) | 0.0226 | 107.817 | 0.683 | visually_observable | True |
| P0065 | stop_gesture | frontal/active_from_start | airport | 95 / 95 (125 slots) | 0.0 | 0.0 | 2.921 | visually_observable | True |
| P0067 | stop_gesture | oblique_30/active_from_start | hospital_1 | 120 / 120 (125 slots) | 0.0 | 0.0 | 2.902 | visually_observable | True |
| P0070 | stop_gesture | oblique_60/pulse | socialcost_office | 125 / 125 (125 slots) | 0.0 | 0.0 | 3.282 | visually_observable | True |
| P0071 | stop_gesture | frontal/activates_on_approach | airport | 125 / 125 (125 slots) | 0.0 | 0.0 | 1.245 | visually_observable | True |
| P0074 | stop_gesture | oblique_60/active_from_start | airport | 125 / 125 (125 slots) | 0.0 | 0.0 | 0.038 | visually_observable | True |
| P0076 | stop_gesture | oblique_30/active_then_deactivates | hospital_1 | 125 / 125 (125 slots) | 0.0 | 0.0 | 0.083 | visually_observable | True |
| P0084 | stop_gesture | oblique_30/active_from_start | airport | 125 / 125 (125 slots) | 0.0 | 0.0 | 0.168 | visually_observable | True |
| P0098 | stop_gesture | frontal/pulse | hospital_1 | 110 / 110 (125 slots) | 0.0 | 0.0 | 0.130 | visually_observable | True |
| P0108 | stop_gesture | oblique_60/active_from_start | reception | 125 / 125 (125 slots) | 0.0 | 0.0 | 1.553 | visually_observable | True |
| P0109 | stop_gesture | frontal/activates_on_approach | hospital_1 | 109 / 109 (125 slots) | 0.0 | 0.0 | 1.542 | visually_observable | True |
| P0115 | wheelchair | slow_forward_same | hospital_1 | 0 / 20 (125 slots) | 0.0815 | 1.215 | 1.537 | visually_observable | True |
| P0121 | wheelchair | slow_forward_same | airport | 5 / 25 (125 slots) | 0.0254 | 0.25 | 1.835 | visually_observable | True |
| P0137 | wheelchair | crossing | hospital_1 | 0 / 99 (125 slots) | 0.1183 | 5.393 | 0.179 | visually_observable | True |
| P0138 | wheelchair | stationary | socialcost_office | 125 / 125 (125 slots) | 0.0 | 0.0 | 1.607 | visually_observable | True |
| P0141 | wheelchair | slow_forward_opposite | hospital_1 | 0 / 0 (125 slots) | 0.0 | 0.0 | 2.122 | visually_observable | True |
| P0146 | wheelchair | turning | hospital_1 | 0 / 41 (125 slots) | 1.8255 | 176.26 | 0.032 | visually_observable | True |
| P0149 | wheelchair | slow_forward_opposite | airport | 4 / 4 (125 slots) | 0.0 | 0.0 | 0.326 | visually_observable | True |
| P0150 | wheelchair | boundary_wait | hospital_1 | 125 / 125 (125 slots) | 0.0 | 0.0 | 0.073 | visually_observable | True |
| P0154 | wheelchair | slow_forward_opposite | reception | 0 / 15 (125 slots) | 0.1467 | 53.318 | 0.057 | visually_observable | True |
| P0156 | wheelchair | crossing | hospital_1 | 67 / 95 (125 slots) | 0.1384 | 0.784 | 3.230 | visually_observable | True |
| P0165 | queue | sparse/parallel/advancing | socialcost_office | 33 / 665 (125 slots) | 4.1508 | 100.416 | 0.248 | hidden_state | True |
| P0166 | queue | dense/perpendicular/static | hospital_1 | 745 / 745 (125 slots) | 0.0001 | 0.001 | 0.039 | hidden_state | True |
| P0168 | queue | normal/oblique/static | socialcost_office | 1000 / 1000 (125 slots) | 0.0001 | 0.0 | 1.610 | hidden_state | True |
| P0172 | queue | dense/perpendicular/static | socialcost_office | 417 / 417 (125 slots) | 0.0 | 0.0 | 0.026 | hidden_state | True |
| P0173 | queue | normal/perpendicular/advancing | hospital_1 | 42 / 686 (125 slots) | 2.8335 | 178.741 | 2.960 | hidden_state | True |
| P0181 | queue | normal/perpendicular/advancing | airport | 0 / 295 (125 slots) | 3.0114 | 117.751 | 2.265 | hidden_state | True |
| P0182 | queue | dense/perpendicular/advancing | socialcost_office | 4 / 372 (125 slots) | 2.9881 | 37.746 | 0.128 | hidden_state | True |
| P0191 | queue | dense/perpendicular/advancing | socialcost_office | 0 / 544 (125 slots) | 4.7628 | 127.886 | 2.834 | hidden_state | True |
| P0199 | queue | dense/oblique/static | socialcost_office | 996 / 996 (125 slots) | 0.0001 | 0.0 | 0.056 | hidden_state | True |
| P0206 | queue | sparse/perpendicular/static | reception | 950 / 950 (125 slots) | 0.0 | 0.0 | 0.011 | hidden_state | True |

## Episodes

| episode | family | cond | world | split | ok | failing |
|---|---|---|---|---|---|---|
| 000019 | conversation | on | A | train | True |  |
| 000020 | conversation | off | A | train | True |  |
| 000021 | conversation | on | D | test_b | True |  |
| 000022 | conversation | off | D | test_b | True |  |
| 000045 | conversation | on | B | test_a | True |  |
| 000046 | conversation | off | B | test_a | True |  |
| 000047 | conversation | on | A | test_c | True |  |
| 000048 | conversation | off | A | test_c | True |  |
| 000063 | conversation | on | B | train | True |  |
| 000064 | conversation | off | B | train | True |  |
| 000065 | conversation | on | C | val | True |  |
| 000066 | conversation | off | C | val | True |  |
| 000077 | conversation | on | C | train | True |  |
| 000078 | conversation | off | C | train | True |  |
| 000079 | conversation | on | B | train | True |  |
| 000080 | conversation | off | B | train | True |  |
| 000089 | conversation | on | C | train | True |  |
| 000090 | conversation | off | C | train | True |  |
| 000101 | conversation | on | C | train | True |  |
| 000102 | conversation | off | C | train | True |  |
| 000113 | conversation | on | A | train | True |  |
| 000114 | conversation | off | A | train | True |  |
| 000115 | conversation | on | A | train | True |  |
| 000116 | conversation | off | A | train | True |  |
| 000129 | stop_gesture | on | C | test_c | True |  |
| 000130 | stop_gesture | off | C | test_c | True |  |
| 000133 | stop_gesture | on | B | train | True |  |
| 000134 | stop_gesture | off | B | train | True |  |
| 000139 | stop_gesture | on | A | val | True |  |
| 000140 | stop_gesture | off | A | val | True |  |
| 000141 | stop_gesture | on | C | train | True |  |
| 000142 | stop_gesture | off | C | train | True |  |
| 000147 | stop_gesture | on | C | train | True |  |
| 000148 | stop_gesture | off | C | train | True |  |
| 000151 | stop_gesture | on | B | train | True |  |
| 000152 | stop_gesture | off | B | train | True |  |
| 000167 | stop_gesture | on | C | train | True |  |
| 000168 | stop_gesture | off | C | train | True |  |
| 000195 | stop_gesture | on | B | test_a | True |  |
| 000196 | stop_gesture | off | B | test_a | True |  |
| 000215 | stop_gesture | on | D | test_b | True |  |
| 000216 | stop_gesture | off | D | test_b | True |  |
| 000217 | stop_gesture | on | B | train | True |  |
| 000218 | stop_gesture | off | B | train | True |  |
| 000229 | wheelchair | on | B | train | True |  |
| 000230 | wheelchair | off | B | train | True |  |
| 000241 | wheelchair | on | C | train | True |  |
| 000242 | wheelchair | off | C | train | True |  |
| 000273 | wheelchair | on | B | test_c | True |  |
| 000274 | wheelchair | off | B | test_c | True |  |
| 000275 | wheelchair | on | A | train | True |  |
| 000276 | wheelchair | off | A | train | True |  |
| 000281 | wheelchair | on | B | train | True |  |
| 000282 | wheelchair | off | B | train | True |  |
| 000291 | wheelchair | on | B | val | True |  |
| 000292 | wheelchair | off | B | val | True |  |
| 000297 | wheelchair | on | C | train | True |  |
| 000298 | wheelchair | off | C | train | True |  |
| 000299 | wheelchair | on | B | test_a | True |  |
| 000300 | wheelchair | off | B | test_a | True |  |
| 000307 | wheelchair | on | D | test_b | True |  |
| 000308 | wheelchair | off | D | test_b | True |  |
| 000311 | wheelchair | on | B | train | True |  |
| 000312 | wheelchair | off | B | train | True |  |
| 000329 | queue | on | A | train | True |  |
| 000330 | queue | off | A | train | True |  |
| 000331 | queue | on | B | train | True |  |
| 000332 | queue | off | B | train | True |  |
| 000335 | queue | on | A | test_a | True |  |
| 000336 | queue | off | A | test_a | True |  |
| 000343 | queue | on | A | train | True |  |
| 000344 | queue | off | A | train | True |  |
| 000345 | queue | on | B | test_c | True |  |
| 000346 | queue | off | B | test_c | True |  |
| 000361 | queue | on | C | train | True |  |
| 000362 | queue | off | C | train | True |  |
| 000363 | queue | on | A | train | True |  |
| 000364 | queue | off | A | train | True |  |
| 000381 | queue | on | A | val | True |  |
| 000382 | queue | off | A | val | True |  |
| 000397 | queue | on | A | train | True |  |
| 000398 | queue | off | A | train | True |  |
| 000411 | queue | on | D | test_b | True |  |
| 000412 | queue | off | D | test_b | True |  |
| 000425 | group_walking | na | C | test_c | True |  |
| 000426 | group_walking | na | C | train | True |  |
| 000433 | group_walking | na | A | train | True |  |
| 000436 | group_walking | na | D | test_b | True |  |
| 000446 | group_walking | na | B | train | True |  |
| 000448 | group_walking | na | C | train | True |  |
| 000453 | group_walking | na | A | test_a | True |  |
| 000462 | group_walking | na | A | train | True |  |
| 000463 | group_walking | na | A | train | True |  |
| 000466 | group_walking | na | A | train | True |  |
| 000474 | group_walking | na | A | val | True |  |
| 000476 | group_walking | na | B | train | True |  |
| 000483 | partial_observability | na | A | train | True |  |
| 000485 | partial_observability | na | B | train | True |  |
| 000489 | partial_observability | na | B | train | True |  |
| 000494 | partial_observability | na | C | train | True |  |
| 000498 | partial_observability | na | B | test_c | True |  |
| 000505 | partial_observability | na | A | train | True |  |
| 000524 | partial_observability | na | B | val | True |  |
| 000525 | partial_observability | na | C | train | True |  |
| 000526 | partial_observability | na | D | test_b | True |  |
| 000527 | partial_observability | na | A | train | True |  |
| 000536 | partial_observability | na | C | train | True |  |
| 000538 | partial_observability | na | C | val | True |  |
| 000546 | partial_observability | na | A | train | True |  |
| 000547 | partial_observability | na | C | test_a | True |  |
| 000556 | partial_observability | na | A | test_a | True |  |
| 000560 | partial_observability | na | B | test_c | True |  |
| 000561 | controls | na | D | test_b | True |  |
| 000568 | controls | na | C | test_c | True |  |
| 000569 | controls | na | C | train | True |  |
| 000573 | controls | na | C | train | True |  |
| 000577 | controls | na | C | test_a | True |  |
| 000580 | controls | na | A | val | True |  |
| 000584 | controls | na | A | train | True |  |
| 000586 | controls | na | A | train | True |  |
