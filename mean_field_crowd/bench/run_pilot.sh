#!/bin/bash
# Pilot: Phase 1 forecasting, then seed 5 of the navigation benchmark.
B=/opt/arena_ws/data/mf_bench
bash $B/run_bench.sh forecast
bash $B/run_bench.sh seeds 5
echo "PILOT_DONE $(date -Is)" >>$B/STATUS.txt
