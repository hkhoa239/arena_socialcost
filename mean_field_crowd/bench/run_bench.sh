#!/bin/bash
# Mean-field navigation benchmark pipeline. Detached-safe: every step writes its output file and
# is skipped when that file exists, so the script can be restarted and never overwrites results.
#
#   bash run_bench.sh forecast            Phase 1 data collection + forecasting benchmark
#   bash run_bench.sh seeds 5 [6 7]       train the four arms per seed, validation sweep, test, ablation
#
# Status: data/mf_bench/STATUS.txt (one line per step), partial results: data/mf_bench/results/.
cd /opt/arena_ws
source /opt/arena_ws/source
B=/opt/arena_ws/data/mf_bench
A=/opt/arena_ws/data/agents
ST=$B/STATUS.txt
mkdir -p $B/eval $B/forecast $B/results $B/logs
exec >>"$B/logs/pipeline.log" 2>&1
status() { echo "$* $(date -Is)" | tee -a "$ST"; }

ARMS="A5_FWD ONLINE_FWD RESIDUAL_FWD A5_FREE"
VAL_STAGES="b_static b_dense"
VAL_EPS=10
TEST_RANDOM="b_empty b_static b_moderate b_dense b_stress"
TEST_SCEN="b_headon b_crossing"
TEST_EPS=25
SCEN_EPS=10
ABLATION_STAGES="b_moderate b_dense b_stress b_headon b_crossing"
CKPTS="t20000 t40000 t60000 t80000 t100000"

zip_of() { # run ckpt
  if [ "$2" = t100000 ]; then echo "$A/$1/last_model.zip"; else echo "$A/$1/tick_${2#t}.zip"; fi
}

evaluate() { # run cfg ckpt stage seed eps [mode]
  local run=$1 cfg=$2 ckpt=$3 stage=$4 seed=$5 eps=$6 mode=${7:-}
  local out=$B/eval/$run/${ckpt}${mode:+__$mode}_${stage}.json
  [ -f "$out" ] && return 0
  local zip; zip=$(zip_of "$run" "$ckpt")
  [ -f "$zip" ] || { status "EVAL_SKIP $run $ckpt $stage missing $zip"; return 0; }
  status "EVAL_BEGIN $run $ckpt ${mode:-normal} $stage"
  bash $B/eval_one.sh "$cfg" "$zip" "$stage" "$seed" "$eps" "$out" "$mode"
  status "EVAL_EXIT $run $ckpt ${mode:-normal} $stage $?"
}

train() { # run
  local run=$1 cfg=$B/configs/$1.yaml log=$B/logs/train_$1.log
  if [ -f "$A/$run/last_model.zip" ] && grep -q "total_timesteps.*| 100352" "$log" 2>/dev/null; then return 0; fi
  if [ -d "$A/$run" ] && [ ! -f "$log" ]; then status "TRAIN_REFUSE $run agent dir exists without a log"; return 1; fi
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  local device=cuda; [ "${free:-0}" -ge 6500 ] || device=cpu
  sed -i "s/^\(\s*device:\).*/\1 $device/" "$cfg"
  for attempt in 1 2; do
    pkill -f arena_runtime.launch.py; pkill -f train_agent.py; pkill -f run_mujoco; sleep 5
    status "TRAIN_BEGIN $run device=$device free_mib=$free attempt=$attempt"
    ( while sleep 300; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/GPU $(date -Is) /"; done ) >>"$B/logs/gpu_$run.log" 2>&1 &
    local mon=$!
    arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$cfg" >>"$log" 2>&1
    local ec=$?
    kill $mon 2>/dev/null
    pkill -f train_agent.py; pkill -f run_mujoco; sleep 5
    status "TRAIN_EXIT $run $ec"
    if [ -f "$A/$run/last_model.zip" ] && grep -q "total_timesteps.*| 100352" "$log"; then return 0; fi
    # A failed run leaves a partial agent dir; keep it for inspection under a new name and retry from scratch.
    [ -d "$A/$run" ] && mv "$A/$run" "$A/${run}_failed_attempt$attempt"
  done
  status "TRAIN_FAILED $run"
  return 1
}

best_val() { # run -> ckpt with the highest validation collision-free success (ties: later checkpoint)
  python3 - "$B/eval/$1" <<'EOF'
import json, sys
from pathlib import Path
best, best_score = "t100000", -1.0
for ckpt in ("t20000", "t40000", "t60000", "t80000", "t100000"):
    scores = []
    for f in Path(sys.argv[1]).glob(f"val_{ckpt}_*.json"):
        eps = json.loads(f.read_text())["episodes"]
        scores += [float(bool(e.get("goal_bonus")) and "SUCCESS" in str(e["reason"]) and not any(ev for s in e["series"] for ev in (s.get("events") or []))) for e in eps]
    if scores and sum(scores) / len(scores) >= best_score:
        best, best_score = ckpt, sum(scores) / len(scores)
print(best)
EOF
}

case "$1" in
forecast)
  status "FORECAST_BEGIN"
  DRIVER_CFG=/opt/arena_ws/data/mf_nav/configs/AGENT_5_NAV100K_STATIC.yaml
  DRIVER=$A/AGENT_5_NAV100K_STATIC/last_model.zip
  for stage in fc_n1 fc_n5 fc_n10 fc_n15 fc_n20 fc_n10_static fc_headon fc_crossing fc_parallel; do
    eps=10; case $stage in fc_headon|fc_crossing|fc_parallel) eps=5;; esac
    out=$B/forecast/$stage.json
    [ -f "$out" ] && continue
    status "FC_COLLECT_BEGIN $stage"
    bash $B/eval_one.sh "$DRIVER_CFG" "$DRIVER" "$stage" "bench-fc-$stage" "$eps" "$out"
    status "FC_COLLECT_EXIT $stage $?"
  done
  python3 $B/forecast_bench.py $B/results/forecast_bench.json $B/forecast/fc_*.json >"$B/logs/forecast_bench.log" 2>&1
  status "FORECAST_DONE $?"
  ;;
seeds)
  shift
  for seed in "$@"; do
    for arm in $ARMS; do
      run=BENCH_${arm}_S$seed
      cfg=$B/configs/$run.yaml
      train "$run" || continue
      for ckpt in $CKPTS; do
        for stage in $VAL_STAGES; do
          out=$B/eval/$run/val_${ckpt}_${stage}.json
          [ -f "$out" ] && continue
          zip=$(zip_of "$run" "$ckpt")
          status "VAL_BEGIN $run $ckpt $stage"
          bash $B/eval_one.sh "$cfg" "$zip" "$stage" "bench-val-$stage" "$VAL_EPS" "$out"
          status "VAL_EXIT $run $ckpt $stage $?"
        done
      done
      best=$(best_val "$run")
      echo "$best" >"$B/eval/$run/best_val_ckpt.txt"
      status "BEST_VAL $run $best"
      for ckpt in $(echo t100000 $best | tr ' ' '\n' | sort -u); do
        for stage in $TEST_RANDOM; do evaluate "$run" "$cfg" "$ckpt" "$stage" "bench-test-$stage" $TEST_EPS; done
        for stage in $TEST_SCEN; do evaluate "$run" "$cfg" "$ckpt" "$stage" "bench-test-$stage" $SCEN_EPS; done
      done
      if [ "$arm" = RESIDUAL_FWD ]; then
        for mode in online zero; do
          for stage in $ABLATION_STAGES; do
            eps=$TEST_EPS; case $stage in b_headon|b_crossing) eps=$SCEN_EPS;; esac
            evaluate "$run" "$cfg" t100000 "$stage" "bench-test-$stage" $eps $mode
          done
        done
      fi
      python3 $B/bench_analyze.py $B/eval $B/results/navigation_partial.json >>"$B/logs/analyze.log" 2>&1
      status "RUN_DONE $run"
    done
    status "SEED_DONE $seed"
  done
  ;;
*)
  echo "usage: run_bench.sh forecast | seeds SEED..."; exit 1;;
esac
