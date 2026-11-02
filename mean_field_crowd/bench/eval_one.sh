#!/bin/bash
# eval_one.sh CONFIG CHECKPOINT STAGE RUN_SEED EPISODES OUT_JSON [FORECAST_MODE]
# Own runtime with robot.mobile:=rosnav_rl (never nav2), up to 3 attempts on spawn failures.
cd /opt/arena_ws
source /opt/arena_ws/source
CFG="$1"; ZIP="$2"; STAGE="$3"; SEED="$4"; EPS="$5"; OUT="$6"; MODE="${7:-}"
mkdir -p "$(dirname "$OUT")"
stop_sim() {
  if [ -f /tmp/mf_bench_runtime.pid ]; then
    kill -- "-$(cat /tmp/mf_bench_runtime.pid)" 2>/dev/null || true
    kill "$(cat /tmp/mf_bench_runtime.pid)" 2>/dev/null || true
    rm -f /tmp/mf_bench_runtime.pid
  fi
  pkill -f arena_runtime.launch.py 2>/dev/null; sleep 3
}
for attempt in 1 2 3; do
  stop_sim
  setsid ros2 launch arena_bringup arena_runtime.launch.py sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl >/tmp/mf_bench_runtime.log 2>&1 &
  echo $! >/tmp/mf_bench_runtime.pid
  for _ in $(seq 1 40); do timeout 3 ros2 topic echo --once /clock >/dev/null 2>&1 && break; sleep 2; done
  sleep 8
  if [ "$(pgrep -f nav2_controller/controller_server | wc -l)" != 0 ]; then echo "ABORT nav2 running"; stop_sim; exit 2; fi
  MF_FORECAST_MODE="$MODE" python3 /opt/arena_ws/data/mf_nav/trace_eval.py "$CFG" "$ZIP" "$STAGE" "$EPS" "$SEED" "$OUT" >"${OUT%.json}.log" 2>&1
  [ -f "$OUT" ] && { stop_sim; exit 0; }
  echo "attempt $attempt failed: $(grep -m1 -E 'RuntimeError|Error' "${OUT%.json}.log")"
  sleep 20
done
stop_sim
exit 1
