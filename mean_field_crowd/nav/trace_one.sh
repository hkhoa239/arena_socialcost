#!/bin/bash
# trace_one.sh NAME CONFIG STAGE SEED [EPISODES]
# Starts its own runtime, runs trace_eval.py, stops the runtime.
cd /opt/arena_ws
source /opt/arena_ws/source
ROOT=/opt/arena_ws/data/mf_nav
NAME="$1"; CFG="$2"; STAGE="$3"; SEED="$4"; EPS="${5:-5}"; ZIP="${6:-/opt/arena_ws/data/agents/${NAME}/last_model.zip}"
TAG="${7:-$NAME}"
mkdir -p "$ROOT/trace"
stop_sim() {
  if [ -f /tmp/mf_nav_runtime.pid ]; then
    kill -- "-$(cat /tmp/mf_nav_runtime.pid)" 2>/dev/null || true
    kill "$(cat /tmp/mf_nav_runtime.pid)" 2>/dev/null || true
    rm -f /tmp/mf_nav_runtime.pid
  fi
  sleep 3
}
stop_sim
setsid ros2 launch arena_bringup arena_runtime.launch.py sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl >/tmp/mf_nav_runtime.log 2>&1 &
echo $! >/tmp/mf_nav_runtime.pid
for _ in $(seq 1 40); do
  timeout 3 ros2 topic echo --once /clock >/dev/null 2>&1 && break
  sleep 2
done
sleep 8
NAV2=$(pgrep -f "nav2_controller/controller_server" | wc -l)
echo "TRACE_BEGIN $TAG $STAGE nav2_procs=$NAV2 $(date -Is)" >>"$ROOT/trace/STATUS.txt"
if [ "$NAV2" != 0 ]; then echo "TRACE_ABORT nav2 running" >>"$ROOT/trace/STATUS.txt"; stop_sim; exit 1; fi
echo "TRACE_RUN $TAG $STAGE $(date -Is)" >>"$ROOT/trace/STATUS.txt"
python3 "$ROOT/trace_eval.py" "$CFG" "$ZIP" "$STAGE" "$EPS" "$SEED" "$ROOT/trace/${TAG}_${STAGE}.json" >"$ROOT/trace/${TAG}_${STAGE}.log" 2>&1
echo "TRACE_EXIT $TAG $STAGE $? $(date -Is)" >>"$ROOT/trace/STATUS.txt"
stop_sim
