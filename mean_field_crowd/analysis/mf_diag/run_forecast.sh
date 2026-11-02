#!/bin/bash
set +e
source /opt/arena_ws/source >/tmp/mf_forecast_source.log
cd /opt/arena_ws
LOG=/opt/arena_ws/data/mf_diag/forecast.log
exec >>"$LOG" 2>&1
echo "FORECAST_START $(date -Is)"
CFG=/opt/arena_ws/src/Arena/arena_training/configs/agent_5_diag_forecast.yaml
echo "TRAIN_FORECAST $(date -Is)"
arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$CFG"
echo "TRAIN_FORECAST_EXIT $? $(date -Is)"
sleep 8
stop_sim() {
  if [ -f /tmp/mf_diag_runtime.pid ]; then
    kill -- "-$(cat /tmp/mf_diag_runtime.pid)" 2>/dev/null
    kill "$(cat /tmp/mf_diag_runtime.pid)" 2>/dev/null
    rm -f /tmp/mf_diag_runtime.pid
  fi
  sleep 3
}
echo "ROLLOUT_BEGIN forecast headon $(date -Is)"
stop_sim
setsid ros2 launch arena_bringup arena_runtime.launch.py sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo >/tmp/mf_diag_runtime.log 2>&1 &
echo $! >/tmp/mf_diag_runtime.pid
for _ in $(seq 1 40); do
  if timeout 3 ros2 topic echo --once /clock >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
sleep 8
python /opt/arena_ws/data/mf_diag/diag_reward.py "$CFG" /opt/arena_ws/data/agents/AGENT_5_DIAG_FORECAST/last_model.zip headon 3
echo "ROLLOUT_EXIT $? $(date -Is)"
stop_sim
echo "FORECAST_DONE $(date -Is)"
