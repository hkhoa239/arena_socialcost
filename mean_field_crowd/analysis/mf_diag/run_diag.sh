#!/bin/bash
# Brief reward diagnostic. One simulator at a time.
# 1. Reward terms of the 50k AGENT_5 policy on collision and crowd episodes.
# 2. AGENT_5 trained for 20,000 steps with no pedestrians.
# 3. 12,000-step pedestrian-stage runs with and without the current-distance penalty.
# 4. Head-on rollouts of those two short policies.
set +e
source /opt/arena_ws/source >/tmp/mf_diag_source.log
cd /opt/arena_ws
LOG=/opt/arena_ws/data/mf_diag/diag.log
mkdir -p /opt/arena_ws/data/mf_diag
exec >>"$LOG" 2>&1
echo "DIAG_START $(date -Is)"

stop_sim() {
  if [ -f /tmp/mf_diag_runtime.pid ]; then
    kill -- "-$(cat /tmp/mf_diag_runtime.pid)" 2>/dev/null
    kill "$(cat /tmp/mf_diag_runtime.pid)" 2>/dev/null
    rm -f /tmp/mf_diag_runtime.pid
  fi
  sleep 3
}

rollout() {
  local config="$1" checkpoint="$2" stage="$3" episodes="$4"
  echo "ROLLOUT_BEGIN $checkpoint $stage $(date -Is)"
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
  python /opt/arena_ws/data/mf_diag/diag_reward.py "$config" "$checkpoint" "$stage" "$episodes"
  echo "ROLLOUT_EXIT $? $(date -Is)"
  stop_sim
}

PILOT=/opt/arena_ws/src/Arena/arena_training/configs/sb_agent_5_pilot50k.yaml
EMPTY=/opt/arena_ws/src/Arena/arena_training/configs/agent_5_diag_empty.yaml
BASE=/opt/arena_ws/src/Arena/arena_training/configs/agent_5_diag_base.yaml
RISK=/opt/arena_ws/src/Arena/arena_training/configs/agent_5_diag_risk.yaml
ZIP5=/opt/arena_ws/data/agents/AGENT_5_PILOT50K/last_model.zip

rollout "$PILOT" "$ZIP5" headon 3
rollout "$PILOT" "$ZIP5" peds_5 2

echo "TRAIN_EMPTY $(date -Is)"
arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$EMPTY"
echo "TRAIN_EMPTY_EXIT $? $(date -Is)"
sleep 8

echo "TRAIN_BASE $(date -Is)"
arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$BASE"
echo "TRAIN_BASE_EXIT $? $(date -Is)"
sleep 8

echo "TRAIN_RISK $(date -Is)"
arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$RISK"
echo "TRAIN_RISK_EXIT $? $(date -Is)"
sleep 8

rollout "$BASE" /opt/arena_ws/data/agents/AGENT_5_DIAG_BASE/last_model.zip headon 3
rollout "$RISK" /opt/arena_ws/data/agents/AGENT_5_DIAG_RISK/last_model.zip headon 3
echo "DIAG_DONE $(date -Is)"
