#!/bin/bash
# Detached navigation comparison. Survives the Cursor session because the
# Arena container keeps running it.
cd /opt/arena_ws
source /opt/arena_ws/source
ROOT=/opt/arena_ws/data/mf_nav
mkdir -p "$ROOT/configs" "$ROOT/eval"
exec >>"$ROOT/pipeline.log" 2>&1
echo "PIPELINE_START $(date -Is) pid=$$"
echo $$ >"$ROOT/pipeline.pid"
echo "PIPELINE_START $(date -Is)" >"$ROOT/STATUS.txt"

stop_sim() {
  if [ -f /tmp/mf_nav_runtime.pid ]; then
    kill -- "-$(cat /tmp/mf_nav_runtime.pid)" 2>/dev/null || true
    kill "$(cat /tmp/mf_nav_runtime.pid)" 2>/dev/null || true
    rm -f /tmp/mf_nav_runtime.pid
  fi
  sleep 3
}

choose_device() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${free_mib:-0}" -ge 6500 ]; then
    echo cuda
  else
    echo cpu
  fi
}

DEVICE=$(choose_device)
echo "DEVICE $DEVICE free_check $(date -Is)" | tee -a "$ROOT/STATUS.txt"
python3 "$ROOT/prepare_configs.py" "$DEVICE"

train_one() {
  local name="$1"
  local cfg="$ROOT/configs/${name}.yaml"
  local log="$ROOT/train_${name}.log"
  local zip="/opt/arena_ws/data/agents/${name}/last_model.zip"
  echo "TRAIN_BEGIN $name device=$DEVICE $(date -Is)" | tee -a "$ROOT/STATUS.txt"
  arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$cfg" >"$log" 2>&1
  local ec=$?
  echo "TRAIN_EXIT $name $ec $(date -Is)" | tee -a "$ROOT/STATUS.txt"
  if grep -q OutOfMemoryError "$log"; then
    echo "OOM $name" | tee -a "$ROOT/STATUS.txt"
    if [ "$DEVICE" = cuda ]; then
      DEVICE=cpu
      python3 "$ROOT/prepare_configs.py" cpu
      echo "TRAIN_RETRY $name cpu $(date -Is)" | tee -a "$ROOT/STATUS.txt"
      arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$cfg" >"$log" 2>&1
      ec=$?
      echo "TRAIN_EXIT $name $ec $(date -Is)" | tee -a "$ROOT/STATUS.txt"
    fi
  fi
  if [ ! -f "$zip" ]; then
    echo "TRAIN_MISSING_ZIP $name" | tee -a "$ROOT/STATUS.txt"
    return 1
  fi
  return 0
}

start_runtime() {
  stop_sim
  setsid ros2 launch arena_bringup arena_runtime.launch.py sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo >/tmp/mf_nav_runtime.log 2>&1 &
  echo $! >/tmp/mf_nav_runtime.pid
  for _ in $(seq 1 40); do
    if timeout 3 ros2 topic echo --once /clock >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  sleep 8
}

eval_one() {
  local name="$1"
  local cfg="$2"
  local stage="$3"
  local seed="$4"
  local zip="/opt/arena_ws/data/agents/${name}/last_model.zip"
  local dest="$ROOT/eval/${name}_${stage}.json"
  if [ ! -f "$zip" ]; then
    echo "EVAL_SKIP $name $stage missing zip" | tee -a "$ROOT/STATUS.txt"
    return 0
  fi
  echo "EVAL_BEGIN $name $stage $(date -Is)" | tee -a "$ROOT/STATUS.txt"
  start_runtime
  python3 "$ROOT/eval_one.py" "$cfg" "$zip" "$stage" 5 "$seed" "$dest" >"$ROOT/eval_${name}_${stage}.log" 2>&1
  local ec=$?
  stop_sim
  echo "EVAL_EXIT $name $stage $ec $(date -Is)" | tee -a "$ROOT/STATUS.txt"
  python3 "$ROOT/summarize.py" || true
}

echo "QUEUE progress-empty, static original, static progress, ped1 original, ped1 proximity, ped1 progress" | tee -a "$ROOT/STATUS.txt"

train_one AGENT_5_NAV_PROGRESS || true
eval_one AGENT_5_DIAG_EMPTY /opt/arena_ws/src/Arena/arena_training/configs/agent_5_diag_empty.yaml empty nav-empty-5
eval_one AGENT_5_NAV_PROGRESS "$ROOT/configs/AGENT_5_NAV_PROGRESS.yaml" empty nav-empty-5

train_one AGENT_5_NAV_STATIC || true
train_one AGENT_5_NAV_STATIC_PROGRESS || true
eval_one AGENT_5_NAV_STATIC "$ROOT/configs/AGENT_5_NAV_STATIC.yaml" static nav-static-5
eval_one AGENT_5_NAV_STATIC_PROGRESS "$ROOT/configs/AGENT_5_NAV_STATIC_PROGRESS.yaml" static nav-static-5

train_one AGENT_5_NAV_PED1 || true
train_one AGENT_5_NAV_PED1_RISK || true
train_one AGENT_5_NAV_PED1_PROGRESS || true
eval_one AGENT_5_NAV_PED1 "$ROOT/configs/AGENT_5_NAV_PED1.yaml" ped1 nav-ped1-5
eval_one AGENT_5_NAV_PED1_RISK "$ROOT/configs/AGENT_5_NAV_PED1_RISK.yaml" ped1 nav-ped1-5
eval_one AGENT_5_NAV_PED1_PROGRESS "$ROOT/configs/AGENT_5_NAV_PED1_PROGRESS.yaml" ped1 nav-ped1-5
eval_one AGENT_5_NAV_PED1 "$ROOT/configs/AGENT_5_NAV_PED1.yaml" headon nav-headon-5
eval_one AGENT_5_NAV_PED1_RISK "$ROOT/configs/AGENT_5_NAV_PED1_RISK.yaml" headon nav-headon-5
eval_one AGENT_5_NAV_PED1_PROGRESS "$ROOT/configs/AGENT_5_NAV_PED1_PROGRESS.yaml" headon nav-headon-5

python3 "$ROOT/summarize.py" || true
echo "PIPELINE_DONE $(date -Is)" | tee -a "$ROOT/STATUS.txt"
stop_sim
