#!/bin/bash
# Detached: corrected matched re-eval, then two 100k AGENT_5 runs and their checkpoint evals.
# Eval runtimes use robot.mobile:=rosnav_rl (trace_one.sh), same as arena train.
cd /opt/arena_ws
source /opt/arena_ws/source
ROOT=/opt/arena_ws/data/mf_nav
CFG=$ROOT/configs
ST=$ROOT/STATUS_100K.txt
exec >>"$ROOT/pipeline_100k.log" 2>&1
echo $$ >"$ROOT/pipeline_100k.pid"
echo "PIPELINE_START $(date -Is) pid=$$" | tee -a "$ST"
EPS=10

trace() {  # name cfg stage seed [zip tag]
  echo "EVAL_BEGIN $4 $1 $3 ${6:-} $(date -Is)" | tee -a "$ST"
  bash "$ROOT/trace_one.sh" "$1" "$2" "$3" "$4" "$EPS" "${5:-/opt/arena_ws/data/agents/$1/last_model.zip}" "${6:-$1}"
  echo "EVAL_EXIT $1 $3 ${6:-} $? $(date -Is)" | tee -a "$ST"
}

# A. corrected matched re-eval of the existing 20k checkpoints
trace AGENT_5_DIAG_EMPTY /opt/arena_ws/src/Arena/arena_training/configs/agent_5_diag_empty.yaml empty nav-empty-5
trace AGENT_5_NAV_PROGRESS $CFG/AGENT_5_NAV_PROGRESS.yaml empty nav-empty-5
trace AGENT_5_NAV_STATIC $CFG/AGENT_5_NAV_STATIC.yaml static nav-static-5
trace AGENT_5_NAV_STATIC_PROGRESS $CFG/AGENT_5_NAV_STATIC_PROGRESS.yaml static nav-static-5
for n in AGENT_5_NAV_PED1 AGENT_5_NAV_PED1_RISK AGENT_5_NAV_PED1_PROGRESS; do
  trace $n $CFG/$n.yaml ped1 nav-ped1-5
  trace $n $CFG/$n.yaml headon nav-headon-5
done
echo "PHASE_A_DONE $(date -Is)" | tee -a "$ST"

# B. 100k training from scratch
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
DEVICE=cpu; [ "${free_mib:-0}" -ge 6500 ] && DEVICE=cuda
echo "DEVICE $DEVICE free=$free_mib $(date -Is)" | tee -a "$ST"
python3 "$ROOT/prepare_100k.py" "$DEVICE"
for spec in AGENT_5_NAV100K_EMPTY:empty AGENT_5_NAV100K_STATIC:static; do
  name=${spec%%:*}; stage=${spec##*:}
  pkill -f arena_runtime.launch.py; sleep 3
  echo "TRAIN_BEGIN $name $(date -Is)" | tee -a "$ST"
  arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:="$CFG/$name.yaml" >"$ROOT/train_$name.log" 2>&1
  echo "TRAIN_EXIT $name $? $(date -Is)" | tee -a "$ST"
  pkill -f train_agent.py; pkill -f run_mujoco; sleep 5
  grep -q OutOfMemoryError "$ROOT/train_$name.log" && echo "OOM $name" | tee -a "$ST"
  # C. matched evals of every checkpoint on the training stage
  dir=/opt/arena_ws/data/agents/$name
  for t in 20000 40000 60000 80000; do
    [ -f "$dir/tick_$t.zip" ] && trace $name "$CFG/$name.yaml" $stage nav-$stage-5 "$dir/tick_$t.zip" "${name}_t$t"
  done
  [ -f "$dir/last_model.zip" ] && trace $name "$CFG/$name.yaml" $stage nav-$stage-5 "$dir/last_model.zip" "${name}_t100000"
done
# zero-shot checks of the static final model
d=/opt/arena_ws/data/agents/AGENT_5_NAV100K_STATIC/last_model.zip
if [ -f "$d" ]; then
  trace AGENT_5_NAV100K_STATIC $CFG/AGENT_5_NAV100K_STATIC.yaml empty nav-empty-5 "$d" AGENT_5_NAV100K_STATIC_t100000
  trace AGENT_5_NAV100K_STATIC $CFG/AGENT_5_NAV100K_STATIC.yaml ped1 nav-ped1-5 "$d" AGENT_5_NAV100K_STATIC_t100000
  trace AGENT_5_NAV100K_STATIC $CFG/AGENT_5_NAV100K_STATIC.yaml headon nav-headon-5 "$d" AGENT_5_NAV100K_STATIC_t100000
fi
echo "PIPELINE_DONE $(date -Is)" | tee -a "$ST"
