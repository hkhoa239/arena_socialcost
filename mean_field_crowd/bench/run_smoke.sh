#!/bin/bash
# Detached smoke tests: three short trainings, then short evals on the new scenarios.
cd /opt/arena_ws
source /opt/arena_ws/source
B=/opt/arena_ws/data/mf_bench
NAV=/opt/arena_ws/data/mf_nav
ST=$B/smoke/STATUS.txt
exec >>"$B/smoke/run_smoke.log" 2>&1
echo "SMOKE_START $(date -Is) pid=$$" | tee -a "$ST"
python3 $B/smoke_configs.py cuda
for name in BENCH_RESIDUAL_FWD_S5_SMOKE BENCH_ONLINE_FWD_S5_SMOKE BENCH_A5_FREE_S5_SMOKE; do
  pkill -f arena_runtime.launch.py; sleep 3
  echo "TRAIN_BEGIN $name $(date -Is)" | tee -a "$ST"
  arena train sim:=mujoco headless:=true world:=map_empty human:=arena humansim.local_planner:=pedvo robot.mobile:=rosnav_rl train_config:=$B/smoke/$name.yaml >"$B/smoke/train_$name.log" 2>&1
  echo "TRAIN_EXIT $name $? $(date -Is)" | tee -a "$ST"
  pkill -f train_agent.py; pkill -f run_mujoco; sleep 5
  d=/opt/arena_ws/data/agents/$name
  ls $d | tr '\n' ' ' | sed "s/^/FILES $name: /" | tee -a "$ST"; echo | tee -a "$ST"
  [ -f $d/last_model.zip ] && python3 $B/check_residual.py $d/last_model.zip | tee -a "$ST"
done
# eval smoke: residual smoke checkpoint on the new scenarios, 2 episodes each
for stage in b_dense b_stress b_headon b_crossing; do
  echo "EVAL_BEGIN $stage $(date -Is)" | tee -a "$ST"
  bash $NAV/viz_one.sh BENCH_RESIDUAL_FWD_S5_SMOKE $B/smoke/BENCH_RESIDUAL_FWD_S5_SMOKE.yaml $stage bench-smoke-$stage 2 /opt/arena_ws/data/agents/BENCH_RESIDUAL_FWD_S5_SMOKE/last_model.zip SMOKE_RESIDUAL
  echo "EVAL_EXIT $stage $? $(date -Is)" | tee -a "$ST"
done
echo "SMOKE_DONE $(date -Is)" | tee -a "$ST"
