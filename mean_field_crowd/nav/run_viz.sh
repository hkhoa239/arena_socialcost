#!/bin/bash
# Detached: waits for run_nav100k.sh to finish (one simulator), records 5 matched episodes per
# policy x scenario with laser hits and pedestrians, then renders videos and overviews.
cd /opt/arena_ws
source /opt/arena_ws/source
ROOT=/opt/arena_ws/data/mf_nav
CFG=$ROOT/configs
ST=$ROOT/viz/STATUS.txt
mkdir -p "$ROOT/viz/data" "$ROOT/viz/videos" "$ROOT/viz/overview"
exec >>"$ROOT/viz/run_viz.log" 2>&1
echo "VIZ_WAIT $(date -Is) pid=$$" | tee -a "$ST"
until grep -q PIPELINE_DONE "$ROOT/STATUS_100K.txt" 2>/dev/null; do
  pgrep -f run_nav100k.sh >/dev/null || { grep -q PIPELINE_DONE "$ROOT/STATUS_100K.txt" || echo "VIZ_NOTE pipeline gone without PIPELINE_DONE" | tee -a "$ST"; break; }
  sleep 30
done
sleep 10
A=/opt/arena_ws/data/agents
# name | config | stage | seed | checkpoint | tag
JOBS=(
  "AGENT_5_NAV_PROGRESS|$CFG/AGENT_5_NAV_PROGRESS.yaml|empty|nav-empty-5|$A/AGENT_5_NAV_PROGRESS/last_model.zip|NAV_PROGRESS_20k"
  "AGENT_5_NAV100K_EMPTY|$CFG/AGENT_5_NAV100K_EMPTY.yaml|empty|nav-empty-5|$A/AGENT_5_NAV100K_EMPTY/last_model.zip|NAV100K_EMPTY_100k"
  "AGENT_5_NAV_STATIC_PROGRESS|$CFG/AGENT_5_NAV_STATIC_PROGRESS.yaml|static|nav-static-5|$A/AGENT_5_NAV_STATIC_PROGRESS/last_model.zip|NAV_STATIC_PROGRESS_20k"
  "AGENT_5_NAV100K_STATIC|$CFG/AGENT_5_NAV100K_STATIC.yaml|static|nav-static-5|$A/AGENT_5_NAV100K_STATIC/last_model.zip|NAV100K_STATIC_100k"
  "AGENT_5_NAV_PED1_PROGRESS|$CFG/AGENT_5_NAV_PED1_PROGRESS.yaml|ped1|nav-ped1-5|$A/AGENT_5_NAV_PED1_PROGRESS/last_model.zip|NAV_PED1_PROGRESS_20k"
  "AGENT_5_NAV_PED1_RISK|$CFG/AGENT_5_NAV_PED1_RISK.yaml|ped1|nav-ped1-5|$A/AGENT_5_NAV_PED1_RISK/last_model.zip|NAV_PED1_RISK_20k"
  "AGENT_5_NAV_PED1_PROGRESS|$CFG/AGENT_5_NAV_PED1_PROGRESS.yaml|headon|nav-headon-5|$A/AGENT_5_NAV_PED1_PROGRESS/last_model.zip|NAV_PED1_PROGRESS_20k"
  "AGENT_5_NAV_PED1_RISK|$CFG/AGENT_5_NAV_PED1_RISK.yaml|headon|nav-headon-5|$A/AGENT_5_NAV_PED1_RISK/last_model.zip|NAV_PED1_RISK_20k"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r name cfg stage seed zip tag <<<"$job"
  echo "VIZ_EVAL_BEGIN $tag $stage $(date -Is)" | tee -a "$ST"
  bash "$ROOT/viz_one.sh" "$name" "$cfg" "$stage" "$seed" 5 "$zip" "$tag"
  echo "VIZ_EVAL_EXIT $tag $stage $? $(date -Is)" | tee -a "$ST"
done
for f in "$ROOT"/viz/data/*.json; do
  base=$(basename "$f" .json)
  python3 "$ROOT/render_episode.py" overview "$f" "$ROOT/viz/overview/$base.png"
  for i in 0 1 2 3 4; do
    python3 "$ROOT/render_episode.py" video "$f" $i "$ROOT/viz/videos/${base}_ep$((i+1)).mp4" --stride 2 --fps 10 || true
  done
  echo "VIZ_RENDERED $base $(date -Is)" | tee -a "$ST"
done
echo "VIZ_DONE $(date -Is)" | tee -a "$ST"
