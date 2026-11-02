#!/bin/bash
# Detached: videos of the best checkpoint (AGENT_5_NAV100K_STATIC last_model) on every scenario,
# 10 matched episodes each, same run_seeds as the matched eval in trace/.
cd /opt/arena_ws
source /opt/arena_ws/source
ROOT=/opt/arena_ws/data/mf_nav
ST=$ROOT/viz/STATUS.txt
exec >>"$ROOT/viz/run_viz.log" 2>&1
NAME=AGENT_5_NAV100K_STATIC
CFG=$ROOT/configs/$NAME.yaml
ZIP=/opt/arena_ws/data/agents/$NAME/last_model.zip
TAG=NAV100K_STATIC_100k
for stage in empty static ped1 headon; do
  echo "VIZ_BEST_BEGIN $stage $(date -Is)" | tee -a "$ST"
  bash "$ROOT/viz_one.sh" "$NAME" "$CFG" "$stage" "nav-$stage-5" 10 "$ZIP" "$TAG"
  echo "VIZ_BEST_EXIT $stage $? $(date -Is)" | tee -a "$ST"
  f="$ROOT/viz/data/${TAG}_${stage}.json"
  python3 "$ROOT/render_episode.py" overview "$f" "$ROOT/viz/overview/${TAG}_${stage}.png"
  for i in $(seq 0 9); do
    python3 "$ROOT/render_episode.py" video "$f" $i "$ROOT/viz/videos/${TAG}_${stage}_ep$((i+1)).mp4" || true
  done
  echo "VIZ_BEST_RENDERED $stage $(date -Is)" | tee -a "$ST"
done
echo "VIZ_BEST_DONE $(date -Is)" | tee -a "$ST"
