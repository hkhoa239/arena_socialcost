#!/bin/bash
# Retry of the head-on part of run_viz_best.sh (first attempt: map_server timed out at spawn).
cd /opt/arena_ws
source /opt/arena_ws/source
ROOT=/opt/arena_ws/data/mf_nav
ST=$ROOT/viz/STATUS.txt
exec >>"$ROOT/viz/run_viz.log" 2>&1
NAME=AGENT_5_NAV100K_STATIC
TAG=NAV100K_STATIC_100k
f="$ROOT/viz/data/${TAG}_headon.json"
for attempt in 1 2 3; do
  echo "VIZ_HEADON_BEGIN attempt=$attempt $(date -Is)" | tee -a "$ST"
  bash "$ROOT/viz_one.sh" "$NAME" "$ROOT/configs/$NAME.yaml" headon nav-headon-5 10 /opt/arena_ws/data/agents/$NAME/last_model.zip "$TAG"
  [ -f "$f" ] && break
  sleep 20
done
[ -f "$f" ] || { echo "VIZ_HEADON_FAILED $(date -Is)" | tee -a "$ST"; exit 1; }
python3 "$ROOT/render_episode.py" overview "$f" "$ROOT/viz/overview/${TAG}_headon.png"
for i in $(seq 0 9); do python3 "$ROOT/render_episode.py" video "$f" $i "$ROOT/viz/videos/${TAG}_headon_ep$((i+1)).mp4" || true; done
echo "VIZ_HEADON_DONE $(date -Is)" | tee -a "$ST"
