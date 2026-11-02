"""Kept for viz_one.sh. trace_eval.py now records heading, world-frame laser hits, and
pedestrian positions on every step itself, so this only forwards to it.

usage: viz_eval.py CONFIG CHECKPOINT STAGE EPISODES RUN_SEED DEST_JSON
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/opt/arena_ws/data/mf_nav")
import trace_eval  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(trace_eval.main())
