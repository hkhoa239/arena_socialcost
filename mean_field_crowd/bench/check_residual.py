"""Residual weights inside a saved SB3 zip equal the frozen checkpoint file (all extractors)."""

import io
import sys
import zipfile

import torch

FILE = "/opt/arena_ws/data/mf_pedvo/interaction_residual.pt"
reference = torch.load(FILE, map_location="cpu", weights_only=True)
with zipfile.ZipFile(sys.argv[1]) as zf:
    policy = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=False)
found = 0
worst = 0.0
for key, value in policy.items():
    if ".residual.net." in key:
        leaf = key.split(".residual.")[1]
        worst = max(worst, float((value - reference[leaf]).abs().max()))
        found += 1
print(f"RESIDUAL_CHECK {sys.argv[1]} tensors={found} max_abs_diff={worst:.3e} {'FROZEN_OK' if found and worst == 0.0 else 'MISMATCH'}")
