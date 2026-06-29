"""Export the LATEST (highest-step) non-optim checkpoint in a folder to safetensors.

Usage: python scratchpad/export_latest.py <folder> <prefix> <out.safetensors>
Finds <folder>/<prefix>_<step>.pth with the max <step> (excluding *_optim*), then delegates to
pth_to_sft.py for the actual conversion. Robust to variable training lengths (a longer QAT run does
not save to a fixed _124.pth), so pipelines need not hardcode the final step count.
Run from the repo root (so 'scratchpad/pth_to_sft.py' resolves).
"""
import glob
import os
import re
import subprocess
import sys

folder, prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]
cands = []
for p in glob.glob(os.path.join(folder, f"{prefix}_*.pth")):
    b = os.path.basename(p)
    if "optim" in b:
        continue
    m = re.match(rf"{re.escape(prefix)}_(\d+)\.pth$", b)
    if m:
        cands.append((int(m.group(1)), p))
if not cands:
    print(f"ERROR: no {prefix}_<step>.pth checkpoints in {folder}")
    sys.exit(1)
step, path = max(cands)
print(f"latest checkpoint: {path} (step {step})")
sys.exit(subprocess.call([sys.executable, "scratchpad/pth_to_sft.py", path, out]))
