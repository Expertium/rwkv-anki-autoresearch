"""Resolve a training phase's LATEST exported learnable codebooks to stable file names.

train_rwkv (with RWKV_QAT_PQ_LEARN=1 / RWKV_QAT_SHIFT_PQ_LEARN=1) exports
{folder}/{prefix}_wkvcb_{step}.txt and {prefix}_shiftcb_{step}.txt at every checkpoint save.
The cb Parameters are process-globals initialized from the RWKV_QAT_PQ / RWKV_QAT_SHIFT_PQ env
files (NOT part of the model state_dict), so the NEXT phase (decay after WS, eval after decay)
must have its env pointed at the PREVIOUS phase's final exports or codebook<->weight
co-adaptation silently breaks at the seam.

Usage: python scratchpad/resolve_run_cbs.py <folder> <prefix> <out_wkv> <out_shift>
Copies the max-step exports to the stable out paths. Exit 1 (loud) if either is missing.
"""
import glob
import os
import re
import shutil
import sys

folder, prefix, out_wkv, out_shift = sys.argv[1:5]


def latest(kind):
    best_step, best_path = -1, None
    for p in glob.glob(f"{folder}/{prefix}_{kind}_*.txt"):
        m = re.match(rf"{re.escape(prefix)}_{kind}_(\d+)\.txt$", os.path.basename(p))
        if m and int(m.group(1)) > best_step:
            best_step, best_path = int(m.group(1)), p
    return best_step, best_path


ok = True
for kind, out in (("wkvcb", out_wkv), ("shiftcb", out_shift)):
    step, path = latest(kind)
    if path is None:
        print(f"ERROR: no {prefix}_{kind}_<step>.txt in {folder} (LEARN flags on but nothing exported?)")
        ok = False
        continue
    shutil.copyfile(path, out)
    print(f"{kind}: {path} (step {step}) -> {out}")
sys.exit(0 if ok else 1)
