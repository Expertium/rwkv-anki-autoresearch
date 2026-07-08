"""Set up a TRAIN_MODE=D cosine-decay phase from a WS-final checkpoint whose final step count is
data-dependent. Finds the latest {folder}/{ws_prefix}_{step}.pth (excl *optim*), copies its optimizer
to the resume form {ws_prefix}_{step}_optim.pth, and writes a decay toml that loads it. Usage:
  python write_decay_setup.py <folder> <ws_prefix> <decay_prefix> <out_toml> <train_db> <ustart> <uend> <decay_epochs> [peak_lr]
decay_epochs may be fractional (e.g. 0.27) -> total decay steps = int(decay_epochs * num_groups).
peak_lr (optional, default 1e-3) sets the decay's starting LR -- the cosine decays from peak_lr to 0,
so it MUST match the WS phase's peak_lr (the HP tuner passes each trial's own peak_lr)."""
import glob
import os
import re
import shutil
import sys

folder, ws_prefix, decay_prefix, out, train_db, ustart, uend, depochs = sys.argv[1:9]
peak_lr = sys.argv[9] if len(sys.argv) > 9 else "1e-3"
cands = []
for p in glob.glob(f"{folder}/{ws_prefix}_*.pth"):
    b = os.path.basename(p)
    if "optim" in b:
        continue
    m = re.match(rf"{re.escape(ws_prefix)}_(\d+)\.pth$", b)
    if m:
        cands.append((int(m.group(1)), p))
if not cands:
    print(f"ERROR: no {ws_prefix}_<step>.pth in {folder}")
    sys.exit(1)
step, _ = max(cands)
src_optim = f"{folder}/{ws_prefix}_optim_{step}.pth"
dst_optim = f"{folder}/{ws_prefix}_{step}_optim.pth"
if os.path.exists(src_optim):
    shutil.copyfile(src_optim, dst_optim)
    print(f"optim resume: {src_optim} -> {dst_optim}")
else:
    print(f"WARNING: optim {src_optim} missing (decay will start from fresh optimizer state)")
with open(out, "w") as f:
    f.write(f'''# Decay phase auto-written by write_decay_setup.py. {decay_prefix} cosine-decay from {ws_prefix}_{step}.
TRAIN_USERS_START = {ustart}
TRAIN_USERS_END = {uend}
VALIDATE_USERS_START = 5001
VALIDATE_USERS_END = 5010

TRAIN_DATASET_LMDB_PATH = "{train_db}"
TRAIN_DATASET_LMDB_SIZE = 400_000_000_000
VALIDATE_DATASET_LMDB_PATH = "F:/rwkv_lmdb/test_db_5k"
VALIDATE_DATASET_LMDB_SIZE = 250_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000

# 4 fetch workers (was 7, Andrew 2026-07-08): each costs ~2.6 GB RAM and fetch runs far
# ahead of demand (~4 ms get() waits); 4 still fully hides prep. Worker count never
# affects batch content/order (seeded shuffle), only parallelism.
NUM_FETCH_PROCESSES = 4
MAX_TRAIN_GLOBAL_LEN = 110000

TRAIN_MODE = "D"
STEP_OFFSET = 1
WARMUP_STEPS = 0
EPOCHS = {depochs}
VALIDATE_EVERY = 100000
PEAK_LR = {peak_lr}

LOAD_MODEL = true
LOAD_MODEL_FOLDER = "{folder}"
LOAD_MODEL_NAME = "{ws_prefix}_{step}"
SAVE_MODEL_FOLDER = "{folder}"
SAVE_MODEL_PREFIX = "{decay_prefix}"
DEVICE = "cuda"
DTYPE = "bfloat16"

USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
''')
print(f"wrote {out}: decay {depochs} epochs from {ws_prefix}_{step} (WS-final step {step})")
