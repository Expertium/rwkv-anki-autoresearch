"""Find the latest {folder}/{prefix}_{step}.pth (excluding *_optim*) and write a get_result eval toml
to <out> pointing at it. 5k-phase defaults (2026-07-03): eval users 5001-5200 on test_db_5k -- the
100/1500-user workbench era is over. Used when the final step count is data-dependent. Usage:
  python scratchpad/write_eval_toml.py <folder> <prefix> <out_toml> <FILE_AHEAD> <FILE_IMM> [user_start user_end]
user_start/user_end (optional, default 5001 5200): champion runs pass 5001 10000 for the full eval."""
import glob
import os
import re
import sys

folder, prefix, out, fa, fi = sys.argv[1:6]
user_start = sys.argv[6] if len(sys.argv) > 6 else "5001"
user_end = sys.argv[7] if len(sys.argv) > 7 else "5200"
cands = []
for p in glob.glob(f"{folder}/{prefix}_*.pth"):
    b = os.path.basename(p)
    if "optim" in b:
        continue
    m = re.match(rf"{re.escape(prefix)}_(\d+)\.pth$", b)
    if m:
        cands.append((int(m.group(1)), p.replace("\\", "/")))
if not cands:
    print(f"ERROR: no {prefix}_<step>.pth in {folder}")
    sys.exit(1)
step, path = max(cands)
with open(out, "w") as f:
    f.write(f'''FILE_AHEAD = "{fa}"
FILE_IMM = "{fi}"
MODEL_PATH = "{path}"
DEVICE = "cuda"
DTYPE = "bfloat16"
DATASET_LMDB_PATH = "F:/rwkv_lmdb/test_db_5k"
DATASET_LMDB_SIZE = 250_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000
RAW = false
RAW_DB_PATH = "raw/result_db"
RAW_DB_SIZE = 1_000_000_000
USER_START = {user_start}
USER_END = {user_end}
NUM_FETCH_PROCESSES = 7
''')
print(f"wrote {out} -> {path} (step {step})")
