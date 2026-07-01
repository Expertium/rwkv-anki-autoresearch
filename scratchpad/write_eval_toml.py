"""Find the latest {folder}/{prefix}_{step}.pth (excluding *_optim*) and write a get_result eval toml
to <out> pointing at it (eval users 101-200, test_db). Used when the final step count is data-dependent
(e.g. 1 epoch on a variable number of users). Usage:
  python scratchpad/write_eval_toml.py <folder> <prefix> <out_toml> <FILE_AHEAD> <FILE_IMM>"""
import glob
import os
import re
import sys

folder, prefix, out, fa, fi = sys.argv[1:6]
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
DATASET_LMDB_PATH = "test_db"
DATASET_LMDB_SIZE = 8_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 2_000_000_000
RAW = false
RAW_DB_PATH = "raw/result_db"
RAW_DB_SIZE = 1_000_000_000
USER_START = 101
USER_END = 200
NUM_FETCH_PROCESSES = 7
''')
print(f"wrote {out} -> {path} (step {step})")
