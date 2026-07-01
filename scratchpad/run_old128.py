"""Baseline-to-beat: train the OLD d=128 arch (2.76M params) FROM SCRATCH on users 1-100, eval on
101-200. Isolates architecture from data (the published old model trained on 5000 users; this trains
on the SAME 100 users our small champion uses). Memory: d=128 won't fit MAX=66000 on 12GB, so we use
the small-chunk db (sc8k, max chunk ~16384) at MAX=18000 -> full coverage (no drops), ~8GB.
NOTE: architecture.py must already be the d=128 arch (the .cmd swaps it before running this)."""
import os, re, sys, json, time, subprocess

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
os.chdir(ROOT); sys.path.insert(0, ROOT)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
from rwkv.train_rwkv import get_groups

DB, DBSZ, MAX, EP = "train_db_sc8k", 4_000_000_000, 18000, 6
groups = get_groups(DB, DBSZ, MAX, list(range(1, 101)))
steps = EP * len(groups)
ckpt = f"scratchpad/old128_ws/old128_{steps}.pth"
print(f"groups={len(groups)} total_steps={steps} ckpt={ckpt}", flush=True)

TRAIN = f"""TRAIN_USERS_START = 1
TRAIN_USERS_END = 100
VALIDATE_USERS_START = 101
VALIDATE_USERS_END = 102
TRAIN_DATASET_LMDB_PATH = "{DB}"
TRAIN_DATASET_LMDB_SIZE = {DBSZ}
VALIDATE_DATASET_LMDB_PATH = "test_db"
VALIDATE_DATASET_LMDB_SIZE = 8_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 2_000_000_000
NUM_FETCH_PROCESSES = 7
MAX_TRAIN_GLOBAL_LEN = {MAX}
TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = 200
EPOCHS = {EP}
VALIDATE_EVERY = {steps}
PEAK_LR = 7e-4
LOAD_MODEL = false
SAVE_MODEL_FOLDER = "scratchpad/old128_ws"
SAVE_MODEL_PREFIX = "old128"
DEVICE = "cuda"
DTYPE = "bfloat16"
USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
"""
EVAL = f"""FILE_AHEAD = "RWKV-old128"
FILE_IMM = "RWKV-P-old128"
MODEL_PATH = "{ckpt}"
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
"""
open("scratchpad/old128_train.toml", "w").write(TRAIN)
open("scratchpad/old128_eval.toml", "w").write(EVAL)


def run(cmd, outfile):
    print(f"\n$ {' '.join(cmd)} -> {outfile}", flush=True)
    with open(outfile, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return open(outfile, encoding="utf-8", errors="replace").read()


t0 = time.time()
tout = run([PY, "-u", "-m", "rwkv.train_rwkv", "--config", "scratchpad/old128_train.toml"],
           "scratchpad/old128_train.out")
train_min = (time.time() - t0) / 60.0
lns = re.findall(r"loss_n per second:\s*([\d.]+)", tout)
thr = float(lns[-1]) if lns else float("nan")
res = {"name": "old d=128 (trained 1-100)", "params": 2762884, "d_model": 128,
       "trained_on": "1-100", "chunk": "sc8k 8192", "max_train": MAX,
       "steps": steps, "train_min": round(train_min, 1), "throughput": round(thr, 0)}

if not os.path.exists(ckpt):
    res["error"] = "ckpt missing (train OOM/crash?)"
    print("!! ckpt missing; train tail:\n" + tout[-2000:], flush=True)
else:
    for f in ["result/RWKV-old128.jsonl", "result/RWKV-P-old128.jsonl"]:
        try: os.remove(f)
        except FileNotFoundError: pass
    eout = run([PY, "-u", "-m", "rwkv.get_result", "--config", "scratchpad/old128_eval.toml"],
               "scratchpad/old128_eval.out")
    try:
        def m(f):
            rows = [json.loads(l) for l in open(f)]
            return sum(r["metrics"]["LogLoss"] for r in rows) / len(rows), len(rows)
        ah, na = m("result/RWKV-old128.jsonl")
        im, ni = m("result/RWKV-P-old128.jsonl")
        res.update({"ahead": round(ah, 6), "imm": round(im, 6), "eval_users": na})
    except Exception as e:
        res["error"] = f"eval parse/OOM: {e}"
        print("!! eval failed:\n" + eout[-2000:], flush=True)

json.dump(res, open("scratchpad/old128_result.json", "w"), indent=2)
print("\n=== OLD d=128 BASELINE RESULT ===")
print(json.dumps(res, indent=2), flush=True)
