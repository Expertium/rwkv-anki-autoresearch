"""Batch-size (MAX_TRAIN_GLOBAL_LEN) / throughput sweep BEFORE 5k HP tuning (Andrew).

Goal: find the MAX_TRAIN_GLOBAL_LEN that maximizes GPU training throughput (reviews/s) without OOMing
the 12 GB card. Bigger MAX packs more 8192-review chunks per group -> bigger WKV batch dim B -> better
GPU utilization, until VRAM runs out. Floor = 66000 (below it get_groups drops data).

Runs the champion recipe (H=2/K=16, empty_cache=0 = realistic) on train_db_sc8k_1500 for ~120 steps per
MAX via train_rwkv's RWKV_MAX_STEPS bench mode (30 warmup steps excluded). Parses BENCH_RESULT + counts
CUDA OOMs (train_rwkv's try/except swallows them, so we detect via the 'out of memory' log text). Stops
climbing once a run OOMs or peak reserved VRAM nears the limit. Launch DETACHED; monitor batch_sweep.log.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
BENCH_DIR = ROOT / "scratchpad" / "bench"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "scratchpad" / "batch_sweep_results.jsonl"

MAX_VALUES = [66000, 88000, 110000, 132000, 154000, 176000, 200000, 240000]
CAP_STEPS = 120
WARMUP = 30
VRAM_CEILING_GB = 11.3   # stop climbing past this (leave headroom on the 12 GB card)

CFG_TMPL = """TRAIN_USERS_START = 1000
TRAIN_USERS_END = 2499
VALIDATE_USERS_START = 101
VALIDATE_USERS_END = 110
TRAIN_DATASET_LMDB_PATH = "train_db_sc8k_1500"
TRAIN_DATASET_LMDB_SIZE = 80_000_000_000
VALIDATE_DATASET_LMDB_PATH = "test_db"
VALIDATE_DATASET_LMDB_SIZE = 8_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 2_000_000_000
NUM_FETCH_PROCESSES = 10
MAX_TRAIN_GLOBAL_LEN = {maxlen}
TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = 100
EPOCHS = 50
VALIDATE_EVERY = 100000000
PEAK_LR = 1e-3
LOAD_MODEL = false
SAVE_MODEL_FOLDER = "scratchpad/bench"
SAVE_MODEL_PREFIX = "bench{maxlen}"
DEVICE = "cuda"
DTYPE = "bfloat16"
USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
"""


def run_one(maxlen):
    cfg_path = BENCH_DIR / f"bench_{maxlen}.toml"
    cfg_path.write_text(CFG_TMPL.format(maxlen=maxlen))
    log_path = BENCH_DIR / f"bench_{maxlen}.log"

    env = dict(os.environ)
    env.update({
        "RWKV_N_HEADS": "2", "RWKV_HEAD_DIM": "16",
        "RWKV_EMPTY_CACHE_EVERY": "0",
        "RWKV_MAX_STEPS": str(CAP_STEPS), "RWKV_BENCH_WARMUP": str(WARMUP),
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "10",
        "PYTHONPATH": str(ROOT),
    })
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    print(f"\n=== MAX_TRAIN_GLOBAL_LEN={maxlen}: running {CAP_STEPS} steps ===", flush=True)
    with open(log_path, "w") as lf:
        subprocess.run([py, "-u", "-m", "rwkv.train_rwkv", "--config", f"scratchpad/bench/bench_{maxlen}.toml"],
                       cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    text = log_path.read_text(errors="replace")
    oom = len(re.findall(r"out of memory", text, re.IGNORECASE))
    m = re.search(r"BENCH_RESULT .*", text)
    if not m:
        print(f"  MAX={maxlen}: NO BENCH_RESULT (crash?). oom_hits={oom}", flush=True)
        return {"maxlen": maxlen, "ok": False, "oom": oom}
    line = m.group(0)
    def g(key):
        mm = re.search(rf"{key}=([0-9.]+)", line)
        return float(mm.group(1)) if mm else None
    rec = {"maxlen": maxlen, "ok": True, "oom": oom,
           "steps_per_sec": g("steps_per_sec"), "reviews_per_sec": g("reviews_per_sec"),
           "peak_reserved_gb": g("peak_reserved_gb"), "elapsed_s": g("elapsed_s")}
    with open(RESULTS, "a") as rf:
        rf.write(json.dumps(rec) + "\n")
    print(f"  MAX={maxlen}: steps/s={rec['steps_per_sec']:.3f} reviews/s={rec['reviews_per_sec']:.0f} "
          f"peak_vram={rec['peak_reserved_gb']:.2f}GB oom_hits={oom}", flush=True)
    return rec


def main():
    results = []
    for maxlen in MAX_VALUES:
        rec = run_one(maxlen)
        results.append(rec)
        if not rec.get("ok"):
            print(f"STOP: MAX={maxlen} crashed/no result -> ceiling reached.", flush=True)
            break
        if rec["oom"] > 0:
            print(f"STOP: MAX={maxlen} hit {rec['oom']} OOM(s) -> over the limit; pick a smaller MAX.", flush=True)
            break
        if rec["peak_reserved_gb"] and rec["peak_reserved_gb"] >= VRAM_CEILING_GB:
            print(f"STOP: MAX={maxlen} peak {rec['peak_reserved_gb']:.2f}GB >= {VRAM_CEILING_GB}GB ceiling.", flush=True)
            break

    print("\n================ BATCH SWEEP SUMMARY ================", flush=True)
    print(f"{'MAX':>8} {'steps/s':>9} {'reviews/s':>11} {'peak_GB':>8} {'oom':>4}", flush=True)
    safe = []
    for r in results:
        if r.get("ok"):
            print(f"{r['maxlen']:>8} {r['steps_per_sec']:>9.3f} {r['reviews_per_sec']:>11.0f} "
                  f"{r['peak_reserved_gb']:>8.2f} {r['oom']:>4}", flush=True)
            if r["oom"] == 0 and r["peak_reserved_gb"] < VRAM_CEILING_GB:
                safe.append(r)
        else:
            print(f"{r['maxlen']:>8} {'CRASH/OOM':>9}", flush=True)
    if safe:
        best = max(safe, key=lambda r: r["reviews_per_sec"])
        print(f"\nRECOMMENDED MAX_TRAIN_GLOBAL_LEN = {best['maxlen']}  "
              f"(reviews/s={best['reviews_per_sec']:.0f}, peak_vram={best['peak_reserved_gb']:.2f}GB)", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
