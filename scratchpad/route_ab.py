"""Route-R follow-ups (a) accuracy confirm/sweep + (b) higher-MAX speed, on the 100/100 workbench.
Self-contained orchestrator: for each run it computes total_steps from get_groups (so it knows the
final checkpoint name), writes a train toml + an eval toml, runs train then get_result (101-200),
and parses throughput + by-user-mean LogLoss. Prints a final table vs the route-R baselines
(base65k imm 0.296890 / sc8k imm 0.289628). Sequential (one GPU). Run via run_route_ab.cmd."""
import os, re, sys, json, time, subprocess

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
from rwkv.train_rwkv import get_groups

# name, db, db_size, MAX, epochs
RUNS = [
    ("sc8k_s2",   "train_db_sc8k", 4_000_000_000,  66000, 6),  # a1: confirm sc8k win, 2nd augmentation seed
    ("sc4k",      "train_db_sc4k", 4_000_000_000,  66000, 6),  # a2: even smaller chunk (4096)
    ("sc8k_m132", "train_db_sc8k", 4_000_000_000, 132000, 6),  # b1: higher MAX -> fewer steps (speed)
    ("sc8k_m200", "train_db_sc8k", 4_000_000_000, 200000, 6),  # b2: higher MAX (more)
]

TRAIN_TMPL = """TRAIN_USERS_START = 1
TRAIN_USERS_END = 100
VALIDATE_USERS_START = 101
VALIDATE_USERS_END = 102
TRAIN_DATASET_LMDB_PATH = "{db}"
TRAIN_DATASET_LMDB_SIZE = {db_size}
VALIDATE_DATASET_LMDB_PATH = "test_db"
VALIDATE_DATASET_LMDB_SIZE = 8_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 2_000_000_000
NUM_FETCH_PROCESSES = 7
MAX_TRAIN_GLOBAL_LEN = {max}
TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = 200
EPOCHS = {epochs}
VALIDATE_EVERY = {steps}
PEAK_LR = 7e-4
LOAD_MODEL = false
SAVE_MODEL_FOLDER = "scratchpad/ab_{name}"
SAVE_MODEL_PREFIX = "{name}"
DEVICE = "cuda"
DTYPE = "bfloat16"
USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
"""

EVAL_TMPL = """FILE_AHEAD = "RWKV-ab-{name}"
FILE_IMM = "RWKV-P-ab-{name}"
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


def by_user_mean(fa, fi):
    def m(f):
        rows = [json.loads(l) for l in open(f)]
        return sum(r["metrics"]["LogLoss"] for r in rows) / len(rows)
    return m(f"result/{fa}.jsonl"), m(f"result/{fi}.jsonl")


def run(cmd, outfile):
    print(f"\n$ {' '.join(cmd)}  -> {outfile}", flush=True)
    with open(outfile, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return open(outfile, encoding="utf-8", errors="replace").read()


summary = []
for name, db, db_size, MAX, epochs in RUNS:
    print(f"\n================ RUN {name} (db={db} MAX={MAX}) ================", flush=True)
    groups = get_groups(db, db_size, MAX, list(range(1, 101)))
    steps = epochs * len(groups)
    ckpt = f"scratchpad/ab_{name}/{name}_{steps}.pth"
    tcfg = f"scratchpad/ab_train_{name}.toml"
    ecfg = f"scratchpad/ab_eval_{name}.toml"
    open(tcfg, "w").write(TRAIN_TMPL.format(name=name, db=db, db_size=db_size, max=MAX, epochs=epochs, steps=steps))
    open(ecfg, "w").write(EVAL_TMPL.format(name=name, ckpt=ckpt))
    print(f"groups={len(groups)} total_steps={steps} ckpt={ckpt}", flush=True)

    t0 = time.time()
    tout = run([PY, "-u", "-m", "rwkv.train_rwkv", "--config", tcfg], f"scratchpad/ab_train_{name}.out")
    train_min = (time.time() - t0) / 60.0
    # parse the LAST 'loss_n per second' (training throughput, excludes validation)
    lns = re.findall(r"loss_n per second:\s*([\d.]+)", tout)
    sps = re.findall(r"Steps per second:\s*([\d.]+)", tout)
    thr = float(lns[-1]) if lns else float("nan")
    if not os.path.exists(ckpt):
        print(f"!! ckpt missing for {name}; tail:\n" + tout[-1500:], flush=True)
        summary.append((name, MAX, steps, train_min, thr, None, None)); continue

    # delete stale eval jsonls so get_result re-evals
    for f in [f"result/RWKV-ab-{name}.jsonl", f"result/RWKV-P-ab-{name}.jsonl"]:
        try: os.remove(f)
        except FileNotFoundError: pass
    eout = run([PY, "-u", "-m", "rwkv.get_result", "--config", ecfg], f"scratchpad/ab_eval_{name}.out")
    try:
        ah, im = by_user_mean(f"RWKV-ab-{name}", f"RWKV-P-ab-{name}")
    except Exception as e:
        print(f"!! eval parse failed for {name}: {e}\n" + eout[-1500:], flush=True)
        ah = im = None
    summary.append((name, MAX, steps, train_min, thr, ah, im))
    print(f">>> {name}: train_min={train_min:.1f} thr={thr:.0f} rev/s  ahead={ah} imm={im}", flush=True)

print("\n\n================ ROUTE A/B SUMMARY ================")
print(f"{'run':12} {'MAX':>7} {'steps':>6} {'train_min':>9} {'rev/s':>8} {'ahead':>9} {'imm':>9}")
print(f"{'base65k(ref)':12} {66000:>7} {1020:>6} {15.2:>9} {31287:>8} {0.329804:>9.5f} {0.296890:>9.5f}")
print(f"{'sc8k(ref)':12} {66000:>7} {960:>6} {17.3:>9} {27345:>8} {0.322033:>9.5f} {0.289628:>9.5f}")
for name, MAX, steps, tm, thr, ah, im in summary:
    a = f"{ah:9.5f}" if ah is not None else f"{'ERR':>9}"
    i = f"{im:9.5f}" if im is not None else f"{'ERR':>9}"
    print(f"{name:12} {MAX:>7} {steps:>6} {tm:>9.1f} {thr:>8.0f} {a} {i}")
print("\nNotes: sc8k(ref) imm 0.289628 is the run to confirm (a). rev/s is the speed metric (b).")
print("Augmentation noise ~0.0018 imm. Higher MAX = fewer steps = fewer updates at fixed 6 epochs.")
