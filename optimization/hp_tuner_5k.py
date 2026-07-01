"""Greedy coordinate-descent HP tuner for the 5k phase -- run on the 1500-user PROXY
(train_db_sc8k_1500, users 1000-2499, eval 101-200) with the 5k compute-budget SHAPE
(2 WS epochs + 0.5 decay epochs) and the H=2/K=16 champion arch. The winning HPs transfer to the
full 1-5000 run (warmup may need rescaling to the 5k step count). Full-5k tuning would be ~4-5 days
per sweep; the proxy is ~85 min/trial.

Stateless POLICY over an append-only journal (optimization/tuner_5k_log.jsonl). The agent calls `next`
to emit the next trial, detaches the generated self-recording .cmd, and repeats until DONE. Each .cmd
runs the FULL champion recipe: WS (2 ep) -> write_decay_setup -> decay (0.5 ep) -> write_eval_toml ->
eval (101-200) -> self-record. Mirrors scratchpad/run_h2k16.cmd.

Levers (coordinate order = high-impact/cheap first): peak_lr, warmup_steps, weight_decay, clip.
WS epochs FIXED at 2, decay FIXED at 0.5 (Andrew's 5k budget). Defaults = the H2K16 champion HPs.
Objective minimized = ahead + imm (fp32, by-user mean on 101-200). The strict accept gate is applied
separately when declaring a champion. CLI matches hp_tuner.py: next / record <name> /
record-baseline <ahead> <imm> / status / loop.
"""
import json
import os
import subprocess
import sys

ROOT = "C:/Users/Andrew/rwkv-anki-autoresearch"
JOURNAL = f"{ROOT}/optimization/tuner_5k_log.jsonl"
TRIAL_DIR = f"{ROOT}/scratchpad/tuner5k"
# train_db_sc8k_1500 @ MAX=66000 -> get_groups gives 3351 groups (the H2K16 champion's 1 WS epoch = 3351 steps).
GROUPS_PER_EPOCH = 3351
WS_EPOCHS = 2        # FIXED (5k budget)
DECAY_EPOCHS = 0.5   # FIXED (5k budget)
TRAIN_DB = "train_db_sc8k_1500"
USTART, UEND = 1000, 2499
NUM_FETCH = 10       # max-useful fetch (GPU saturates ~8-10 on a clean box); Andrew 2026-06-30 raised
# 5->10 as CPU frees up (FSRS postponed). Needs FETCH_AHEAD>=10 in train_rwkv.py to be usable (now 10).

# (param, grid). EPOCHS is NOT tuned (fixed budget). peak_lr around the champion 1e-3 (larger data may
# want more); warmup over the 6702-step WS; wd/clip robust levers.
SPACE = [
    ("peak_lr",      [7e-4, 1.0e-3, 1.4e-3, 2.0e-3]),
    ("warmup_steps", [200, 400, 800]),
    ("weight_decay", [0.0, 0.01, 0.05, 0.1]),
    ("clip",         [0.1, 0.25, 0.5]),
]
DEFAULTS = {"peak_lr": 1e-3, "warmup_steps": 200, "weight_decay": 0.01, "clip": 0.25}
PARAMS = [p for p, _ in SPACE]


def canon(cfg):
    return (round(float(cfg["peak_lr"]), 8), int(cfg["warmup_steps"]),
            round(float(cfg["weight_decay"]), 6), round(float(cfg["clip"]), 6))


def obj(rec):
    return rec["ahead"] + rec["imm"]


def load_journal():
    recs = []
    if os.path.exists(JOURNAL):
        for line in open(JOURNAL):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def find(recs, cfg):
    k = canon(cfg)
    for r in recs:
        if canon(r["config"]) == k:
            return r
    return None


def compute(recs):
    """Replay coordinate descent. Returns ('need', cfg, param) or ('done', best_cfg)."""
    best = dict(DEFAULTS)
    if find(recs, DEFAULTS) is None:
        return ("need", dict(DEFAULTS), "baseline")
    for param, grid in SPACE:
        results = {}
        for v in grid:
            cfg = dict(best)
            cfg[param] = v
            r = find(recs, cfg)
            if r is None:
                return ("need", cfg, param)
            results[v] = obj(r)
        best[param] = min(results, key=lambda v: results[v])
    return ("done", best)


def trial_name(param, cfg):
    if param == "baseline":  # the champion-HP anchor, RUN as the first trial (2WS+0.5decay is a new budget)
        return "hp5k_baseline"
    v = cfg[param]
    vs = f"{v:g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"hp5k_{param}_{vs}"


def ws_steps():
    return WS_EPOCHS * GROUPS_PER_EPOCH  # 6702


def write_trial_files(name, param, cfg):
    folder = f"{TRIAL_DIR}/{name}"
    os.makedirs(folder, exist_ok=True)
    ws_ts = ws_steps()
    pval_str = f"{cfg[param]:g}" if param in cfg else "baseline"
    # --- WS training toml (H2K16 proxy recipe; tuned TOML fields = peak_lr, warmup, epochs=2) ---
    ws_toml = f"""# HP5k trial {name}: param={param} -> {pval_str}.  Full config: {json.dumps(cfg)}
TRAIN_USERS_START = {USTART}
TRAIN_USERS_END = {UEND}
VALIDATE_USERS_START = 101
VALIDATE_USERS_END = 110

TRAIN_DATASET_LMDB_PATH = "{TRAIN_DB}"
TRAIN_DATASET_LMDB_SIZE = 80_000_000_000
VALIDATE_DATASET_LMDB_PATH = "test_db"
VALIDATE_DATASET_LMDB_SIZE = 8_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 2_000_000_000

NUM_FETCH_PROCESSES = {NUM_FETCH}
MAX_TRAIN_GLOBAL_LEN = 66000

TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = {int(cfg["warmup_steps"])}
EPOCHS = {WS_EPOCHS}
VALIDATE_EVERY = 1000000
PEAK_LR = {cfg["peak_lr"]:g}

LOAD_MODEL = false
SAVE_MODEL_FOLDER = "scratchpad/tuner5k/{name}"
SAVE_MODEL_PREFIX = "{name}ws"
DEVICE = "cuda"
DTYPE = "bfloat16"

USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
"""
    with open(f"{folder}/{name}_ws.toml", "w") as f:
        f.write(ws_toml)

    # --- sidecar (config + param for `record`) ---
    with open(f"{folder}/{name}.json", "w") as f:
        json.dump({"name": name, "param": param, "config": cfg, "ws_steps": ws_ts}, f)

    # --- self-recording trial .cmd (detach this). The decay + eval tomls are written at RUNTIME by the
    #     helpers (the WS-final and decay-final step counts are data-dependent). H2K16 env at the top
    #     applies to WS, decay AND eval. Decay starts from this trial's peak_lr (passed to write_decay_setup). ---
    cmd = f"""@echo off
cd /d C:\\Users\\Andrew\\rwkv-anki-autoresearch
set LOG=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\tuner5k\\{name}.log
set OMP_NUM_THREADS={NUM_FETCH}
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\\Users\\Andrew\\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY={cfg["weight_decay"]:g}
set RWKV_CLIP={cfg["clip"]:g}
echo ===== TRIAL {name} (param={param}={pval_str}) cfg={json.dumps(cfg)} START %DATE% %TIME% ===== > "%LOG%"
echo === WS {WS_EPOCHS} epochs ({USTART}-{UEND}) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/{name}/{name}_ws.toml >> "%LOG%" 2>&1
echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_decay_setup.py scratchpad/tuner5k/{name} {name}ws {name}d scratchpad/tuner5k/{name}/{name}_decay.toml {TRAIN_DB} {USTART} {UEND} {DECAY_EPOCHS} {cfg["peak_lr"]:g} >> "%LOG%" 2>&1
echo === DECAY {DECAY_EPOCHS} epoch %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/{name}/{name}_decay.toml >> "%LOG%" 2>&1
del /Q result\\RWKV-{name}.jsonl result\\RWKV-P-{name}.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_eval_toml.py scratchpad/tuner5k/{name} {name}d scratchpad/tuner5k/{name}/{name}_eval.toml RWKV-{name} RWKV-P-{name} >> "%LOG%" 2>&1
echo === EVAL 101-200 %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.get_result --config scratchpad/tuner5k/{name}/{name}_eval.toml >> "%LOG%" 2>&1
echo === RECORD {name} %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe optimization/hp_tuner_5k.py record {name} >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
"""
    with open(f"{folder}/{name}.cmd", "w") as f:
        f.write(cmd)
    return ws_ts


def by_user_mean(path):
    tot, n = 0.0, 0
    for line in open(path):
        r = json.loads(line)
        tot += r["metrics"]["LogLoss"]
        n += 1
    return tot / n, n


def cmd_next():
    recs = load_journal()
    out = compute(recs)
    if out[0] == "done":
        best = out[1]
        print("DONE")
        print("BEST CONFIG:", json.dumps(best))
        r = find(recs, best)
        if r:
            print(f"  ahead {r['ahead']:.6f}  imm {r['imm']:.6f}  (objective {obj(r):.6f})")
        return
    _, cfg, param = out
    name = trial_name(param, cfg)
    ts = write_trial_files(name, param, cfg)
    pv = "(champion HPs)" if param == "baseline" else f"{cfg[param]:g}"
    print(f"NEXT {name}")
    print(f"  param={param}  value={pv}  full={json.dumps(cfg)}  ws_steps={ts}")
    print(f"  cmd=scratchpad/tuner5k/{name}/{name}.cmd")


def cmd_record(name):
    side = json.load(open(f"{TRIAL_DIR}/{name}/{name}.json"))
    ahead, na = by_user_mean(f"{ROOT}/result/RWKV-{name}.jsonl")
    imm, ni = by_user_mean(f"{ROOT}/result/RWKV-P-{name}.jsonl")
    rec = {"name": name, "param": side["param"], "config": side["config"],
           "ahead": round(ahead, 6), "imm": round(imm, 6), "users": na}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED {name}: ahead {ahead:.6f} imm {imm:.6f} (users {na}/{ni}) obj {ahead+imm:.6f}")


def cmd_record_baseline(ahead, imm):
    rec = {"name": "baseline", "param": "baseline", "config": dict(DEFAULTS),
           "ahead": round(float(ahead), 6), "imm": round(float(imm), 6)}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED baseline: ahead {ahead} imm {imm}")


def cmd_loop():
    """Self-driving coordinate descent: run every remaining trial (its WS->decay->eval->record .cmd)
    until DONE. Resumable -- replays the journal on restart, so a teardown continues from the next
    trial. Launch this DETACHED (survives Esc). Each trial .cmd self-records to the journal."""
    while True:
        recs = load_journal()
        out = compute(recs)
        if out[0] == "done":
            best = out[1]
            print("TUNER DONE. best:", json.dumps(best), flush=True)
            r = find(recs, best)
            base = next((x for x in recs if x["param"] == "baseline"), None)
            if r and base:
                print(f"  best vs baseline: ahead {base['ahead']-r['ahead']:+.6f}  "
                      f"imm {base['imm']-r['imm']:+.6f}  (obj {obj(base)-obj(r):+.6f})", flush=True)
            return
        _, cfg, param = out
        name = trial_name(param, cfg)
        write_trial_files(name, param, cfg)
        print(f"\n===== TRIAL {name}  param={param}  cfg={json.dumps(cfg)} =====", flush=True)
        cmd_path = f"{TRIAL_DIR}/{name}/{name}.cmd".replace("/", "\\")
        rc = subprocess.call(["cmd", "/c", cmd_path])
        if find(load_journal(), cfg) is None:
            print(f"ABORT: {name} did not record (rc={rc}). Check scratchpad/tuner5k/{name}.log. Stopping.",
                  flush=True)
            return


def cmd_status():
    recs = load_journal()
    print(f"{'name':30} {'param':14} {'ahead':>9} {'imm':>9} {'obj':>9}")
    for r in recs:
        print(f"{r['name']:30} {r['param']:14} {r['ahead']:9.6f} {r['imm']:9.6f} {obj(r):9.6f}")
    out = compute(recs)
    if out[0] == "done":
        best = out[1]
        print("\nCOORDINATE DESCENT COMPLETE. best:", json.dumps(best))
        r = find(recs, best)
        base = next((x for x in recs if x["param"] == "baseline"), None)
        if r and base:
            print(f"  vs baseline: ahead {base['ahead']-r['ahead']:+.6f}  imm {base['imm']-r['imm']:+.6f}")
    else:
        _, cfg, param = out
        if param == "baseline":
            print("\nNEXT: run the baseline trial (champion HPs @ 2WS+0.5decay) -- it runs first as hp5k_baseline")
        else:
            print(f"\nNEXT: probe param={param} value={cfg[param]:g}  ({trial_name(param, cfg)})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "next":
        cmd_next()
    elif cmd == "record":
        cmd_record(sys.argv[2])
    elif cmd == "record-baseline":
        cmd_record_baseline(sys.argv[2], sys.argv[3])
    elif cmd == "status":
        cmd_status()
    elif cmd == "loop":
        cmd_loop()
    else:
        print("unknown command:", cmd)
