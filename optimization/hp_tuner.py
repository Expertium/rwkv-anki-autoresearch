"""Greedy coordinate-descent (Hooke-Jeeves-style) hyperparameter tuner for the 100/100 workbench.

Stateless POLICY over an append-only journal (optimization/tuner_log.jsonl). The agent is the
orchestrator: it calls `next` to get the next trial, detaches the generated .cmd (train+eval on the
sc8k 8192-chunk db, MAX=66000, augmentation OFF), and the .cmd self-records via `record` at the end.
Repeat until `next` prints DONE.

Why coordinate descent (not CMA-ES / Bayesian): ~22 min/trial means a ~12-trial budget, far too small
for covariance estimation; greedy 1-D sweeps with a tiny grid extract most of the gain cheaply. The
previous param's WINNER is reused as the anchor for the next param (so #new-trials = sum(len(grid)-1)).

Objective minimized during tuning = ahead + imm (fp32, by-user mean on 101-200). The strict research
acceptance gate (BOTH modes better by >=0.0003 vs the champion, params/state caps) is applied SEPARATELY
when declaring a new champion -- tuning just finds the lowest-loss config to then gate.

Coordinate order = cheap-to-vary & high-impact first; EPOCHS last (it makes trials longer). peak_lr /
warmup_steps / epochs are TOML fields; weight_decay / clip are env-var overrides read by train_rwkv.py
(RWKV_WEIGHT_DECAY / RWKV_CLIP). All defaults equal the current champion recipe, so the champion's
aug-off run IS the baseline anchor (recorded as name='baseline').

CLI:
  python optimization/hp_tuner.py next            # emit next trial (writes tuner/<name>_{ws,eval}.toml + <name>.cmd + <name>.json); prints name or DONE
  python optimization/hp_tuner.py record <name>   # read result/RWKV[-P]-<name>.jsonl -> append {name,param,config,ahead,imm} to journal
  python optimization/hp_tuner.py record-baseline <ahead> <imm>   # seed the baseline (champion aug-off) row
  python optimization/hp_tuner.py status          # print journal + current best
"""
import json
import os
import subprocess
import sys

ROOT = "C:/Users/Andrew/rwkv-anki-autoresearch"
JOURNAL = f"{ROOT}/optimization/tuner_log.jsonl"
TRIAL_DIR = f"{ROOT}/scratchpad/tuner"
GROUPS_PER_EPOCH = 160  # sc8k db @ MAX=66000 -> get_groups gives 160 groups (verified: 6 ep -> 960 steps)

# (param, grid). Coordinate order = high-impact/cheap first; epochs LAST (longer trials).
SPACE = [
    ("peak_lr",      [3.5e-4, 5e-4, 7e-4, 1.0e-3, 1.4e-3]),
    ("warmup_steps", [100, 200, 400]),
    ("weight_decay", [0.0, 0.01, 0.05, 0.1]),
    ("clip",         [0.25, 0.5, 1.0]),
    ("epochs",       [6, 9, 12, 15]),
]
DEFAULTS = {"peak_lr": 7e-4, "warmup_steps": 200, "weight_decay": 0.01, "clip": 0.5, "epochs": 6}
TOML_FIELD = {"peak_lr": "PEAK_LR", "warmup_steps": "WARMUP_STEPS", "epochs": "EPOCHS"}
ENV_VAR = {"weight_decay": "RWKV_WEIGHT_DECAY", "clip": "RWKV_CLIP"}

PARAMS = [p for p, _ in SPACE]


def canon(cfg):
    return (round(float(cfg["peak_lr"]), 8), int(cfg["warmup_steps"]),
            round(float(cfg["weight_decay"]), 6), round(float(cfg["clip"]), 6), int(cfg["epochs"]))


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
    v = cfg[param]
    vs = f"{v:g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"hp_{param}_{vs}"


def total_steps(cfg):
    return int(cfg["epochs"]) * GROUPS_PER_EPOCH


def write_trial_files(name, param, cfg):
    os.makedirs(TRIAL_DIR, exist_ok=True)
    ts = total_steps(cfg)
    # --- training toml (sc8k recipe; tuned TOML fields overridden) ---
    train_toml = f"""# HP-tuner trial {name}: param={param} -> {cfg[param]}.  Full config: {json.dumps(cfg)}
TRAIN_USERS_START = 1
TRAIN_USERS_END = 100
VALIDATE_USERS_START = 101
VALIDATE_USERS_END = 110

TRAIN_DATASET_LMDB_PATH = "train_db_sc8k"
TRAIN_DATASET_LMDB_SIZE = 4_000_000_000
VALIDATE_DATASET_LMDB_PATH = "test_db"
VALIDATE_DATASET_LMDB_SIZE = 8_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 2_000_000_000

NUM_FETCH_PROCESSES = 7
MAX_TRAIN_GLOBAL_LEN = 66000

TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = {int(cfg["warmup_steps"])}
EPOCHS = {int(cfg["epochs"])}
VALIDATE_EVERY = 1000000
PEAK_LR = {cfg["peak_lr"]:g}

LOAD_MODEL = false
SAVE_MODEL_FOLDER = "scratchpad/tuner/{name}"
SAVE_MODEL_PREFIX = "{name}"
DEVICE = "cuda"
DTYPE = "bfloat16"

USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
"""
    with open(f"{TRIAL_DIR}/{name}_ws.toml", "w") as f:
        f.write(train_toml)

    # --- eval toml (101-200) ---
    eval_toml = f"""# HP-tuner trial {name} eval on 101-200 (champion arch).
FILE_AHEAD = "RWKV-{name}"
FILE_IMM = "RWKV-P-{name}"

MODEL_PATH = "scratchpad/tuner/{name}/{name}_{ts}.pth"
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
    with open(f"{TRIAL_DIR}/{name}_eval.toml", "w") as f:
        f.write(eval_toml)

    # --- sidecar (config + param for `record`) ---
    with open(f"{TRIAL_DIR}/{name}.json", "w") as f:
        json.dump({"name": name, "param": param, "config": cfg, "total_steps": ts}, f)

    # --- self-recording trial .cmd (detach this) ---
    env_lines = "".join(
        f"set {ENV_VAR[p]}={cfg[p]:g}\n" for p in ENV_VAR if p in cfg
    )
    cmd = f"""@echo off
cd /d C:\\Users\\Andrew\\rwkv-anki-autoresearch
set LOG=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\tuner\\{name}.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\\Users\\Andrew\\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
{env_lines}echo ===== TRIAL {name} (param={param}={cfg[param]:g}) START %DATE% %TIME% ===== > "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner/{name}_ws.toml >> "%LOG%" 2>&1
del /Q result\\RWKV-{name}.jsonl result\\RWKV-P-{name}.jsonl 2>nul
echo ===== EVAL {name} %TIME% ===== >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.get_result --config scratchpad/tuner/{name}_eval.toml >> "%LOG%" 2>&1
echo ===== RECORD {name} %TIME% ===== >> "%LOG%"
.venv\\Scripts\\python.exe optimization/hp_tuner.py record {name} >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
"""
    with open(f"{TRIAL_DIR}/{name}.cmd", "w") as f:
        f.write(cmd)
    return ts


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
    if param == "baseline":
        print("NEED_BASELINE: record the champion aug-off run first via 'record-baseline <ahead> <imm>'")
        print("BASELINE CONFIG:", json.dumps(cfg))
        return
    name = trial_name(param, cfg)
    ts = write_trial_files(name, param, cfg)
    print(f"NEXT {name}")
    print(f"  param={param}  value={cfg[param]:g}  full={json.dumps(cfg)}  total_steps={ts}")
    print(f"  cmd=scratchpad/tuner/{name}.cmd")


def cmd_record(name):
    side = json.load(open(f"{TRIAL_DIR}/{name}.json"))
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


PY = ".venv/Scripts/python.exe"


def trial_env(cfg):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "7"
    env["RWKV_DETERMINISTIC"] = "1"
    env["RWKV_AUGMENT_SEED"] = "1234"
    env["RWKV_WEIGHT_DECAY"] = f"{cfg['weight_decay']:g}"
    env["RWKV_CLIP"] = f"{cfg['clip']:g}"
    return env


def cmd_loop():
    """Self-driving coordinate descent: run every remaining trial (train -> eval -> record) until DONE.
    Resumable -- replays the journal on restart, so a teardown just continues from the next trial.
    Launch this DETACHED (survives Esc); it self-records each trial to the journal."""
    while True:
        recs = load_journal()
        out = compute(recs)
        if out[0] == "done":
            print("TUNER DONE. best:", json.dumps(out[1]), flush=True)
            r = find(recs, out[1])
            base = next((x for x in recs if x["param"] == "baseline"), None)
            if r and base:
                print(f"  best vs baseline: ahead {base['ahead']-r['ahead']:+.6f}  "
                      f"imm {base['imm']-r['imm']:+.6f}  (obj {obj(base)-obj(r):+.6f})", flush=True)
            return
        _, cfg, param = out
        if param == "baseline":
            print("NEED BASELINE: run 'record-baseline <ahead> <imm>' first.", flush=True)
            return
        name = trial_name(param, cfg)
        ts = write_trial_files(name, param, cfg)
        env = trial_env(cfg)
        print(f"\n===== TRIAL {name}  param={param}={cfg[param]:g}  total_steps={ts} =====", flush=True)
        # train
        rc = subprocess.call([PY, "-u", "-m", "rwkv.train_rwkv", "--config",
                              f"scratchpad/tuner/{name}_ws.toml"], cwd=ROOT, env=env)
        ckpt = f"{ROOT}/scratchpad/tuner/{name}/{name}_{ts}.pth"
        if not os.path.exists(ckpt):
            print(f"ABORT: training produced no checkpoint {ckpt} (rc={rc}). Stopping loop.", flush=True)
            return
        # eval (delete stale jsonls so get_result re-evals all users)
        for f in (f"{ROOT}/result/RWKV-{name}.jsonl", f"{ROOT}/result/RWKV-P-{name}.jsonl"):
            if os.path.exists(f):
                os.remove(f)
        rc = subprocess.call([PY, "-u", "-m", "rwkv.get_result", "--config",
                              f"scratchpad/tuner/{name}_eval.toml"], cwd=ROOT, env=env)
        if not os.path.exists(f"{ROOT}/result/RWKV-P-{name}.jsonl"):
            print(f"ABORT: eval produced no result for {name} (rc={rc}). Stopping loop.", flush=True)
            return
        cmd_record(name)


def cmd_status():
    recs = load_journal()
    print(f"{'name':28} {'param':14} {'ahead':>9} {'imm':>9} {'obj':>9}")
    for r in recs:
        print(f"{r['name']:28} {r['param']:14} {r['ahead']:9.6f} {r['imm']:9.6f} {obj(r):9.6f}")
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
            print("\nNEXT: record baseline (champion aug-off) first")
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
