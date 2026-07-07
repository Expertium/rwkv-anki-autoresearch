"""Greedy coordinate-descent HP tuner for the 5k phase -- FULL 5k: train users 1-5000
(train_db_5k_h1, MAX=110000), tune-eval on the HELD-OUT subset 5001-5200 (test_db_5k), the H=2/K=16
champion arch, QUANT-AWARE throughout (methodology a: WS + decay + eval all run with the fused
card/note fake-quant env). The 1500-proxy era is over (proxy proved unfaithful, see notes 2026-06-30).
PREREQ: build STEP3 finished + `python optimization/count_groups_5k.py` run once (writes
optimization/groups_5k.json = the real GROUPS_PER_EPOCH for the 2-epoch WS budget).

Stateless POLICY over an append-only journal (optimization/tuner_5k_log.jsonl). The agent calls `next`
to emit the next trial, detaches the generated self-recording .cmd, and repeats until DONE. Each .cmd
runs the FULL champion recipe: WS (2 ep) -> write_decay_setup -> decay (0.5 ep) -> write_eval_toml ->
eval (101-200) -> self-record. Mirrors scratchpad/run_h2k16.cmd.

Levers (coordinate order = high-impact/cheap first): peak_lr, warmup_steps, weight_decay, clip, decay_ratio.
WS epochs FIXED at 2; decay epochs = WS x decay_ratio, a TUNED lever with ratio in [1/10, 1/2.5] -> decay
0.2-0.8 epochs (Andrew 2026-07-01). Defaults = the H2K16 champion HPs (decay_ratio 0.25 -> 0.5 decay ep).

WILCOXON EARLY-PRUNING (Andrew 2026-07-02, methodology pt 9): every trial writes a per-step WS trace and,
when optimization/champion_5k.json exists (written by promote_champion_5k.py after the pre-tune champion
run), runs with RWKV_PRUNE_REF -> the trainer aborts (exit 42) iff BOTH modes are worse at p<1e-4 on the
growing 300n window. RWKV_PRUNE_MIN_STEP = 2x the TRIAL's warmup (a big-warmup trial is worse early by
construction; delaying the first check avoids false prunes). A pruned trial records its ESTIMATED logloss
(champ_final + cand@s - champ@s, from the .pruned.json marker) to the journal with "pruned": true --
coordinate descent proceeds on the estimate (an abysmal trial never wins a coordinate anyway).
Objective minimized = ahead + imm (fp32, by-user mean on 5001-5200). The strict accept gate is applied
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
# GROUPS_PER_EPOCH depends on the finished train_db_5k_h1: run `python optimization/count_groups_5k.py`
# once after build STEP3 -> it writes optimization/groups_5k.json, loaded here.
_GROUPS_JSON = f"{ROOT}/optimization/groups_5k.json"


def _load_groups_per_epoch():
    if not os.path.exists(_GROUPS_JSON):
        raise SystemExit("groups_5k.json missing -- run `python optimization/count_groups_5k.py` "
                         "after build STEP3 (train_db_5k_h1) completes")
    with open(_GROUPS_JSON) as fh:
        return int(json.load(fh)["groups_per_epoch"])


GROUPS_PER_EPOCH = None  # resolved lazily via ws_steps()
WS_EPOCHS = 2        # FIXED (5k budget)
# Decay epochs are now a TUNED lever (Andrew 2026-07-01): decay_ep = WS_EPOCHS * decay_ratio,
# ratio in [1/10, 1/2.5] -> decay in [0.2, 0.8] epochs. Default ratio 0.25 -> 0.5 decay ep (unchanged).
TRAIN_DB = "train_db_5k_h1"
USTART, UEND = 1, 5000
EVAL_USTART, EVAL_UEND = 5001, 5200   # tune-eval: held-out subset of 5001-10000
# Methodology (a): every 5k run trains AND evaluates quant-aware (fused card/note fake-quant).
# 2026-07-08: the sibling's FINAL locked recipe q72u (72 b/layer: joint-uv b10 WKV cb + m2b12 shift cb
# + 1-bit norms + int3 shift scope), with CODEBOOK LEARNING ON (Andrew, tonight's direction #1): both
# cbs init from the reference q72u catalogs and train per-run; the trial cmd repoints the env at each
# phase seam (WS-final exports feed the decay, decay-final exports feed the eval — the cb Parameters
# are process-globals initialized from these env files, NOT part of the ckpt; see resolve_run_cbs.py).
# RWKV_NO_JIT=1: the grafted q72u paths (fake_pq_shift, joint cb) are unverified under TorchScript;
# the sibling always ran NO_JIT. A/B JIT once at champion-run launch before removing.
QAT_ENV = ("set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4\n"
           "set RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt\n"
           "set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt\n"
           "set RWKV_QAT_PQ_LEARN=1\n"
           "set RWKV_QAT_SHIFT_PQ_LEARN=1\n"
           "set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3\n"
           "set RWKV_QAT_NORM_BITS=1\n"
           "set RWKV_QAT_FUSED=1\n"
           "set RWKV_NO_JIT=1\n")
NUM_FETCH = 10       # max-useful fetch (GPU saturates ~8-10 on a clean box); Andrew 2026-06-30 raised
# 5->10 as CPU frees up (FSRS postponed). Needs FETCH_AHEAD>=10 in train_rwkv.py to be usable (now 10).

# (param, grid). EPOCHS is NOT tuned (fixed budget). peak_lr around the champion 1e-3 (larger data may
# want more); warmup over the 6702-step WS; wd/clip robust levers.
SPACE = [
    ("peak_lr",      [7e-4, 1.0e-3, 1.4e-3, 2.0e-3]),
    ("warmup_steps", [200, 400, 800]),
    ("weight_decay", [0.0, 0.01, 0.05, 0.1]),
    ("clip",         [0.1, 0.25, 0.5]),
    ("decay_ratio",  [0.1, 0.2, 0.25, 0.4]),   # decay_ep = 2*ratio in [0.2, 0.8]; ratio in [1/10, 1/2.5]
]
DEFAULTS = {"peak_lr": 1e-3, "warmup_steps": 200, "weight_decay": 0.01, "clip": 0.25, "decay_ratio": 0.25}
PARAMS = [p for p, _ in SPACE]


def canon(cfg):
    return (round(float(cfg["peak_lr"]), 8), int(cfg["warmup_steps"]),
            round(float(cfg["weight_decay"]), 6), round(float(cfg["clip"]), 6),
            round(float(cfg.get("decay_ratio", 0.25)), 6))  # .get: pre-lever journal recs == implicit 0.25 (0.5 decay ep)


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
    global GROUPS_PER_EPOCH
    if GROUPS_PER_EPOCH is None:
        GROUPS_PER_EPOCH = _load_groups_per_epoch()
    return WS_EPOCHS * GROUPS_PER_EPOCH


def write_trial_files(name, param, cfg):
    folder = f"{TRIAL_DIR}/{name}"
    os.makedirs(folder, exist_ok=True)
    ws_ts = ws_steps()
    decay_ep = WS_EPOCHS * float(cfg["decay_ratio"])  # tuned lever (ratio in [1/10, 1/2.5])
    pval_str = f"{cfg[param]:g}" if param in cfg else "baseline"
    # Early-prune env (methodology pt 9): trace always on; prune only when a champion reference exists.
    # min_step = 2x this trial's warmup so warmup-heavy configs aren't false-pruned while still climbing.
    trace_rel = f"scratchpad/tuner5k/{name}/{name}_ws_trace.jsonl"
    champion_ref = f"{ROOT}/optimization/champion_5k.json"
    prune_lines = f"set RWKV_STEP_TRACE={trace_rel}\n"
    if os.path.exists(champion_ref):
        prune_lines += ("set RWKV_PRUNE_REF=optimization/champion_5k.json\n"
                        f"set RWKV_PRUNE_MIN_STEP={2 * int(cfg['warmup_steps'])}\n")
    # --- WS training toml (H2K16 proxy recipe; tuned TOML fields = peak_lr, warmup, epochs=2) ---
    ws_toml = f"""# HP5k trial {name}: param={param} -> {pval_str}.  Full config: {json.dumps(cfg)}
TRAIN_USERS_START = {USTART}
TRAIN_USERS_END = {UEND}
VALIDATE_USERS_START = 5001
VALIDATE_USERS_END = 5010

TRAIN_DATASET_LMDB_PATH = "{TRAIN_DB}"
TRAIN_DATASET_LMDB_SIZE = 400_000_000_000
VALIDATE_DATASET_LMDB_PATH = "F:/rwkv_lmdb/test_db_5k"
VALIDATE_DATASET_LMDB_SIZE = 250_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000

NUM_FETCH_PROCESSES = {NUM_FETCH}
MAX_TRAIN_GLOBAL_LEN = 110000

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
{QAT_ENV}{prune_lines}echo ===== TRIAL {name} (param={param}={pval_str}) cfg={json.dumps(cfg)} START %DATE% %TIME% ===== > "%LOG%"
echo === WS {WS_EPOCHS} epochs ({USTART}-{UEND}) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/{name}/{name}_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo === WILCOXON-PRUNED - recording estimated logloss %TIME% === >> "%LOG%"
  .venv\\Scripts\\python.exe optimization/hp_tuner_5k.py record-pruned {name} >> "%LOG%" 2>&1
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/resolve_run_cbs.py scratchpad/tuner5k/{name} {name}ws scratchpad/tuner5k/{name}/cb_wkv_ws.txt scratchpad/tuner5k/{name}/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/tuner5k/{name}/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/tuner5k/{name}/cb_shift_ws.txt
echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_decay_setup.py scratchpad/tuner5k/{name} {name}ws {name}d scratchpad/tuner5k/{name}/{name}_decay.toml {TRAIN_DB} {USTART} {UEND} {decay_ep:g} {cfg["peak_lr"]:g} >> "%LOG%" 2>&1
echo === DECAY {decay_ep:g} epoch (ratio {cfg["decay_ratio"]:g}) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/{name}/{name}_decay.toml >> "%LOG%" 2>&1
echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/resolve_run_cbs.py scratchpad/tuner5k/{name} {name}d scratchpad/tuner5k/{name}/cb_wkv_final.txt scratchpad/tuner5k/{name}/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/tuner5k/{name}/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/tuner5k/{name}/cb_shift_final.txt
del /Q result\\RWKV-{name}.jsonl result\\RWKV-P-{name}.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_eval_toml.py scratchpad/tuner5k/{name} {name}d scratchpad/tuner5k/{name}/{name}_eval.toml RWKV-{name} RWKV-P-{name} >> "%LOG%" 2>&1
echo === EVAL {EVAL_USTART}-{EVAL_UEND} (quant-aware) %TIME% === >> "%LOG%"
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


def cmd_record_pruned(name):
    """Record a Wilcoxon-pruned trial from its .pruned.json marker: journal gets the ESTIMATED
    logloss (champ_final + cand@s - champ@s) flagged "pruned": true, so descent proceeds."""
    side = json.load(open(f"{TRIAL_DIR}/{name}/{name}.json"))
    marker = json.load(open(f"{TRIAL_DIR}/{name}/{name}_ws_trace.jsonl.pruned.json"))
    rec = {"name": name, "param": side["param"], "config": side["config"],
           "ahead": round(float(marker["estimated_ahead"]), 6),
           "imm": round(float(marker["estimated_imm"]), 6),
           "pruned": True, "pruned_at_step": int(marker["pruned_at_step"]),
           "p_ahead": marker["p_ahead"], "p_imm": marker["p_imm"]}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED-PRUNED {name} @ step {rec['pruned_at_step']}: est ahead {rec['ahead']:.6f} "
          f"est imm {rec['imm']:.6f} (p_a {marker['p_ahead']:.2e}, p_i {marker['p_imm']:.2e})")


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
    print(f"{'name':30} {'param':14} {'ahead':>9} {'imm':>9} {'obj':>9}  note")
    for r in recs:
        note = f"PRUNED@{r['pruned_at_step']} (estimated)" if r.get("pruned") else ""
        print(f"{r['name']:30} {r['param']:14} {r['ahead']:9.6f} {r['imm']:9.6f} {obj(r):9.6f}  {note}")
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
    elif cmd == "record-pruned":
        cmd_record_pruned(sys.argv[2])
    elif cmd == "record-baseline":
        cmd_record_baseline(sys.argv[2], sys.argv[3])
    elif cmd == "status":
        cmd_status()
    elif cmd == "loop":
        cmd_loop()
    else:
        print("unknown command:", cmd)
