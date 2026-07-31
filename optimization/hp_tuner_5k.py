"""Greedy coordinate-descent HP tuner for the MAX=65536 era -- REBUILT 2026-07-30.

WHY IT WAS REBUILT: the previous version targeted the d=32 H=2/K=16 arch at MAX=110000, ran
QUANT-AWARE throughout, used a 2-epoch WS budget and evaluated on users 101-200. Every one of
those is now wrong. The trunk is the d=80 A18 lineage (iter 31/32), training is PLAIN (QAT has
been parked since iter 14), WS is FIXED at 1 epoch (Andrew 2026-07-09) and the gate metric is
the RECTIFIED logloss (Andrew 2026-07-27).

WHAT IT IS FOR: the 2026-07-30 speedup phase adopted MAX_TRAIN_GLOBAL_LEN = 65536, which is
1.61x faster but drops the group count 22,346 -> 10,935 -- i.e. HALF the optimizer steps per
epoch at unchanged LR. That cost -0.000264 ahead / -0.000307 imm vs iter 31 rectified (both
modes the same direction = a real systematic loss, not seed noise). Andrew accepted the speed
and directed this tuner to recover the accuracy. Hence the lever order below is NOT the old one:
the batch doubled, so the LEARNING RATES come first.

RECIPE PER TRIAL (mirrors scratchpad/maxval/run_maxval.cmd, the validated MAX=65536 run):
  40-step sanity (VRAM + proves the speed flags actually engaged)
  -> WS 1 epoch, train users 1-5000, train_db_5k_h1, MAX=65536, NUM_FETCH_PROCESSES=2
  -> write_decay_setup -> cosine decay (WS x decay_ratio epochs)
  -> write_eval_toml -> RECTIFIED eval (RWKV_EVAL_PAVA=1) on the tune-eval subset 5001-6000
  -> self-record to the journal.
Cost ~4.3 h/trial (WS 2h37 + decay ~40 min + eval ~1 h; the subset is 51.0M of the VAL half's
128.8M reviews).

THE BASELINE IS FREE: the `maxval` run IS the default config, and its rectified jsonls already
cover 5001-7500. Restricted to 5001-6000 it scores ahead 0.299250 / imm 0.266335, seeded into
the journal as the baseline row -- so trial 1 is a real probe, not an anchor re-run.

TUNE-EVAL SUBSET: 5001-6000 (1000 users), the post-champ5k_t1 remedy -- the old 200-user subset
could not resolve sub-0.001 effects and inverted at n=5000. Sanity check on this subset: it
ranks maxval vs iter 31 the same way the full VAL half does (subset +0.000113/+0.000309 vs full
+0.000264/+0.000306), so it is a usable proxy. Any sub-0.001 winner STILL needs confirming on
the full VAL half (5001-7500) before it becomes the recipe.

VALIDATION-BASED EARLY PRUNING is on, against optimization/tuner65k_vprune_ref.json (built from
maxval's own val trajectory + its 5001-6000 finals -- a matched reference on this exact trunk
and batch size). The trainer aborts (exit 42) iff BOTH modes' val loss exceed the reference by
>= 0.004 ahead AND 0.006 imm at 2 consecutive checkpoints. min_step = max(1000, 2 x the trial's
warmup) so a long-warmup trial is not killed for being slow by construction. This is the
sign-correct rule for regularization levers (the train-loss rule is not -- see the
decay_ratio_0p1 false-kill audit). It matters most for the LR grid, where a 2.8x LR probe can
diverge and would otherwise burn 4.3 h.

Objective minimized = ahead + imm (rectified, by-user mean on 5001-6000).
CLI: next / record <name> / record-pruned <name> / record-baseline <ahead> <imm> / status / loop.
"""
import json
import os
import subprocess
import sys

ROOT = "C:/Users/Andrew/rwkv-anki-autoresearch"
JOURNAL = f"{ROOT}/optimization/tuner_5k_log.jsonl"
TRIAL_DIR = f"{ROOT}/scratchpad/tuner65k"
VPRUNE_REF = f"{ROOT}/optimization/tuner65k_vprune_ref.json"

# Verified from the maxval run: its WS final checkpoint is mvws_10935.pth, i.e. 1 epoch of
# train_db_5k_h1 (users 1-5000) at MAX=65536 = 10,935 groups. (The old optimization/groups_5k.json
# says 6554 -- that is MAX=110000 and is stale; it is not read any more.)
GROUPS_PER_EPOCH = 10935
WS_EPOCHS = 1                      # FIXED (Andrew 2026-07-09, the champ5k_b1 budget A/B)
TRAIN_DB = "train_db_5k_h1"
USTART, UEND = 1, 5000
EVAL_USTART, EVAL_UEND = 5001, 6000
NUM_FETCH = 2                      # adopted 2026-07-30: halves the 3.8 GB/h RAM climb, fetch is
                                   # 2.3 ms of a ~1,450 ms step so it costs nothing
MAX_LEN = 65536                    # the adopted batch dim
VALIDATE_EVERY = 1000              # MUST match the vprune ref's cadence (pairing is by exact step)

# The iter-31/A18 trunk. Every run on this lineage sets all of these; a missing flag silently
# trains a different model, so they live in one string used verbatim by every trial.
TRUNK_ENV = (
    "set RWKV_DETERMINISTIC=1\n"
    "set RWKV_AUGMENT_SEED=1234\n"
    "set RWKV_EMPTY_CACHE_EVERY=1\n"
    "set RWKV_EMPTY_CACHE_WINDOW=0\n"
    "set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py\n"
    "set RWKV_GRU_HEAD=3\n"
    "set RWKV_PAVA_LAMBDA=0.1\n"
    "set RWKV_PROBE_DENSITY=0.08\n"
    "set RWKV_PROBE_DUR=0.0\n"
    "set RWKV_MUON=1\n"
    "set RWKV_MUON_MOMENTUM=0.95\n"
    "set RWKV_NO_AHEAD_RESIDUAL=1\n"
    "set RWKV_STRIP_L0_VLORA=1\n"
    "set RWKV_ZERO_FEATURES=22\n"
    "set RWKV_STATE_CLAMP_TAU=300\n"
    "set RWKV_STATE_CLAMP_WINDOW=32768\n"
    "set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
    "deck_id:1,deck_id:2,card_id:1\n"
)
# The speed stack adopted 2026-07-30 (1.68x, validated accuracy-neutral). Defaults are OFF in
# code, so these must be set explicitly. Cleared before eval, exactly as run_maxval.cmd does.
SPEED_ENV = ("set RWKV_MUON_BATCHED=1\n"
             "set RWKV_NO_JIT=1\n"
             "set RWKV_QAT_COMPILE=1\n")

BASE_PEAK_LR = 1e-3      # the AdamW group's base LR (57,412 params)
BASE_MUON_LR = 0.02      # the Muon groups' base LR (500,800 matrix params -- the bulk of the model)

# (param, grid). ORDER IS THE COORDINATE ORDER and it is deliberate: MAX doubled the batch, so
# the learning rates are the levers with a mechanistic reason to have moved. Everything after
# them is the usual robustness sweep.
SPACE = [
    # 200 steps was 0.9% of a 22,346-step epoch and is now 1.8% of a 10,935-step one. Bigger
    # batches usually want proportionally MORE warmup, not less; upstream used 20,000.
    # RESOLVED 2026-07-31: 400 wins, interior optimum (ahead is an inverted U), no extension needed.
    ("warmup_steps",  [200, 400, 800]),
    # Muon carries ~90% of the parameters, so re-balance its share against AdamW's at fixed overall
    # scale. muon_lr = BASE_MUON_LR * lr_mult * muon_lr_mult.
    # ★ 0.5 was the phase's big win (+0.000601/+0.000371 incremental, first trial to clear the bar
    # in both modes), and it sits on the LOW EDGE -- hence 0.25, added 2026-07-31 on Andrew's call
    # to probe it now rather than after the remaining coordinates.
    # ⚠ 2.0 was DROPPED, not silently skipped. It had been generated and ran ~25 min before being
    # stopped. Two independent results predict it is worse: the lr_mult coordinate is a 4-point
    # MONOTONIC dose-response in which more LR is always worse, and on this very lever 0.5 beat 1.0
    # by a wide margin. Spending 4.2 h to confirm that was worse value than 0.25 and lr_mult 0.7.
    # Add it back here if the shape is ever in doubt (warmup turned out non-monotonic, so shapes
    # are not always safe to assume).
    # ★★ 0.125 added 2026-08-01 -- SECOND consecutive edge win, and the two modes disagree about
    # whether it is exhausted. ahead is decelerating (1.0->0.5 gave +0.000601, 0.5->0.25 only
    # +0.000181) but imm is NOT (+0.000371 then +0.000411). imm was the mode still short of the
    # bar, so the lever that is still paying on it is worth one more probe.
    # ⚠ RESEARCH FLAG, not a tuning detail: RWKV_MUON_LR=0.02 was tuned at MAX=32768 and iter 29
    # accepted Muon on the strength of its imm gain. If 0.125 (muon_lr 0.0025, 8x below default)
    # keeps winning, the honest question becomes whether Muon still earns its place at THIS batch
    # size -- an iter-29-level question, not a coordinate. Raise it with Andrew; do not answer it
    # by letting the grid slide toward zero.
    ("muon_lr_mult",  [1.0, 0.5, 0.25, 0.125]),
    # Robustness levers. wd kept winning grid edges in the d=32 era (0.1, then 0.2), but that was
    # a different arch and 4x the params -- start from the champion 0.01 and probe upward.
    ("weight_decay",  [0.01, 0.05, 0.1]),
    ("clip",          [0.25, 0.5]),
    ("decay_ratio",   [0.25, 0.4]),
    # ★ lr_mult MOVED TO LAST and its grid REPLACED, 2026-07-31.
    # It originally ran FIRST with [1.0, 1.41, 2.0, 2.8] -- upward only, because the design
    # anchored on "the batch doubled, so the LR should rise". That heuristic was wrong in both
    # directions: raising the LR hurt monotonically across all four points, and the phase's biggest
    # win came from LOWERING an LR (muon_lr_mult 0.5). The original grid could not have found that,
    # since it contained no downward probe at all.
    # So the live question is the one it never asked: at the tuned config, does lowering the JOINT
    # LR help? Hence [1.0, 0.7], evaluated LAST so it sees the winning warmup/muon/wd/clip/decay.
    # The upward points are NOT re-run: that result is settled, monotonic over 4 points, and
    # recorded in optimization/TRAINING_SPEED.md. Their journal rows are untouched.
    # ⚠ Reordering is SAFE for the replay: coordinate descent evaluates each coordinate at the
    # incumbent of the ones BEFORE it, and every recorded row already has lr_mult=1.0 (the
    # default), so warmup/muon rows still match their configs exactly. Verified before relaunch.
    ("lr_mult",       [1.0, 0.7]),
]
DEFAULTS = {"lr_mult": 1.0, "warmup_steps": 200, "muon_lr_mult": 1.0,
            "weight_decay": 0.01, "clip": 0.25, "decay_ratio": 0.25}
PARAMS = [p for p, _ in SPACE]


def peak_lr(cfg):
    return BASE_PEAK_LR * float(cfg["lr_mult"])


def muon_lr(cfg):
    return BASE_MUON_LR * float(cfg["lr_mult"]) * float(cfg["muon_lr_mult"])


def canon(cfg):
    # .get with the DEFAULT for every later-added lever: journal rows written before a lever
    # existed ran with the (env-unset ==) default value, so they canon onto the same point.
    return tuple(round(float(cfg.get(p, DEFAULTS[p])), 8) for p in PARAMS)


# THE BAR: "recover what MAX=65536 cost" == reach iter 31's numbers ON THIS SAME 1000-user
# subset. Computed 2026-07-30 by restricting result/RWKV{,-P}-iter31_algo_rect.jsonl to
# 5001-6000. Note the two modes are NOT equidistant from the baseline (0.299250/0.266335):
# MAX=65536 hurt imm ~2.7x more than ahead here, so a trial that recovers ahead alone has
# not done the job.
BAR = {"ahead": 0.299137, "imm": 0.266026}


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
    if param == "baseline":
        return "t65_baseline"
    v = cfg[param]
    vs = f"{v:g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"t65_{param}_{vs}"


def ws_steps():
    return WS_EPOCHS * GROUPS_PER_EPOCH


def write_trial_files(name, param, cfg):
    folder = f"{TRIAL_DIR}/{name}"
    os.makedirs(folder, exist_ok=True)
    ws_ts = ws_steps()
    decay_ep = WS_EPOCHS * float(cfg["decay_ratio"])
    plr, mlr = peak_lr(cfg), muon_lr(cfg)
    pval_str = f"{cfg[param]:g}" if param in cfg else "baseline"

    # Stale-result hygiene, done HERE rather than in the .cmd. The .cmd retries a failed eval
    # WITHOUT deleting (eval_sharded skips users it already banked, which is what makes the
    # giant-user OOM recoverable -- the 2026-07-30 big-eval ops rule). Deleting at generation
    # time keeps that property while still guaranteeing a regenerated trial starts clean.
    for f in (f"{ROOT}/result/RWKV-{name}.jsonl", f"{ROOT}/result/RWKV-P-{name}.jsonl",
              f"{ROOT}/result/RWKV-{name}-s0.jsonl", f"{ROOT}/result/RWKV-P-{name}-s0.jsonl",
              f"{folder}/{name}_ws_trace.jsonl", f"{folder}/{name}_ws_trace.jsonl.val.jsonl",
              f"{folder}/{name}_ws_trace.jsonl.pruned.json"):
        if os.path.exists(f):
            os.remove(f)

    # vprune min_step: never kill a trial before 2x its own warmup (a long-warmup trial is worse
    # early BY CONSTRUCTION), and never before the documented floor of 1000.
    vprune_min = max(1000, 2 * int(cfg["warmup_steps"]))

    ws_toml = f"""# HP tuner (MAX=65536 era) trial {name}: param={param} -> {pval_str}
# Full config: {json.dumps(cfg)}  ->  PEAK_LR {plr:g}, RWKV_MUON_LR {mlr:g}
# Recipe = scratchpad/maxval/run_maxval.cmd (the validated MAX=65536 run) with the HPs swapped.
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
MAX_TRAIN_GLOBAL_LEN = {MAX_LEN}

TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = {int(cfg["warmup_steps"])}
EPOCHS = {WS_EPOCHS}
VALIDATE_EVERY = {VALIDATE_EVERY}
PEAK_LR = {plr:g}

LOAD_MODEL = false
SAVE_MODEL_FOLDER = "scratchpad/tuner65k/{name}"
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

    with open(f"{folder}/{name}.json", "w") as f:
        json.dump({"name": name, "param": param, "config": cfg, "ws_steps": ws_ts,
                   "peak_lr": plr, "muon_lr": mlr}, f)

    cmd = f"""@echo off
REM Auto-generated by optimization/hp_tuner_5k.py -- do NOT edit while running (cmd.exe re-reads
REM a running .cmd at a saved byte offset).
cd /d C:\\Users\\Andrew\\rwkv-anki-autoresearch
set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\tuner65k\\{name}
set LOG=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\tuner65k\\{name}.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\\Users\\Andrew\\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
{TRUNK_ENV}REM ---- this trial's HPs ----
set RWKV_MUON_LR={mlr:g}
set RWKV_WEIGHT_DECAY={cfg["weight_decay"]:g}
set RWKV_CLIP={cfg["clip"]:g}
REM ---- the adopted speed stack (cleared again before eval) ----
{SPEED_ENV}
echo ===== TRIAL {name} (param={param}={pval_str}) cfg={json.dumps(cfg)} peak_lr={plr:g} muon_lr={mlr:g} START %DATE% %TIME% ===== > "%LOG%"

echo === SANITY 40 steps (VRAM + speed-flag proof) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner65k/{name}/{name}_ws.toml > "%DIR%\\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
set RWKV_MAX_STEPS=
REM An env typo that silently disables a speed flag would cost ~2 extra hours PER TRIAL, so prove
REM both engaged before committing to the run.
findstr /C:"BATCHED Newton-Schulz" "%DIR%\\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MUONNOTBATCHED %DATE% %TIME% >> "%LOG%"
  exit /b 16
)
findstr /C:"[compile] torch.compile" "%DIR%\\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOCOMPILE %DATE% %TIME% >> "%LOG%"
  exit /b 17
)
echo SANITY OK %TIME% >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/tuner65k/{name}/{name}_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/tuner65k_vprune_ref.json
set RWKV_VPRUNE_MIN_STEP={vprune_min}
echo === WS {WS_EPOCHS} epoch ({USTART}-{UEND}, MAX={MAX_LEN} -^> ~{ws_ts} steps) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner65k/{name}/{name}_ws.toml > "%DIR%\\ws_%STAMP%.log" 2>&1
if %ERRORLEVEL%==42 (
  echo === VAL-PRUNED - recording estimated logloss %TIME% === >> "%LOG%"
  .venv\\Scripts\\python.exe optimization/hp_tuner_5k.py record-pruned {name} >> "%LOG%" 2>&1
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
REM A crashed WS must NOT cascade into decay/eval -- write_decay_setup takes the LATEST ckpt, so
REM a half-trained one would be silently decayed and evaluated as if it were the real trial.
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=
set RWKV_VPRUNE_REF=

echo === DECAY SETUP ({decay_ep:g} ep, ratio {cfg["decay_ratio"]:g}) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_decay_setup.py scratchpad/tuner65k/{name} {name}ws {name}d scratchpad/tuner65k/{name}/{name}_decay.toml {TRAIN_DB} {USTART} {UEND} {decay_ep:g} {plr:g} {MAX_LEN} > "%DIR%\\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner65k/{name}/{name}_decay.toml > "%DIR%\\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo DECAY OK %TIME% >> "%LOG%"

echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_eval_toml.py scratchpad/tuner65k/{name} {name}d scratchpad/tuner65k/{name}/{name}_eval.toml RWKV-{name} RWKV-P-{name} {EVAL_USTART} {EVAL_UEND} > "%DIR%\\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
REM Eval runs WITHOUT the training speed flags, exactly as run_maxval.cmd does.
set RWKV_MUON_BATCHED=
set RWKV_QAT_COMPILE=
set RWKV_NO_JIT=
set RWKV_EVAL_PAVA=1
REM Users 5002/5905/5995 (266k-367k reviews) OOM the 12 GB card iff the desktop is holding
REM several GB of VRAM. eval_sharded SKIPS users already banked, so a retry costs only the
REM remainder -- hence three attempts and NO del between them (the 2026-07-30 ops rule).
echo === EVAL {EVAL_USTART}-{EVAL_UEND} RECTIFIED, attempt 1 %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/{name}/{name}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\\Scripts\\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/{name}/{name}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\\Scripts\\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/{name}/{name}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\\eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 8
)
echo EVAL OK %TIME% >> "%LOG%"

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
    pv = "(baseline HPs)" if param == "baseline" else f"{cfg[param]:g}"
    print(f"NEXT {name}")
    print(f"  param={param}  value={pv}  full={json.dumps(cfg)}")
    print(f"  peak_lr={peak_lr(cfg):g}  muon_lr={muon_lr(cfg):g}  ws_steps={ts}")
    print(f"  cmd=scratchpad/tuner65k/{name}/{name}.cmd")


def cmd_record(name):
    side = json.load(open(f"{TRIAL_DIR}/{name}/{name}.json"))
    ahead, na = by_user_mean(f"{ROOT}/result/RWKV-{name}.jsonl")
    imm, ni = by_user_mean(f"{ROOT}/result/RWKV-P-{name}.jsonl")
    rec = {"name": name, "param": side["param"], "config": side["config"],
           "ahead": round(ahead, 6), "imm": round(imm, 6), "users": na,
           "peak_lr": side["peak_lr"], "muon_lr": side["muon_lr"]}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED {name}: ahead {ahead:.6f} imm {imm:.6f} (users {na}/{ni}) obj {ahead+imm:.6f}")


def fit_vprune_alpha(recs):
    """Calibrate the val-delta -> final-tune-eval-delta shrinkage slope (per mode), from trials
    that COMPLETED honestly and carry a val sidecar: x = (trial val - reference val) at each WS
    checkpoint >= 1000, y = (trial tune-eval - baseline tune-eval). Early val gaps compress as
    training converges AND val (review-pooled, 10 users) is a different scale from the recorded
    metric (by-user mean, 1000 users) -- a single through-origin slope absorbs both. Caveat: pairs
    within a trial share one y (effective n = #trials, not #pairs), and completed trials only
    populate small |x|, so kill-scale (>=0.004) estimates are linear extrapolation -- hence the
    clamp. Returns (alpha_ahead, alpha_imm, n_pairs, n_trials); alpha=1.0 fallback."""
    if not os.path.exists(VPRUNE_REF):
        return 1.0, 1.0, 0, 0
    champ = json.load(open(VPRUNE_REF))
    if "val_step" not in champ:
        return 1.0, 1.0, 0, 0
    cvals = {int(s): (a, i) for s, a, i in zip(champ["val_step"], champ["val_ahead"], champ["val_imm"])}
    base = next((r for r in recs if r["param"] == "baseline"), None)
    if base is None:
        return 1.0, 1.0, 0, 0
    xa, ya, xi, yi, n_trials = [], [], [], [], 0
    for r in recs:
        if r.get("pruned") or r["param"] == "baseline":
            continue
        sidecar = f"{TRIAL_DIR}/{r['name']}/{r['name']}_ws_trace.jsonl.val.jsonl"
        side_path = f"{TRIAL_DIR}/{r['name']}/{r['name']}.json"
        if not os.path.exists(sidecar) or not os.path.exists(side_path):
            continue
        # re-probes at a new base REUSE the trial name (and overwrite the dir): only pair a
        # sidecar with the journal row whose config matches the dir's current side json
        side = json.load(open(side_path))
        if canon(side["config"]) != canon(r["config"]):
            continue
        pts = 0
        for line in open(sidecar):
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            s = int(v["step"])
            if s < 1000 or s not in cvals:
                continue
            xa.append(v["val_ahead"] - cvals[s][0]); ya.append(r["ahead"] - base["ahead"])
            xi.append(v["val_imm"] - cvals[s][1]);   yi.append(r["imm"] - base["imm"])
            pts += 1
        n_trials += 1 if pts else 0

    def slope(x, y):
        sxx = sum(v * v for v in x)
        if n_trials < 3 or sxx < 1e-10:
            return 1.0
        a = sum(u * v for u, v in zip(x, y)) / sxx
        if a <= 0:  # noise/anti-correlation = no information -> naive slope, not a fake floor
            return 1.0
        return min(max(a, 0.25), 1.5)
    return slope(xa, ya), slope(xi, yi), len(xa), n_trials


def cmd_record_pruned(name):
    """Record a pruned trial from its .pruned.json marker: the journal gets the ESTIMATED logloss
    flagged "pruned": true, so descent proceeds (an abysmal trial never wins a coordinate anyway).
    Estimate = baseline_tune_eval + alpha * mean(strike-window val deltas): the window mean cuts
    single-checkpoint noise, the fitted alpha corrects early-gap compression and the val ->
    tune-eval scale, and anchoring on the BASELINE JOURNAL ROW keeps pruned rows on the same
    1000-user scale as honest rows."""
    side = json.load(open(f"{TRIAL_DIR}/{name}/{name}.json"))
    marker = json.load(open(f"{TRIAL_DIR}/{name}/{name}_ws_trace.jsonl.pruned.json"))
    rec = {"name": name, "param": side["param"], "config": side["config"],
           "ahead": round(float(marker["estimated_ahead"]), 6),
           "imm": round(float(marker["estimated_imm"]), 6),
           "pruned": True, "pruned_at_step": int(marker["pruned_at_step"]),
           "rule": marker.get("rule", "val"),
           "peak_lr": side["peak_lr"], "muon_lr": side["muon_lr"]}
    detail = ""
    if "val_delta_ahead" in marker:
        rec["val_delta_ahead"] = marker["val_delta_ahead"]
        rec["val_delta_imm"] = marker["val_delta_imm"]
        detail = f"(val d_a {marker['val_delta_ahead']:+.4f}, d_i {marker['val_delta_imm']:+.4f})"
        recs = load_journal()
        base = next((r for r in recs if r["param"] == "baseline"), None)
        if base is not None:
            window = marker.get("window") or [[marker["pruned_at_step"],
                                               marker["val_delta_ahead"], marker["val_delta_imm"]]]
            mda = sum(w[1] for w in window) / len(window)
            mdi = sum(w[2] for w in window) / len(window)
            a_a, a_i, n_pairs, n_trials = fit_vprune_alpha(recs)
            rec["ahead"] = round(base["ahead"] + a_a * mda, 6)
            rec["imm"] = round(base["imm"] + a_i * mdi, 6)
            rec["est_alpha"] = [round(a_a, 4), round(a_i, 4)]
            rec["est_window_mean"] = [round(mda, 6), round(mdi, 6)]
            rec["est_naive"] = [round(float(marker["estimated_ahead"]), 6),
                                round(float(marker["estimated_imm"]), 6)]
            detail += (f" est = baseline + alpha*window_mean, alpha ({a_a:.2f},{a_i:.2f}) "
                       f"from {n_trials} trials/{n_pairs} pairs")
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED-PRUNED {name} @ step {rec['pruned_at_step']}: est ahead {rec['ahead']:.6f} "
          f"est imm {rec['imm']:.6f} {detail}")


def cmd_record_baseline(ahead, imm):
    rec = {"name": "t65_baseline", "param": "baseline", "config": dict(DEFAULTS),
           "ahead": round(float(ahead), 6), "imm": round(float(imm), 6), "users": 1000,
           "peak_lr": BASE_PEAK_LR, "muon_lr": BASE_MUON_LR,
           "source": "scratchpad/maxval (the validated MAX=65536 run) restricted to 5001-6000"}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED baseline: ahead {ahead} imm {imm}")


def cmd_loop():
    """Self-driving coordinate descent: run every remaining trial (its WS->decay->eval->record
    .cmd) until DONE. Resumable -- replays the journal on restart, so a teardown continues from
    the next trial. Launch this DETACHED (survives Esc). Each trial .cmd self-records."""
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
        print(f"\n===== TRIAL {name}  param={param}  cfg={json.dumps(cfg)} "
              f"peak_lr={peak_lr(cfg):g} muon_lr={muon_lr(cfg):g} =====", flush=True)
        cmd_path = f"{TRIAL_DIR}/{name}/{name}.cmd".replace("/", "\\")
        rc = subprocess.call(["cmd", "/c", cmd_path])
        if find(load_journal(), cfg) is None:
            print(f"ABORT: {name} did not record (rc={rc}). "
                  f"Check scratchpad/tuner65k/{name}.log. Stopping.", flush=True)
            return


def cmd_status():
    recs = load_journal()
    print(f"{'name':28} {'param':14} {'peak_lr':>9} {'muon_lr':>9} "
          f"{'ahead':>9} {'imm':>9} {'obj':>9}  note")
    base = next((x for x in recs if x["param"] == "baseline"), None)
    for r in recs:
        note = f"PRUNED@{r['pruned_at_step']} (estimated)" if r.get("pruned") else ""
        if base is not None and r is not base:
            # positive = better than baseline; "BAR" marks a trial that cleared it in BOTH modes
            da, di = base["ahead"] - r["ahead"], base["imm"] - r["imm"]
            cleared = r["ahead"] <= BAR["ahead"] and r["imm"] <= BAR["imm"]
            note = (f"d {da:+.6f}/{di:+.6f}" + ("  ** CLEARS THE BAR **" if cleared else "")
                    + (("  " + note) if note else ""))
        print(f"{r['name']:28} {r['param']:14} {r.get('peak_lr', 0):9.2e} "
              f"{r.get('muon_lr', 0):9.2e} {r['ahead']:9.6f} {r['imm']:9.6f} {obj(r):9.6f}  {note}")
    print(f"{'--- BAR (iter 31 on 5001-6000)':28} {'':14} {'':>9} {'':>9} "
          f"{BAR['ahead']:9.6f} {BAR['imm']:9.6f} {BAR['ahead']+BAR['imm']:9.6f}  "
          f"reach BOTH to have recovered what MAX=65536 cost")
    out = compute(recs)
    if out[0] == "done":
        best = out[1]
        print("\nCOORDINATE DESCENT COMPLETE. best:", json.dumps(best))
        r = find(recs, best)
        base = next((x for x in recs if x["param"] == "baseline"), None)
        if r and base:
            print(f"  vs baseline: ahead {base['ahead']-r['ahead']:+.6f}  "
                  f"imm {base['imm']-r['imm']:+.6f}")
            print("  NOTE: a sub-0.001 winner still needs confirming on the full VAL half "
                  "(5001-7500) before it becomes the recipe.")
    else:
        _, cfg, param = out
        if param == "baseline":
            print("\nNEXT: seed the baseline row -- `record-baseline 0.299250 0.266335` "
                  "(the maxval run restricted to 5001-6000); no GPU run needed.")
        else:
            print(f"\nNEXT: probe param={param} value={cfg[param]:g}  ({trial_name(param, cfg)})"
                  f"  peak_lr={peak_lr(cfg):g} muon_lr={muon_lr(cfg):g}")
    remaining = sum(len(g) for _, g in SPACE) - len(SPACE)
    # 4.0 h MEASURED end-to-end on trials 1 and 2 (WS 2h33 + decay 38 min + eval ~1 h), not projected
    print(f"\n(grid: {remaining} non-default points; ~4.2 h/trial measured => ~{remaining*4.2:.0f} h "
          f"if none prune)")


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
