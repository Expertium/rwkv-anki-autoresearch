"""Generate a full WS + decay + eval runner for one hybrid arm.

A GENERATOR, NOT A CLONE. CLAUDE.md's bug-rate audit found the root cause of the 2026-08 failure
run: runners were cloned across lineages, so each new run silently inherited its ancestor's
defects (27 historical runners carry the `endlocal`-before-the-marker bug alone). The fix it asked
for was one canonical template with the guards baked in. This is that template for the hybrid arms.

THE GUARDS, and what each one is a scar from:

  cd /d before anything          Win32_Process.Create starts in System32, where .venv does not
                                 exist.
  every %VAR% set before use     mk53/mk54 sliced the DIR/LOG block away; %LOG% expanded to empty,
                                 so the runner could not even log why it failed and its waiter
                                 hung forever.
  marker BEFORE endlocal         endlocal restores the pre-setlocal env, so `endlocal & echo ...
                                 >> "%LOG%"` appends to "". iter 53 finished cleanly and never
                                 said so; four chained waiters polled for 45 minutes.
  no  < > & | ^  in REM          cmd parses redirection before it honours REM, so an arrow in a
                                 comment is a syntax error pointing nowhere near the comment.
  guard strings built from the   `decayshape` set alpha 0.5 and kept a findstr for "0.9"; a
  value the runner SETS          correct run would have been rejected at the very end.
  artifact gates, not exit codes train_rwkv can swallow a fatal error and exit 0 -- that is how
                                 the 08-18 resume decayed a half-trained model.
  phase 0 asserts the MODEL      RWKV_STRIP_CMIX silently ignores a name matching no layer, and
                                 every arm changes the depths. See assert_arch.py.

The asserts run on the OUTPUT, which is the half mk53/mk54 missed: they checked only that stale
text did not leak IN, never that required setup SURVIVED.

Usage:
    python scratchpad/hybrid100k/mk_runner.py A
"""

import io
import os
import re
import sys

REPO = r"C:\Users\Andrew\rwkv-anki-autoresearch"
DIRW = r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hybrid100k"
DIRP = "scratchpad/hybrid100k"
CHAMP_PARAMS = 558212
# The champion arch + strip list. Used ONLY by phase 0a, which must load the champion's own
# checkpoint. Anywhere else they would silently make the arm not be the arm.
CHAMP_ARCH = "scratchpad/track2_a18/architecture_d80_lora4_cnd.py"
CHAMP_STRIP = ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
               "deck_id:1,deck_id:2,card_id:1")

# ⚠ THE ARM DEFINITIONS ARE IMPORTED, NEVER RESTATED. The first version of this file hardcoded a
# strip list per arm, and two of the three were WRONG -- I wrote them from the pattern instead of
# from the arch. assert_arch.py caught it before any GPU time (arm B has the SAME depths as A, so
# my invented `deck_id:2` matched nothing; arm C is 2/1/1/1/1, so six of my nine entries matched
# nothing). The bug was not the typing, it was the DUPLICATION: mk_arch.py already owns the depths
# and the priced param count, and a second copy can only ever drift from it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mk_arch  # noqa: E402  -- the single source of truth for what each arm IS


# Runs that are NOT hybrid arms. Same runner, same guards; only the arch, the extra env and
# the gate differ. Keyed by the name passed on the command line.
EXTRA_SPECS = {
    "V1": {
        "arch": DIRP + "/arch_fsrs_v1.py",
        # card_id has 0 layers under V1, so the champion's `card_id:1` would match nothing --
        # exactly the silent-strip trap assert_arch.py exists to catch.
        "strip": ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
                  "deck_id:1,deck_id:2"),
        "params": 488858,
        "desc": ("FSRS-7 (S_long, S_short, D) replaces the card stream's WKV; the trunk EMITS "
                 "the 34 parameters per review"),
        "env": ["RWKV_FSRS_CARD=0"],
        "gate": ("REM GATE: CLAUDE.md's SIZE/SPEED EXCEPTION, pre-registered 2026-08-30 BEFORE\n"
                 "REM this run -- accept iff BOTH modes stay within +0.0015 of the champion AND\n"
                 "REM per-card state strictly shrinks. NOT the parameter-ratio rule: V1 moves\n"
                 "REM per-card state 2,880 floats to 3 while moving parameters only 12.4%, so a\n"
                 "REM ratio bar would be 0.0000694 per mode -- tighter than the ordinary accept\n"
                 "REM bar, which is a category error for a state reduction.\n"
                 "REM Champion iter 53 == 0.297523 ahead / 0.265191 imm on the VAL half."),
    },
}


def arm_spec(arm):
    """(arch, strip list, priced params, description, extra env lines, gate text)."""
    if arm in EXTRA_SPECS:
        d = EXTRA_SPECS[arm]
        return d["arch"], d["strip"], d["params"], d["desc"], d["env"], d["gate"]
    desc, _hd, _nh, L, _hm, _fm, exp_p, _exp_cs = mk_arch.ARMS[arm]
    return DIRP + "/arch_%s.py" % arm, mk_arch.strip_for(L), exp_p, desc, [], ""


WS_TOML = '''# HYBRID ARM {arm}: {desc}
# Priced at {params} parameters (assert_arch.py re-checks this in phase 0).
#
# Clone of the champion's i53_ws.toml with ONLY the save folder and prefix changed -- the arm is
# entirely in the .cmd's RWKV_ARCH_MODULE + RWKV_STRIP_CMIX, so this comparison is arch-only.
# The training recipe (budget, LR, warmup, KD, seed) is the champion's, unchanged.
#
# WHY THE KD DUMP IS STILL VALID FOR A DIFFERENT STUDENT. The dump replays the d=128 TEACHER's
# outputs by step index and verifies a per-step `labels_sum` checksum. Batch composition depends
# on the db, the user range, MAX_TRAIN_GLOBAL_LEN, the fetch-process count and the seeds -- never
# on the student's architecture. So a smaller student consumes the same dump.
# The corollary is the trap recorded in CLAUDE.md: the checksum proves LABEL alignment and says
# nothing about INPUTS, so any input-side change invalidates the dump while it keeps passing.
# WARNING: the data-affecting fields below MUST stay identical to the dump's. Do not "tidy" them.
TRAIN_USERS_START = 1
TRAIN_USERS_END = 5000
VALIDATE_USERS_START = 5001
VALIDATE_USERS_END = 5010

TRAIN_DATASET_LMDB_PATH = "train_db_5k_h1"
TRAIN_DATASET_LMDB_SIZE = 400_000_000_000
VALIDATE_DATASET_LMDB_PATH = "F:/rwkv_lmdb/test_db_5k"
VALIDATE_DATASET_LMDB_SIZE = 250_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000

NUM_FETCH_PROCESSES = 2
MAX_TRAIN_GLOBAL_LEN = 65536

TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = 400
EPOCHS = 1
VALIDATE_EVERY = 1000
PEAK_LR = 1e-3

LOAD_MODEL = false
SAVE_MODEL_FOLDER = "{dirp}"
SAVE_MODEL_PREFIX = "{pfx}_ws"
DEVICE = "cuda"
DTYPE = "bfloat16"

USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
'''

CMD = r'''@echo off
REM ===========================================================================================
REM HYBRID ARM {arm}: {desc}
REM Priced at {params} params vs the champion's 558,212. Design: optimization/HYBRID_100K.md.
REM
{gate}
REM GATE (Andrew 2026-08-28): the track-2 RATIO rule, not the flat accept bar.
REM     ratio == 100,000 * (LL_cand - LL_champ) / (params_champ - params_cand)
REM must be at most 0.0001 in BOTH modes, and params must strictly decrease.
REM Champion iter 53 == 0.297523 ahead / 0.265191 imm on the VAL half, 558,212 params.
REM With {params} params the budget is dparams == {dparams}, so the largest tolerable
REM regression is about {allow} in each mode. For scale the A0-to-A18 ladder scored
REM 0.0000435 ahead / 0.0000240 imm, so this is winnable and not a formality.
REM
REM SINGLE VARIABLE: every training-recipe field matches the champion. The arm is exactly
REM RWKV_ARCH_MODULE plus this arm's own RWKV_STRIP_CMIX.
REM
REM PHASE 0 HAS TWO GUARDS.
REM   1. the scripted-eval smoke -- a PLAIN eval is the only path that TorchScript-compiles the
REM      model, so a runtime break is invisible to training and surfaces only after the decay.
REM      srs_model.py and srs_model_rnn.py were touched on 2026-08-29 (RWKV_ABLATE_FEATURES).
REM   2. assert_arch.py -- RWKV_STRIP_CMIX silently ignores a name that matches no layer, and
REM      every arm changes the per-stream depths, so a stale list yields a different model with
REM      no error at all. It asserts the exact param count and that every entry matched.
REM
REM Do NOT edit this file while it runs. cmd.exe resumes from a saved byte offset, and
REM git checkout is not a safe undo (line endings shift the offset). Iters 43 and 46 died so.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d {repo}
set DIR={dirw}
set LOG=%DIR%\hyb{arm}.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

set PYTHONUNBUFFERED=1
set PYTHONPATH={repo}
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM ---- THE ARM: the architecture, and this arm's OWN strip list ----
set RWKV_ARCH_MODULE={arch}
set RWKV_STRIP_CMIX={strip}
{extra_env}set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_MUON_INCLUDE_LORA=1
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_GRAD_STATS=%DIR%\grad_stats_{pfx}.json
set RWKV_ID_FEATURES=
set RWKV_ABLATE_FEATURES=
set RWKV_MAX_STEPS=

echo ===== HYBRID ARM {arm} ({params} params) START %DATE% %TIME% ===== >> "%LOG%"

REM ================= PHASE 0a: the scripted-eval guard (iter 48's lesson) =================
REM RUNS UNDER THE CHAMPION'S ARCH, NOT THIS ARM'S, and that is load-bearing. The guard asks
REM "does the CURRENT CODE TorchScript-compile and run a plain eval", and it answers by
REM loading a REAL checkpoint: the champion's d=80 i45 one. Under this arm's d=32 arch that
REM checkpoint cannot load, and the first version of this runner died in 9 seconds on 200
REM size mismatches -- the very clone-inheritance failure this generator exists to prevent,
REM committed inside it. The arm's own values are restored explicitly below.
set RWKV_ARCH_MODULE={champ_arch}
set RWKV_STRIP_CMIX={champ_strip}
REM The smoke loads the CHAMPION checkpoint, which has no FSRS core -- the flag
REM must be OFF for it and restored below with the rest of the arm's env.
set RWKV_FSRS_CARD=
"C:\Program Files\Git\bin\bash.exe" scratchpad/parity3/smoke_scripted_eval.sh scratchpad/iter45_kddecay/i45_eval.toml > "%DIR%\smoke_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo {tag} SCRIPTED_EVAL_SMOKE_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_45 >> "%LOG%"
  exit /b 45
)
REM Restore the arm's own values. Deliberately a plain set/restore rather than a nested
REM setlocal: preflight_runner.py's endlocal check is nesting-blind and would flag the inner
REM endlocal as the bug it was built for. Fewer mechanisms, same guarantee, and what is in
REM force at each line is readable without modelling cmd scoping.
set RWKV_ARCH_MODULE={arch}
set RWKV_STRIP_CMIX={strip}
{extra_env}echo {tag} SMOKE_OK %TIME% >> "%LOG%"

REM ================= PHASE 0b: is this the model we priced? =================
.venv\Scripts\python.exe scratchpad/hybrid100k/assert_arch.py {params} > "%DIR%\arch_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo {tag} ARCH_ASSERT_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_46 >> "%LOG%"
  exit /b 46
)
echo {tag} ARCH_OK {params} params %TIME% >> "%LOG%"

REM ================= PHASE A: WS, 1 epoch, KD alpha 0.9 =================
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.9
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config {dirp}/{pfx}_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo {tag} WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_21 >> "%LOG%"
  exit /b 21
)
findstr /C:"[kd-mix] KD ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo {tag} NOKD_WS %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_35 >> "%LOG%"
  exit /b 35
)
findstr /C:"alpha FIXED at 0.9" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo {tag} WRONGALPHA_WS %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_36 >> "%LOG%"
  exit /b 36
)
REM Gate on the ARTIFACT, not the exit code: train_rwkv can swallow a fatal error and exit 0.
if not exist "%DIR%\{pfx}_ws_%WSSTEPS%.pth" (
  echo {tag} NO_WS_CKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_30 >> "%LOG%"
  exit /b 30
)
echo {tag} WS_OK %TIME% >> "%LOG%"

REM ================= PHASE B: decay, ratio 1.0, KD alpha 0.5 =================
REM The champion runs KD alpha 0.9 in WS and 0.5 in DECAY (iters 39 and 45). The reset below
REM lives here on purpose: iter 45 became champion because its runner FORGOT it, and a
REM decay-only generator that slices this phase away would inherit 0.9 by accident.
set RWKV_KD_ALPHA=0.5
.venv\Scripts\python.exe scratchpad/write_decay_setup.py {dirp} {pfx}_ws {pfx}_d %DIR%\{pfx}_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo {tag} DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_22 >> "%LOG%"
  exit /b 22
)
findstr /C:"{pfx}_ws_%WSSTEPS%" "%DIR%\dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo {tag} WRONGCKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_32 >> "%LOG%"
  exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config %DIR%\{pfx}_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo {tag} DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_23 >> "%LOG%"
  exit /b 23
)
findstr /C:"alpha FIXED at 0.5" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo {tag} WRONGALPHA_DECAY %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_36 >> "%LOG%"
  exit /b 36
)
if not exist "%DIR%\{pfx}_d_%WSSTEPS%.pth" (
  echo {tag} NO_DECAY_CKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_31 >> "%LOG%"
  exit /b 31
)
echo {tag} DECAY_OK %TIME% >> "%LOG%"

REM ================= PHASE C: rectified VAL-half eval =================
.venv\Scripts\python.exe scratchpad/write_eval_toml.py {dirp} {pfx}_d %DIR%\{pfx}_eval.toml RWKV-hyb{arm} RWKV-P-hyb{arm} 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo {tag} TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 >> "%LOG%"
  exit /b 24
)
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_GRAD_STATS=
set RWKV_EVAL_PAVA=1
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\{pfx}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo {tag} EVAL attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\{pfx}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo {tag} EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 >> "%LOG%"
  exit /b 25
)
echo {tag} EVAL_OK %TIME% >> "%LOG%"
REM The terminal marker goes BEFORE endlocal: endlocal restores the pre-setlocal environment,
REM so %LOG% would expand to empty and the append would silently go nowhere. That is exactly
REM how iter 53 finished cleanly at 07:11 and stranded four waiters for 45 minutes.
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
'''


def check_output(cmd, arm, arch, strip):
    """Assert on what was WRITTEN, not on what was avoided.

    mk53.py and mk54.py both asserted that stale text did not leak IN (no "iter45", no "i53_").
    Neither asserted that required setup SURVIVED the slice, and that is the half that broke:
    the DIR/LOG block was thrown away, %LOG% expanded to empty, and the failure was maximally
    quiet. These checks run in the other direction.
    """
    assert "cd /d " + REPO in cmd, "cd /d missing: Win32_Process.Create starts in System32"

    declared, first_use = {}, {}
    for i, line in enumerate(cmd.splitlines()):
        m = re.match(r"\s*set (\w+)=", line)
        if m:
            declared.setdefault(m.group(1), i)
        for v in re.findall(r"%(\w+)%", line):
            first_use.setdefault(v, i)
    builtin = {"ERRORLEVEL", "DATE", "TIME", "RANDOM"}
    for v, use in sorted(first_use.items()):
        if v in builtin:
            continue
        assert v in declared and declared[v] <= use, (
            "%%%s%% used at line %d before it is set (%s)" % (v, use, declared.get(v))
        )

    body = cmd.splitlines()
    mark = max(i for i, l in enumerate(body) if l.startswith("echo DONE_EXIT_0"))
    endl = max(i for i, l in enumerate(body) if l.startswith("endlocal"))
    assert mark < endl, "DONE_EXIT_0 must be written BEFORE endlocal"

    for i, l in enumerate(body):
        if l.strip().upper().startswith("REM"):
            bad = sorted(set("<>&|^") & set(l))
            assert not bad, "REM line %d contains %s: cmd parses redirection before REM" % (i, bad)

    # the alpha guards must match the alphas the runner SETS (the decayshape lesson)
    for phase, a in (("ws", "0.9"), ("decay", "0.5")):
        assert "set RWKV_KD_ALPHA=" + a in cmd, phase
        assert 'findstr /C:"alpha FIXED at %s" "%%DIR%%\\%s_%%STAMP%%.log"' % (a, phase) in cmd, phase

    # the arm must carry its OWN arch and strip list
    assert arch in cmd and strip in cmd
    # ★ THE ASSERT THIS FAILURE EARNED. Which arch is in force is POSITIONAL in a .cmd, so
    # checking that the champion arch is 'absent' was exactly the wrong shape of check -- it
    # must be PRESENT at phase 0a (which loads the champion checkpoint) and ABSENT by the
    # time training starts. Assert both, by position.
    lines = cmd.split(chr(10))
    smoke = [i for i, l in enumerate(lines)
             if "smoke_scripted_eval.sh" in l and "bash.exe" in l]
    assert len(smoke) == 1, "expected 1 scripted-eval smoke line, found %d" % len(smoke)
    pre = [l for l in lines[:smoke[0]] if l.startswith("set RWKV_ARCH_MODULE=")]
    assert pre and pre[-1] == "set RWKV_ARCH_MODULE=" + CHAMP_ARCH, (
        "phase 0a would run under %r, but it loads the CHAMPION checkpoint and must run "
        "under %r" % (pre[-1] if pre else None, CHAMP_ARCH))
    train = [i for i, l in enumerate(lines) if "train_rwkv --config" in l]
    assert train, "no training phase found"
    pre_t = [l for l in lines[:train[0]] if l.startswith("set RWKV_ARCH_MODULE=")]
    assert pre_t[-1] == "set RWKV_ARCH_MODULE=" + arch, (
        "training would run under %r, not this arm's %r" % (pre_t[-1], arch))
    # Positional, for the same reason as the arch check above: phase 0a legitimately sets the
    # CHAMPION's strip list (it loads the champion checkpoint), so mere presence proves
    # nothing. What matters is which assignment is live at each phase.
    pre_s = [l for l in lines[:smoke[0]] if l.startswith("set RWKV_STRIP_CMIX=")]
    assert pre_s and pre_s[-1] == "set RWKV_STRIP_CMIX=" + CHAMP_STRIP, (
        "phase 0a must run under the champion strip list, not %r" % (pre_s[-1] if pre_s
                                                                    else None))
    pre_st = [l for l in lines[:train[0]] if l.startswith("set RWKV_STRIP_CMIX=")]
    assert pre_st[-1] == "set RWKV_STRIP_CMIX=" + strip, (
        "training would run under %r, not this arm's %r" % (pre_st[-1], strip))


def main():
    if len(sys.argv) != 2 or (sys.argv[1] not in mk_arch.ARMS
                              and sys.argv[1] not in EXTRA_SPECS):
        print("usage: mk_runner.py {A|B|C|V1}")
        return 2
    arm = sys.argv[1]
    arch, strip, params, desc, extra_env, gate_txt = arm_spec(arm)
    pfx = "hy" + arm.lower()
    extra_env_block = ("\n".join("set " + e for e in extra_env) + "\n"
                       if extra_env else "")
    dparams = CHAMP_PARAMS - params
    allow = 0.0001 * dparams / 100000.0

    toml = WS_TOML.format(arm=arm, desc=desc, params=params, dirp=DIRP, pfx=pfx)
    cmd = CMD.format(arm=arm, desc=desc, params=params, arch=arch, strip=strip, pfx=pfx,
                     repo=REPO, dirw=DIRW, dirp=DIRP, dparams="{:,}".format(dparams),
                     allow="%.6f" % allow, tag="hyb" + arm,
                     champ_arch=CHAMP_ARCH, champ_strip=CHAMP_STRIP,
                     extra_env=extra_env_block, gate=gate_txt)
    check_output(cmd, arm, arch, strip)

    toml_path = os.path.join(DIRW, pfx + "_ws.toml")
    cmd_path = os.path.join(DIRW, "run_hyb" + arm + ".cmd")
    io.open(toml_path, "w", encoding="utf-8", newline="\n").write(toml)
    io.open(cmd_path, "w", encoding="utf-8", newline="\r\n").write(cmd)
    print("wrote " + toml_path)
    print("wrote " + cmd_path)
    print("arm %s: %s params, dparams %s, max tolerable regression %.6f per mode"
          % (arm, "{:,}".format(params), "{:,}".format(dparams), allow))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
