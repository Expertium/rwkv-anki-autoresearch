@echo off
REM ===========================================================================================
REM ITER 48: RETRIEVABILITY-COUPLED RATING HEAD (RWKV_RCOUPLE=1). Family: architecture.
REM PROPOSALS.md item #2, promoted 2026-08-15 after iter 46 refuted the argument that demoted it.
REM
REM THE LEVER: the curve head's logit R(t) is fed into the 4 rating logits via 4 zero-init
REM coefficients. Two mechanisms: the rating head gains R(t) as an INPUT (it has no explicit
REM notion of how due a card is, though the curve head computes exactly that at the SAME t), and
REM a GRADIENT PATH from the better-conditioned imm objective back into the curve head.
REM iter 46 showed the ahead/imm gap is NOT transferable by soft targets -- a same-forward-pass
REM teacher only re-expresses what the student computes -- and concluded the remedy is to change
REM what the ahead path COMPUTES. This does exactly that.
REM
REM SINGLE VARIABLE vs the iter-45 champion: this file is run_iter45.cmd with the save prefixes
REM changed and ONE line added (set RWKV_RCOUPLE=1). Everything else -- seed 4321, KD alpha 0.9
REM WS / 0.5 decay, PAVA 0.2, the tuned HPs, the speed stack -- is identical.
REM ⚠ Unlike iter 45, the lever is a MODEL change, so it applies to BOTH phases and WS is rerun.
REM
REM ZERO-INIT: rcouple_w starts at 0, so step 0 is byte-identical to the champion. A null result
REM therefore cannot be blamed on a bad initialisation -- the coupling is learned or it is not.
REM
REM DEPLOY: this edits the forward pass, so it adds a Rust port gap and needs a fresh parity trace
REM BEFORE shipping. Both Python paths (srs_model + srs_model_rnn) already implement it and agree
REM to 0.000e+00 (scratchpad/parity3/smoke_rcouple.py). Port only if this is ACCEPTED.
REM
REM ⚠ Do NOT git checkout / edit this file while it runs (iters 43 and 46 died that way; git
REM normalises line endings, which shifts cmd.exe's saved byte offset).
REM ⚠ NO del of result jsonls (fresh tag; retries resume from banked users).
REM ⚠ No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter48_rcouple
set LOG=%DIR%\iter48.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 48 (retrievability-coupled rating head) START %DATE% %TIME% ===== > "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_GRAD_STATS=%DIR%\grad_stats.json
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.9

REM ================= THE LEVER, and the ONLY difference from run_iter45.cmd =================
set RWKV_RCOUPLE=1

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter48_rcouple/i48_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
REM THE experiment's own guard: the coupling must actually be ON. A banner proves the value was
REM COMPUTED, so this is necessary but not sufficient -- the sufficiency check is that
REM smoke_rcouple.py asserts the coupling MOVES P(Again) (the qat-inert lesson: a truthful banner
REM on an object that is then discarded). Without this the run silently reduces to iter 45 and
REM would be recorded as a null.
findstr /C:"[RCOUPLE] retrievability-coupled rating head ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NORCOUPLE_WS %DATE% %TIME% >> "%LOG%"
  exit /b 37
)
findstr /C:"[interleave] round-robin layer schedule ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOILV %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
findstr /C:"placement = front-loaded" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGPLACEMENT %DATE% %TIME% >> "%LOG%"
  exit /b 33
)
echo WS OK %TIME% >> "%LOG%"

set RWKV_KD_ALPHA=0.5
set RWKV_GRAD_STATS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter48_rcouple i48_ws i48_d scratchpad/iter48_rcouple/i48_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter48_rcouple/i48_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"[RCOUPLE] retrievability-coupled rating head ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NORCOUPLE_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 38
)
findstr /C:"[kd-mix] KD ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOKD_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 35
)
findstr /C:"alpha FIXED at 0.5" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGALPHA_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 36
)
echo DECAY OK (RCOUPLE ON, KD ON, alpha 0.5) %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter48_rcouple i48_d scratchpad/iter48_rcouple/i48_eval.toml RWKV-iter48_rcouple RWKV-P-iter48_rcouple 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM ⚠ RWKV_RCOUPLE stays SET for the eval -- it is a MODEL property, not a training trick. Clearing
REM it would score a different function than was trained (and would fail to load the checkpoint,
REM since rcouple_w is a conditional Parameter). The speed stack IS cleared, as always.
REM ⚠ NO_JIT is cleared here, so the eval SCRIPTS the model. That is the path where the rank-1
REM regulariser's module-level-global bug would have fired; fixed 2026-08-15 and verified by
REM torch.jit.script succeeding with RCOUPLE both off and on.
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter48_rcouple/i48_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter48_rcouple/i48_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
