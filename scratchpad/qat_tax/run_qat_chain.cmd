@echo off
REM ===========================================================================================
REM THE THREE-CELL QAT TAX CHAIN (Andrew 2026-08-12).   call: run_qat_chain.cmd <TAG> <SHIFT_CB>
REM
REM   cell 1  no QAT, full precision      = iter 45 itself (0.297697 / 0.265375) -- ALREADY HAVE
REM   cell 2  QAT-trained, eval QUANTIZED = the deploy number          -> phase B below
REM   cell 3  QAT-trained, eval FULL PREC = same ckpt, QAT env off     -> phase C below
REM
REM   (2)-(1) = FULL TAX
REM   (2)-(3) = PRECISION DEGRADATION -- what quantization costs a model trained for it
REM   (3)-(1) = MODEL DRIFT           -- what training under fake-quant costs by itself
REM
REM The d=32 record says drift dominates (decay-QAT #39: degradation -0.000127/+0.000018 vs drift
REM +0.001129/+0.002456), which is what makes the split worth the extra eval: if it holds here,
REM reducing the tax means changing HOW WE TRAIN, and a better codebook cannot help much.
REM
REM SINGLE-VARIABLE BY CONSTRUCTION: phase A re-runs iter 45's OWN decay from its OWN WS-final
REM (i45_ws_10935) with the q72u QAT env added and NOTHING else changed -- KD stays at alpha 0.5,
REM same seed, same MAX, same LR. So cell2 - iter45 isolates QAT and nothing else.
REM
REM ⚠ This file is SEPARATE from run_arm.cmd on purpose: that one was executing when this was
REM written, and cmd.exe resumes a batch file from a saved byte offset (editing a running .cmd
REM cost iters 43 and 46).
REM ⚠ Delete stale result jsonls for these tags BEFORE launching -- eval_sharded skips completed
REM users, so old numbers get silently reused. Done from the caller, NOT here: the two eval
REM attempts below deliberately have no `del` between them so the resume property survives an OOM.
REM ⚠ write_decay_setup's folder arg must be RELATIVE (an absolute C:\Users path embeds \U in the
REM toml and tomli dies with "Invalid hex value").
REM
REM WHY 500 USERS (5001-5500) AND NOT THE FULL VAL HALF. A QUANTIZED eval costs ~28 s/user vs
REM ~4.3 s/user plain -- measured 2026-08-12, and it is expected arithmetic, not a bug: QAT adds
REM ~73 ms of fixed per-step work (profiled +13% on a 578 ms TRAINING step), which is ~6x on a
REM much cheaper eval step. The full VAL half would be ~19 h for cell 2 alone. 500 users costs
REM ~3.9 h and still resolves this easily: the paired noise floor is ~1.7e-4 there (7.5e-5 at
REM n=2500, scaled by sqrt(5)) against an expected tax of ~0.001-0.005, i.e. 6-30x margin.
REM ALL cells use the SAME 500 users -- including the PTQ arm -- so every component of the
REM decomposition is exactly paired and the report's tax == degradation + drift CHECK closes to
REM zero. If a component lands marginal, extending cell 3 is cheap (it is a PLAIN eval); it is
REM cell 2 that is expensive, and that is the one 500 users is chosen to afford.
REM ===========================================================================================
setlocal
set TAG=%~1
set SHIFT_CB=%~2
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set SRCREL=scratchpad/iter45_kddecay
set SRC=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter45_kddecay
set LOG=%DIR%\qat_chain.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== QAT CHAIN %TAG% (cb=%SHIFT_CB%) START %DATE% %TIME% ===== >> "%LOG%"

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
set RWKV_EVAL_PAVA=1

REM ---- the q72u QAT recipe, shift catalog swapped for the chosen C=80 refit ----
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt
set RWKV_QAT_SHIFT_PQ=%SHIFT_CB%
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set RWKV_NO_JIT=1

REM ================= PHASE 0: prove the QAT env is NOT inert (2026-08-12 bug) =================
.venv\Scripts\python.exe -u scratchpad/qat_tax/assert_qat_live.py > "%DIR%\%TAG%_qatassert_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% QATINERT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_44 >> "%LOG%"
  endlocal & exit /b 44
)

REM ================= PHASE A: decay-QAT, warm-started from iter 45's WS-final =================
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.5
.venv\Scripts\python.exe scratchpad/write_decay_setup.py %SRCREL% i45_ws %TAG%_d %DIR%\%TAG%_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\%TAG%_dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_22 >> "%LOG%"
  endlocal & exit /b 22
)
findstr /C:"i45_ws_%WSSTEPS%" "%DIR%\%TAG%_dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% WRONGCKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_32 >> "%LOG%"
  endlocal & exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config %DIR%\%TAG%_decay.toml > "%DIR%\%TAG%_decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_23 >> "%LOG%"
  endlocal & exit /b 23
)
REM train_rwkv can swallow fatal errors and exit 0 -> gate on BANNERS + ARTIFACT, not just rc.
findstr /C:"[QAT-SHIFT-PQ] loaded" "%DIR%\%TAG%_decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% NOQAT_DECAY %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_42 >> "%LOG%"
  endlocal & exit /b 42
)
findstr /C:"[kd-mix] KD ON" "%DIR%\%TAG%_decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% NOKD_DECAY %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_43 >> "%LOG%"
  endlocal & exit /b 43
)
if not exist "%SRC%\%TAG%_d_%WSSTEPS%.pth" (
  echo CHAIN %TAG% NOCKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_31 >> "%LOG%"
  endlocal & exit /b 31
)
echo CHAIN %TAG% PHASE_A_OK %TIME% >> "%LOG%"
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

REM ================= PHASE B (CELL 2): eval QUANTIZED, full VAL half =================
.venv\Scripts\python.exe scratchpad/write_eval_toml.py %SRCREL% %TAG%_d %DIR%\%TAG%_q_eval.toml RWKV-%TAG%q RWKV-P-%TAG%q 5001 5500 > "%DIR%\%TAG%_qetoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% QTOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 >> "%LOG%"
  endlocal & exit /b 24
)
REM shard LOGS only (never the result jsonls -- those carry eval_sharded's resume property):
REM the banner guards below grep shard_*.log by wildcard, so a stale log from an earlier QAT run
REM could otherwise answer for this one.
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\%TAG%_q_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_qeval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\%TAG%_q_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_qeval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% QEVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 >> "%LOG%"
  endlocal & exit /b 25
)
REM the banner lands in the SHARD log, not eval_sharded's parent log -- grep both.
findstr /C:"[QAT-SHIFT-PQ] loaded" "%DIR%\%TAG%_qeval1_%STAMP%.log" "%DIR%\%TAG%_qeval2_%STAMP%.log" scratchpad\eval_shards\shard_*.log >nul
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% CELL2_NOT_QUANTIZED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_41 >> "%LOG%"
  endlocal & exit /b 41
)
echo CHAIN %TAG% CELL2_OK %TIME% >> "%LOG%"

REM ================= PHASE C (CELL 3): eval the SAME ckpt at FULL PRECISION =================
REM Every QAT knob cleared. NO_JIT cleared too, so this matches cell 1's exact conditions
REM (iter 45's eval ran JIT-on); jit vs eager is verified bit-identical, so this only removes a
REM difference, never adds one.
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_PQ=
set RWKV_QAT_SHIFT_PQ=
set RWKV_QAT_SHIFT_SCOPE=
set RWKV_QAT_NORM_BITS=
set RWKV_QAT_FUSED=
set RWKV_NO_JIT=
.venv\Scripts\python.exe scratchpad/write_eval_toml.py %SRCREL% %TAG%_d %DIR%\%TAG%_fp_eval.toml RWKV-%TAG%fp RWKV-P-%TAG%fp 5001 5500 > "%DIR%\%TAG%_fpetoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% FPTOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_26 >> "%LOG%"
  endlocal & exit /b 26
)
REM ⚠ MANDATORY here, not just tidiness: phase B's shard log contains the banner, and phase C's
REM guard is INVERTED (it fails if the banner is found). Without this del, a stale phase-B log
REM would fail a perfectly good cell 3.
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\%TAG%_fp_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_fpeval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\%TAG%_fp_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_fpeval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo CHAIN %TAG% FPEVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_27 >> "%LOG%"
  endlocal & exit /b 27
)
REM INVERTED guard: cell 3 must NOT be quantized. If the banner is present the env did not clear
REM and cells 2 and 3 would be the same number (the failure this whole chain exists to avoid).
findstr /C:"[QAT-SHIFT-PQ] loaded" "%DIR%\%TAG%_fpeval1_%STAMP%.log" scratchpad\eval_shards\shard_*.log >nul
if %ERRORLEVEL%==0 (
  echo CHAIN %TAG% CELL3_STILL_QUANTIZED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_45 >> "%LOG%"
  endlocal & exit /b 45
)
echo CHAIN %TAG% CELL3_OK %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
