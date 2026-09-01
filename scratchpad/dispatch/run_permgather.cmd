@echo off
REM ===========================================================================================
REM PHASE 1 SPEEDUP, ROUND 4: PermGather on the INTERLEAVED path -- bit-identity, then speed.
REM
REM THE FIND. `perm_gather` was wired on the SEQUENTIAL stream gather (srs_model.py:1080) and
REM MISSED on the INTERLEAVED one (:1249) -- which has been the champion's path since iter 41 and
REM runs that gather once per layer-step per split, 13 layer-steps deep. Its stock index_select
REM backward is the deterministic sort-based index_add that _PermGather's own docstring prices at
REM "~43% of the whole training step". Independently measured here: indexing is 37.6% of GPU time,
REM and RWKV_DETERMINISTIC=0 is worth +30.9% throughput.
REM
REM WHY IT SHOULD BE FREE. _PermGather.forward does clamp(idx,min=0) then index_select -- which is
REM character-identical to what :1249 already did. The backward differs only by a row-0 pad-sum
REM that adds exact zeros. So this should be BIT-IDENTICAL, not merely equivalent.
REM
REM ⚠ "SHOULD BE" IS NOT "IS", AND THE DIFFERENCE MATTERS HERE. If it perturbs the trajectory at
REM all, then the fixc arm (run with it) and the e2sc re-base (run without) stop being a
REM single-variable pair, and the interval measurement is confounded. That is why phase A is a
REM bit-identity test and not a speed test, and why the speed arms come second.
REM
REM RWKV_PERM_GATHER=0 disables BOTH sites, so arm A0 is "both stock" and A1 is "both permuted" --
REM a valid end-to-end test of the change. Losses are compared line-for-line, not eyeballed.
REM
REM Pinned: determinism=1, empty_cache=1, interleave=1 -- the shipped configuration.
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal enabledelayedexpansion
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\permgather.log
set STAMP=%RANDOM%%RANDOM%
set CFG=%DIR%\swup_65536.toml

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_INTERLEAVE=1
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
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_FSRS_CARD=
set RWKV_ID_FEATURES=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

echo ===== PERMGATHER ROUND 4 START %DATE% %TIME% ===== >> "%LOG%"

REM ---- PHASE A: bit-identity. 40 steps each, losses compared line-for-line. ----
set RWKV_MAX_STEPS=40
set RWKV_BENCH_WARMUP=0
for %%P in (1 0) do (
  set RWKV_PERM_GATHER=%%P
  echo --- trace perm_gather=%%P !TIME! >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\pg_trace%%P_%STAMP%.log" 2>&1
  echo   exit=!ERRORLEVEL! >> "%LOG%"
)
.venv\Scripts\python.exe scratchpad/dispatch/cmp_traces.py "%DIR%\pg_trace1_%STAMP%.log" "%DIR%\pg_trace0_%STAMP%.log" >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo NOT_BIT_IDENTICAL -- the change perturbs the trajectory; it must NOT be used for the >> "%LOG%"
  echo   fixc arm, because e2sc ran without it and the pair would be confounded. >> "%LOG%"
  echo DONE_EXIT_60 %DATE% %TIME% >> "%LOG%"
  exit /b 60
)
echo BIT_IDENTICAL -- safe to ship and safe for the fixc arm %TIME% >> "%LOG%"

REM ---- PHASE B: the speed arms, alternating so drift is visible ----
set RWKV_MAX_STEPS=220
set RWKV_BENCH_WARMUP=120
for %%P in (1 0 1 0) do (
  ping -n 31 127.0.0.1 >nul
  set RWKV_PERM_GATHER=%%P
  echo --- bench perm_gather=%%P !TIME! >> "%LOG%"
  .venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\pg%%P_%STAMP%.log" 2>&1
  findstr /C:"BENCH_RESULT" "%DIR%\pg%%P_%STAMP%.log" >> "%LOG%"
)

REM ---- PHASE C: profile with it ON, to confirm the indexing share actually fell ----
set RWKV_PERM_GATHER=1
set RWKV_MAX_STEPS=
set RWKV_BENCH_WARMUP=
set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=5
echo --- PHASE C: profile, perm_gather=1 %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\pg_prof_%STAMP%.log" 2>&1
findstr /C:"total GPU kernel time" "%DIR%\pg_prof_%STAMP%.log" >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
