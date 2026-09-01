@echo off
REM ===========================================================================================
REM DIAGNOSTIC, not a gate run: is arm C's illegal-memory-access tied to the INTERLEAVE path?
REM
REM Arm C is the only config where a round has exactly ONE participating stream
REM (sched [[0,1],[0,-1],[0,-1],[0,-1],[0,-1]] -- card alone in round 1). Arms A and B ran
REM 10,935 steps each with no fault; arm C faulted twice in 937 steps, at step 791 and step 146.
REM Its loss curve is normal and matches A/B, so it is NOT diverging, and the two runs are
REM bit-identical through step 145 yet died at different steps -- so the fault is not
REM deterministic in the data either. That is the signature of an out-of-bounds access that
REM usually lands in mapped memory.
REM
REM This arm is arm C EXACTLY, with RWKV_INTERLEAVE removed and capped at 800 steps (past both
REM observed failure points). Clean => the bug is in the interleave path. Faults anyway => it is
REM not, and the search moves to the arch itself.
REM
REM A diagnostic run's logloss is meaningless and must never be recorded as a result.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hybrid100k
set LOG=%DIR%\diag_c_noilv.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/hybrid100k/arch_C.py
set RWKV_STRIP_CMIX=card_id:1,preset_id:0,user_id:0
REM ---- THE ONLY DIFFERENCE FROM run_hybC.cmd: no RWKV_INTERLEAVE ----
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
set RWKV_KD_MIX=C:\rwkv_kd_dump\t128_seedpair_65k:10935
set RWKV_KD_ALPHA=0.9
set RWKV_MAX_STEPS=800

echo ===== DIAG C no-interleave START %DATE% %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/hybrid100k/hyc_ws.toml > "%DIR%\diagc_%STAMP%.log" 2>&1
echo exit=%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
findstr /C:"illegal memory access" "%DIR%\diagc_%STAMP%.log" >nul
if %ERRORLEVEL%==0 (
  echo RESULT: FAULTED ANYWAY - not the interleave path >> "%LOG%"
) else (
  echo RESULT: CLEAN through 800 steps - the fault is in the INTERLEAVE path >> "%LOG%"
)
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
