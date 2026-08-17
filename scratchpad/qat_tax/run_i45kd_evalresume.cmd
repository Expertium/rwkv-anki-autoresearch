@echo off
REM ===========================================================================================
REM QAT#2 PHASE B ONLY -- eval RESUME after the 2026-08-17 12:53 machine freeze.
REM
REM WHAT HAPPENED: giant user 6104 (work 1,274,765) pushed VRAM to 11,981 of 12,282 MiB with a
REM loaded desktop. Power fell to 42 W while util still read 99 percent, i.e. the GPU stalled in
REM WDDM paging rather than computing, and the box froze. Andrew forced a dump with RightCtrl
REM plus Space twice; bugcheck 0xE2 MANUALLY_INITIATED_CRASH, MEMORY.DMP 4.4 GB.
REM
REM WHAT SURVIVED: the whole 13 h decay phase. qtaxg_i45kd_d_10935.pth and BOTH exported
REM learnable catalogs are on disk, and 1,103 of 2,500 eval users are already in the result
REM jsonls. get_result skips users already present, so this resumes at 6104 in a FRESH process,
REM which is most of what the solo phase would have bought (clean allocator, giant hit first).
REM
REM DO NOT delete the result jsonls before running this. That is what makes the resume cheap.
REM Phases A0 and A (dump, decay, probe) are deliberately absent -- they are already done and
REM re-running them would overwrite a good checkpoint with a fresh 13 h of GPU.
REM ===========================================================================================

REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set TAG=qtaxg_i45kd
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set SRCREL=scratchpad/iter45_kddecay
set SRC=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter45_kddecay
set LOG=%DIR%\i45kd.log
set STAMP=%RANDOM%%RANDOM%
set WSSTEPS=10935


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
set RWKV_GRAD_STATS=%DIR%\%TAG%_grad_stats.json

REM ---- the frozen quantizer (refit catalogs) + THE ONE CHANGE: both catalogs LEARNABLE ----
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_c80_b10.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_c80_m2b12.txt
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set RWKV_NO_JIT=1

echo ===== QAT#2 EVAL RESUME (after freeze) START %DATE% %TIME% ===== >> "%LOG%"

set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_GRAD_STATS=

REM ================= EVAL ENV: point at the LEARNED catalogs, learning OFF =================
set RWKV_QAT_PQ=%SRCREL%/%TAG%_d_wkvcb_%WSSTEPS%.txt
set RWKV_QAT_SHIFT_PQ=%SRCREL%/%TAG%_d_shiftcb_%WSSTEPS%.txt
set RWKV_QAT_PQ_LEARN=
set RWKV_QAT_SHIFT_PQ_LEARN=

REM ================= PHASE B: full VAL-half eval, quantized with the learned catalogs =========
.venv\Scripts\python.exe scratchpad/write_eval_toml.py %SRCREL% %TAG%_d %DIR%\%TAG%_eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 7500 > "%DIR%\%TAG%_etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo CBLEARN TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 >> "%LOG%"
  endlocal & exit /b 24
)
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\%TAG%_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\%TAG%_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\%TAG%_eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo CBLEARN EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 >> "%LOG%"
  endlocal & exit /b 25
)
findstr /C:"[QAT-SHIFT-PQ] loaded" "%DIR%\%TAG%_eval1_%STAMP%.log" "%DIR%\%TAG%_eval2_%STAMP%.log" scratchpad\eval_shards\shard_*.log >nul
if not %ERRORLEVEL%==0 (
  echo CBLEARN EVAL_NOT_QUANTIZED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_41 >> "%LOG%"
  endlocal & exit /b 41
)
echo CBLEARN EVAL_OK %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
