@echo off
REM ===========================================================================================
REM PHASE 1 SPEEDUP, ROUND 3: WHICH SOURCE LINES emit the indexing ops?
REM
REM Rounds 1-2 established, on a quiet machine:
REM   * the step is GPU-BOUND -- 1,206-1,369 ms/step of GPU kernel time, not the 237 ms
REM     DISPATCH_PLAN records, so CUDA graphs target a cost that is not the bottleneck;
REM   * aten::_index_put_impl_ 18.95% + indexing_backward_kernel 18.67% = 37.6% of GPU time;
REM   * RWKV_DETERMINISTIC=0 is +30.9% throughput and cuts GPU kernel time 1,206 -> 787 ms/step,
REM     i.e. the determinism tax IS the indexing cost, about 419 ms/step.
REM
REM WHAT IS STILL UNKNOWN, AND WHY GUESSING WOULD BE WRONG. Reading the code shows the two obvious
REM sites are ALREADY optimized: `time_shift_gather` is an index_select whose backward is a
REM row-wise index_add (not gather's element-wise scatter-add), and the interleave scatter uses
REM `index_copy`, the collision-free form. So the remaining cost is the BACKWARD of those
REM index_selects, and which of the many call sites dominates is not visible from a
REM bucket-by-kernel-name summary. `_dump_stack_attribution` exists for exactly this and was
REM written in July, when indexing_backward_kernel was 13.5% of GPU time; it is 18.67% now.
REM
REM ⚠ RWKV_PROFILE_STACK=1 SKEWS TIMINGS -- with_stack has real overhead. This is an ATTRIBUTION
REM run. Do not quote its ms/step against rounds 1-2; only the call-site ranking is the output.
REM
REM det stays at 1: the goal is to locate the cost in the configuration we actually ship, not in
REM the fast one we cannot use.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\dispatch
set LOG=%DIR%\attrib.log
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

set RWKV_PROFILE_STEP=150
set RWKV_PROFILE_COUNT=3
set RWKV_PROFILE_STACK=1

echo ===== ATTRIBUTION RUN START %DATE% %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config "%CFG%" > "%DIR%\attrib_%STAMP%.log" 2>&1
echo   exit=%ERRORLEVEL% %TIME% >> "%LOG%"
echo   raw log: attrib_%STAMP%.log >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
