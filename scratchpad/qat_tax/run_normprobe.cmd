@echo off
REM ===========================================================================================
REM NORM PROBE -- what is the 1-bit norm quantizer actually WORTH in logloss? 3 arms, 10 users
REM each, no training. ~12 min of GPU.
REM
REM WHY THIS FIRST, before implementing per-stream norm ranges. The reconstruction ladder says the
REM 1-bit norm is 19% (card) / 31% (note) of the error, and that a FITTED range would cut note's
REM norm error 2.9x. But this week has twice shown reconstruction error does NOT map proportionally
REM to logloss -- the shift catalog's 9% error gap was worth ~0.0004, while the WKV catalog's
REM collapse was worth ~0.006. So measure the norm's logloss weight before building anything.
REM
REM   normfree : refit catalogs, norm quant OFF (exact fp32 norms)  -> UPPER BOUND on any norm fix
REM   norm1    : refit catalogs, norm quant ON at 1 bit             -> today's deploy config
REM   normfree - norm1 = the ENTIRE budget available to items 1-2 (per-stream ranges, learnable
REM   levels). If that gap is small, both are dead regardless of how good the reconstruction looks.
REM   norm2    : norm quant at 2 bits -> what BUYING a bit would give, for comparison. NOT in-budget
REM   (it costs +1 bit/head/layer = ~+5% card state) but it brackets what a better 1-bit scheme
REM   could aspire to: a perfect 1-bit scheme cannot beat an honest 2-bit one.
REM
REM All arms evaluate the SAME plain iter-45 checkpoint, so this is pure PTQ attribution.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set SRCREL=scratchpad/iter45_kddecay
set LOG=%DIR%\normprobe.log
set STAMP=%RANDOM%%RANDOM%

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
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_c80_b10.txt
set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_c80_m2b12.txt
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_FUSED=1
set RWKV_NO_JIT=1
set RWKV_EVAL_PAVA=1

echo ===== NORM PROBE START %DATE% %TIME% ===== > "%LOG%"
call :arm normfree 0
call :arm norm1 1
call :arm norm2 2
echo NORMPROBE_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0

REM ---- :arm <TAG> <NORM_BITS ; 0 = off> ------------------------------------------------------
:arm
setlocal
set TAG=%~1
set NB=%~2
if "%NB%"=="0" (set RWKV_QAT_NORM_BITS=) else (set RWKV_QAT_NORM_BITS=%NB%)
echo --- arm %TAG%: norm_bits=%NB% (0 = exact fp32 norms) --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/qat_tax/assert_qat_live.py >> "%LOG%" 2>&1
.venv\Scripts\python.exe scratchpad/write_eval_toml.py %SRCREL% i45_d %DIR%\np_%TAG%.toml RWKV-np_%TAG% RWKV-P-np_%TAG% 5001 5010 >> "%LOG%" 2>&1
.venv\Scripts\python.exe -u -m rwkv.get_result --config %DIR%\np_%TAG%.toml > "%DIR%\np_%TAG%_%RANDOM%.log" 2>&1
echo ARM_%TAG%_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"
endlocal & exit /b 0
