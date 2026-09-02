@echo off
REM =========================================================================================
REM TEACHER-114 SCREEN. How much does the d=128 KD teacher lose if it stops seeing
REM `scaled_state`? That is exactly what happens when its input projection is re-laid-out into
REM the 114-column RWKV_ID_FEATURES layout, which DROPS that column.
REM
REM Why it matters: featB ran KD-OFF because the teacher cannot forward 114 dims at all, and KD
REM is worth ~0.0019 to this lineage (iters 32/35/39/45). If the teacher survives losing
REM scaled_state, the features phase keeps KD; if not, KD-off is the honest choice.
REM
REM Two arms, one variable:
REM   A  the teacher as it has always run
REM   B  the same teacher with RWKV_ZERO_FEATURES=22 (scaled_state)
REM B minus A is the cost of the re-lay-out, measured rather than argued.
REM
REM ARCH VIA ENV, NOT A FILE SWAP. run_base5k_eval.cmd copies architecture_old_d128.py over
REM rwkv/architecture.py and restores it after; a crash mid-run leaves the tree on the wrong
REM architecture. RWKV_ARCH_MODULE exists specifically to replace that footgun, and generation 4
REM is building in this same tree right now.
REM
REM No angle brackets or arrows in REM lines: cmd parses redirection before it honours REM.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\teacher_114
set LOG=%DIR%\t114.log
set STAMP=%RANDOM%%RANDOM%
set PY=.venv\Scripts\python.exe
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6

REM The override file owns the whole config, so the champion's capacity hooks must be CLEARED --
REM not merely left unset, so an inherited value cannot reach the teacher.
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py
set RWKV_N_HEADS=
set RWKV_HEAD_DIM=
set RWKV_ID_FEATURES=
set RWKV_INTERLEAVE=
set RWKV_STREAM_ORDER=
set RWKV_GRU_HEAD=
set RWKV_STRIP_CMIX=
set RWKV_STRIP_L0_VLORA=
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_PQ=
set RWKV_QAT_SHIFT_PQ=
set RWKV_QAT_SHIFT_SCOPE=
set RWKV_QAT_FUSED=
set RWKV_EVAL_PAVA=
set RWKV_PAVA_LAMBDA=
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0

if not exist "%DIR%" mkdir "%DIR%"
echo ===== T114 SCREEN START %DATE% %TIME% ===== > "%LOG%"

REM ---- ARM A: the teacher as it has always run ----
REM Stale result jsonls MUST go: get_result SKIPS users already present, so leftovers are
REM silently reused and the arm scores someone else's numbers.
set RWKV_ZERO_FEATURES=
if exist "result\RWKV-t114a.jsonl" del /q "result\RWKV-t114a.jsonl"
if exist "result\RWKV-P-t114a.jsonl" del /q "result\RWKV-P-t114a.jsonl"
%PY% -u -m rwkv.get_result --config rwkv/get_result_config_t114a.toml > "%DIR%\armA_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo T114 ARM_A_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_11 %DATE% %TIME% >> "%LOG%"
  exit /b 11
)
echo T114 ARM_A_OK %TIME% >> "%LOG%"

REM ---- ARM B: the same teacher WITHOUT scaled_state ----
set RWKV_ZERO_FEATURES=22
if exist "result\RWKV-t114b.jsonl" del /q "result\RWKV-t114b.jsonl"
if exist "result\RWKV-P-t114b.jsonl" del /q "result\RWKV-P-t114b.jsonl"
%PY% -u -m rwkv.get_result --config rwkv/get_result_config_t114b.toml > "%DIR%\armB_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo T114 ARM_B_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_12 %DATE% %TIME% >> "%LOG%"
  exit /b 12
)
REM The mask must have REACHED the model. A silent no-op would make the arms identical and the
REM screen would report "no cost" for a treatment that never happened -- the vacuous-green shape.
REM The exact banner is `[feat-mask] zeroing input feature dims [22] (train AND eval)`
REM (srs_model.py:528). Checked against the source, because a guard written from memory that
REM greps for a string the code never prints fails EVERY correct run.
findstr /C:"[feat-mask] zeroing input feature dims [22]" "%DIR%\armB_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo T114 MASK_NOT_CONFIRMED -- arm B did not zero dim 22 %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_45 %DATE% %TIME% >> "%LOG%"
  exit /b 45
)
REM ...and arm A must NOT carry it, or the two arms are the same experiment.
findstr /C:"[feat-mask]" "%DIR%\armA_%STAMP%.log" >nul
if %ERRORLEVEL%==0 (
  echo T114 CONTROL_CONTAMINATED -- arm A also masked %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_46 %DATE% %TIME% >> "%LOG%"
  exit /b 46
)
echo T114 ARM_B_OK %TIME% >> "%LOG%"

REM ---- the verdict ----
%PY% scratchpad/teacher_114/t114_verdict.py >> "%LOG%" 2>&1

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
