@echo off
REM ===========================================================================================
REM Dump OUR champion's predictions over the SAME batch stream the d=128 teacher dump used, so
REM the two can be compared offline. ~200 steps, forward-only, minutes of GPU.
REM
REM PURPOSE (Andrew 2026-08-13): decide whether the TEACHER's imm carries information about our
REM ahead's error that OUR OWN imm does not. iter 46 already tested our-own-imm -> our-ahead and
REM got a tie, and its stated cause was that the teacher shared the trunk and forward pass. The
REM d=128 teacher's imm is a genuinely different function, so iter 46 does not settle it -- but
REM if the teacher's imm turns out to be nearly a copy of ours, the branch closes for minutes of
REM GPU instead of a 9 h decay run.
REM
REM Uses the EXISTING dump machinery (RWKV_KD_DUMP_OUT + RWKV_KD_TEACHER), pointing "teacher" at
REM OUR champion -- it walks the same deterministic batch stream and does an eval-mode no_grad
REM forward with no optimizer/checkpoint side effects. RWKV_KD_DUMP_LABELS=1 additionally stores
REM the targets + row masks, which the d=128 dump lacks; the batch stream is identical, so the
REM labels captured here are the labels the teacher saw, and `labels_sum` proves it per step.
REM
REM ⚠ The batch stream must match EXACTLY: same db, same MAX, same seeds. Those are copied from
REM the champion recipe below and must not be "tidied".
REM ⚠ NO QAT env here: we want our model's real predictions, matching how the teacher dump was made.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\immcorr_dump.log
set OUTDIR=C:\rwkv_kd_dump\ours_i45_immcorr

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
set RWKV_DROPOUT_SCALE=0.5

set RWKV_KD_DUMP_OUT=%OUTDIR%
set RWKV_KD_TEACHER=scratchpad/iter45_kddecay/i45_d_10935.pth
set RWKV_KD_STEPS=200
set RWKV_KD_DUMP_LABELS=1

echo ===== IMMCORR DUMP START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config %DIR%\immcorr_dump.toml >> "%LOG%" 2>&1
REM ⚠ GATE ON THE EXIT CODE. The first version echoed DONE_EXIT_0 unconditionally, so a toml parse
REM error (DUMP_EXIT_1) still reported success to the caller and the chain marched on to an
REM analysis with no data. "Gate every phase on exit codes AND artifacts" applies to the small
REM helper runners too, not just the long training chains.
if not %ERRORLEVEL%==0 (
  echo DUMP FAILED %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 1
)
dir /b "%OUTDIR%\step_*.pt" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DUMP PRODUCED NO STEP FILES %DATE% %TIME% >> "%LOG%"
  endlocal & exit /b 2
)
echo IMMCORRDUMP_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
