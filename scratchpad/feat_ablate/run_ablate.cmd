@echo off
REM =========================================================================================
REM DOES `ahead` USE THE NEW TIMESTAMP FEATURES AT ALL?
REM
REM Andrew 2026-09-02, on featB: "Wait, all new features only moved ahead by 0.0003?" The
REM headline understates it -- featB pays the end-to-start penalty inside the same bundle, so
REM features-only is ~+0.00053 ahead, which beats 8 of the 9 accepted iterations. But the 5:1
REM imm-to-ahead ratio is real and the question behind it is fair.
REM
REM This measures RELIANCE without retraining: take featB's own checkpoint, zero groups of the
REM 23 new columns at the input, and see how much each mode degrades. featB's EXISTING per-user
REM results are the control, which is why the user range must match exactly and why only three
REM arms run instead of six.
REM
REM   abl_all     all 23 new columns          how much of featB rests on them at all
REM   abl_clock   the 10 fine-grained TIMING  tod/dow/doy/is_weekend/t_since_any_review
REM   abl_struct  the 13 always-defined ones  tenure, ages, creation batch, deck, sibling
REM
REM ★ IT TESTS P2'S REFUTATION DIRECTLY. featB's verdict showed the gain is NOT concentrated in
REM same-day users, which pointed away from the clock columns and towards the always-defined
REM ones. If that reading is right, abl_struct costs MORE than abl_clock. If it is backwards,
REM the same-day analysis was measuring something else and should not steer feature work.
REM
REM ⚠ AN INFERENCE-TIME ABLATION MEASURES RELIANCE, NOT VALUE. A retrained model recovers some
REM of it -- the delta-rule caveat, and the teacher-114 caveat. Read it as "does ahead use these
REM at all", not as "this is what they are worth".
REM
REM THREE INLINE ARMS, NOT A SUBROUTINE. A first draft used `call :arm`; preflight_runner flagged
REM every DONE_EXIT_ inside it as out of scope. That is a false positive for the call pattern --
REM but the guard has caught real bugs, cmd subroutines are where the %~N / exit /b traps live,
REM and no other runner here uses one. Inlining costs three near-identical blocks and buys a
REM verifiable file. The three blocks were diffed against each other before arming.
REM
REM No angle brackets or arrows in REM lines: cmd parses redirection before it honours REM.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_ablate
set LOG=%DIR%\ablate.log
set STAMP=%RANDOM%%RANDOM%
set PY=.venv\Scripts\python.exe
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6

REM featB's eval env, reproduced exactly -- anything that differs would confound the ablation.
set RWKV_ID_FEATURES=1
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_EVAL_PAVA=1
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ZERO_FEATURES=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

set CLOCK=tod_sin,tod_cos,tod_dev_sin,tod_dev_cos,dow_sin,dow_cos,doy_sin,doy_cos,is_weekend,scaled_t_since_any_review
set STRUCT=scaled_user_tenure,scaled_creation_to_first_review,scaled_deck_age_at_review,card_predates_deck,is_default_deck,scaled_deck_depth,scaled_creation_batch_1min,scaled_creation_batch_1h,scaled_creation_batch_1d,scaled_creation_batch_pos_1h,is_default_preset,scaled_sibling_gap,card_predates_first_review

if not exist "%DIR%" mkdir "%DIR%"
echo ===== FEATURE ABLATION START %DATE% %TIME% ===== > "%LOG%"

REM ---- ARM 1: abl_all, all 23 ----
REM Stale result jsonls MUST go: get_result SKIPS users already present, so leftovers are
REM silently reused and the arm scores a previous arm's numbers.
set RWKV_ABLATE_FEATURES=%CLOCK%,%STRUCT%
if exist "result\RWKV-abl_all.jsonl" del /q "result\RWKV-abl_all.jsonl"
if exist "result\RWKV-P-abl_all.jsonl" del /q "result\RWKV-P-abl_all.jsonl"
%PY% -u -m rwkv.get_result --config scratchpad/feat_ablate/eval_abl_all.toml > "%DIR%\abl_all_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABLATE abl_all_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_11 %DATE% %TIME% >> "%LOG%"
  exit /b 11
)
REM The mask must have REACHED the model with the RIGHT COUNT. An unknown name raises by design,
REM but a silently EMPTY variable would make the arm identical to the control and report "these
REM features do not matter" for a treatment that never happened.
%PY% scratchpad/feat_ablate/check_mask_count.py "%DIR%\abl_all_%STAMP%.log" 23 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABLATE abl_all_WRONG_DIM_COUNT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_46 %DATE% %TIME% >> "%LOG%"
  exit /b 46
)
echo ABLATE abl_all_OK %TIME% >> "%LOG%"

REM ---- ARM 2: abl_clock, the 10 timing columns ----
set RWKV_ABLATE_FEATURES=%CLOCK%
if exist "result\RWKV-abl_clock.jsonl" del /q "result\RWKV-abl_clock.jsonl"
if exist "result\RWKV-P-abl_clock.jsonl" del /q "result\RWKV-P-abl_clock.jsonl"
%PY% -u -m rwkv.get_result --config scratchpad/feat_ablate/eval_abl_clock.toml > "%DIR%\abl_clock_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABLATE abl_clock_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_12 %DATE% %TIME% >> "%LOG%"
  exit /b 12
)
%PY% scratchpad/feat_ablate/check_mask_count.py "%DIR%\abl_clock_%STAMP%.log" 10 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABLATE abl_clock_WRONG_DIM_COUNT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_46 %DATE% %TIME% >> "%LOG%"
  exit /b 46
)
echo ABLATE abl_clock_OK %TIME% >> "%LOG%"

REM ---- ARM 3: abl_struct, the 13 always-defined columns ----
set RWKV_ABLATE_FEATURES=%STRUCT%
if exist "result\RWKV-abl_struct.jsonl" del /q "result\RWKV-abl_struct.jsonl"
if exist "result\RWKV-P-abl_struct.jsonl" del /q "result\RWKV-P-abl_struct.jsonl"
%PY% -u -m rwkv.get_result --config scratchpad/feat_ablate/eval_abl_struct.toml > "%DIR%\abl_struct_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABLATE abl_struct_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_13 %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
%PY% scratchpad/feat_ablate/check_mask_count.py "%DIR%\abl_struct_%STAMP%.log" 13 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo ABLATE abl_struct_WRONG_DIM_COUNT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_46 %DATE% %TIME% >> "%LOG%"
  exit /b 46
)
echo ABLATE abl_struct_OK %TIME% >> "%LOG%"

REM ---- the verdict ----
%PY% scratchpad/feat_ablate/ablate_verdict.py >> "%LOG%" 2>&1

REM Terminal marker BEFORE endlocal: endlocal restores the pre-setlocal environment, so %LOG%
REM would expand to empty and the marker would be appended to "" instead.
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
