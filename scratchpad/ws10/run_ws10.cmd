@echo off
REM ws10 -- the ENDGAME's shared 10-epoch WS phase. Andrew 2026-09-08: "keep current model size and
REM do as you suggested: train WS, then experiment with the decay stage."
REM
REM WHAT THIS IS. One expensive WS run (109,350 steps ~= 34 h) whose checkpoint every later decay-only
REM experiment branches from. A full 10+2 iteration is ~43 h; a decay-only branch off this checkpoint
REM is decay 6.2 h + eval 4.2 h ~= 10.4 h, i.e. today's price in the 10+2 regime. The run is not
REM overhead: its 2-epoch plain decay IS arm 1 of the endgame (the plain-10x number).
REM
REM SINGLE VARIABLE vs realcyc: EPOCHS 1 -> 10 in ws.toml. The env below is realcyc's, byte-identical.
REM
REM RESUME. Over ~34 h a crash or reboot is the one real failure mode (CLAUDE.md), so this loops:
REM train -> if the final checkpoint exists, done -> else if any checkpoint exists, make_resume.py and
REM go again with RWKV_RESUME_SKIP_GROUPS=1. VALIDATE_EVERY=1000 gives 109 resume points, so a crash
REM costs <=1000 steps (~17 min) instead of the run. Attempts are capped so a hard-failing config
REM cannot spin forever.
REM WARNING the resumed tail's dropout draws differ from an uninterrupted run (weights/optimizer are
REM exact) -- statistically equivalent, not bit-reproducible. Carry that into any later comparison.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ws10
set LOG=%DIR%\ws10.log
set STAMP=%RANDOM%%RANDOM%
set TAG=ws10
set STEPS=109350
set CFG=scratchpad/ws10/ws.toml

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
set RWKV_MUON_INCLUDE_LORA=1
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
set RWKV_ZERO_FEATURES=
set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_id5
set RWKV_LABEL_FILTER_DB=F:/rwkv_lmdb/label_filter_db_id_e2s
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

if not exist "%DIR%" mkdir "%DIR%"
echo ===== ws10 START %DATE% %TIME% ===== > "%LOG%"
set ATTEMPT=0

:trainloop
set /a ATTEMPT=%ATTEMPT%+1
if %ATTEMPT% GTR 40 (
  echo %TAG% TOO_MANY_ATTEMPTS %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_29 %DATE% %TIME% >> "%LOG%"
  exit /b 29
)
echo --- attempt %ATTEMPT% using %CFG% %DATE% %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config %CFG% >> "%DIR%\ws_%STAMP%.log" 2>&1
echo --- train_rwkv rc %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
if exist "%DIR%\w10_ws_%STEPS%.pth" goto wsdone
REM Not finished. Resume from the newest checkpoint if there is one; otherwise this is a launch
REM failure, not a crash, and looping would just repeat it.
if not exist "%DIR%\w10_ws_1000.pth" (
  echo %TAG% WS_NO_CHECKPOINT -- launch failure, not a crash %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_21 %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
.venv\Scripts\python.exe scratchpad/make_resume.py %DIR% w10_ws scratchpad/ws10/ws.toml scratchpad/ws10/ws_resume.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% MAKE_RESUME_FAILED %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_26 %DATE% %TIME% >> "%LOG%"
  exit /b 26
)
set CFG=scratchpad/ws10/ws_resume.toml
set RWKV_RESUME_SKIP_GROUPS=1
goto trainloop

:wsdone
REM The run must have used the width it claims: the one check that catches RWKV_ID_FEATURES /
REM RWKV_REAL_CYCLES failing to reach the fetch workers.
findstr /C:"Trainable parameters: 563652" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% WRONG_PARAM_COUNT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_33 %DATE% %TIME% >> "%LOG%"
  exit /b 33
)
echo %TAG% WS_OK after %ATTEMPT% attempt(s) %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
