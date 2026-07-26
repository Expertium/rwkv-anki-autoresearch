@echo off
REM ============================================================================
REM RESEARCH ITER 32 -- full-run distillation from the d=128 2.76M teacher.
REM
REM Motivation, measured today on the identical 2500 val-half users with identical
REM `size`: teacher 0.294612 ahead / 0.263561 imm vs iter-31 champion 0.298909 /
REM 0.267637. We are +0.0043 / +0.0041 BEHIND the model we are shrinking -- ~10x a
REM single iteration's gate. (That teacher number was NOT pending: it was measured
REM 2026-07-03 by run_base5k_eval.cmd on 5001-10000; CLAUDE.md's "PENDING" was stale.)
REM
REM Family = DISTILLATION, standing at 0/1 and NOT closed. Iter 10's attempt was
REM warmup-ONLY KD onto the d=32 trunk, which the lesson bank records as DATA-limited;
REM soft targets cannot fix a data limit. A18 is capacity-limited instead (width ladder
REM closed at an accuracy floor; the second LoRA halving flipped sign), which is the
REM regime KD is for. Record the outcome under distillation, not early-training-
REM intervention -- the mis-filing is why the scoreboard shows the family nowhere.
REM
REM VARIANT: classic fixed-alpha KD (RWKV_KD_ALPHA=0.5) over ALL 22,346 WS steps.
REM Decay runs on hard labels only.
REM
REM TWO DELIBERATE DEPARTURES from the iter-31 template, both from the lesson bank:
REM  1. VPRUNE IS OFF. The scope rule (decay_ratio_0p1 FALSE-KILL audit) says prune only
REM     candidates at MATCHED regularization vs the reference. KD replaces the training
REM     target wholesale, and validation stays on HARD labels, so a KD student is
REM     structurally expected to look worse on val early. That is precisely the
REM     sign-biased situation that killed a config which went on to WIN both modes.
REM     Cost of being wrong here = one run; cost of a false kill = the idea.
REM  2. A SMOKE DUMP RUNS FIRST and is gated on a semantic check, not just an exit code.
REM     The student verifies ALIGNMENT (labels checksum, exit 43) but nothing verifies
REM     the dumped tensors are teacher outputs at all -- a wrong arch/flag combination
REM     yields perfectly aligned garbage. check_dump.py tests that p_curve is inside
REM     (0,1) and p_imm_all sums to 1, and projects the full dump's disk footprint from
REM     the smoke files, since per-step size depends on the padded B*T.
REM
REM Launch DETACHED via detach.ps1 with an ABSOLUTE path (Win32_Process.Create starts in
REM System32; a relative path exits instantly and silently -- cost that one launch today).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter32_kd
set LOG=%DIR%\iter32_kd.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_iter32
set SMOKE=C:\rwkv_kd_dump\t128_smoke
set WSSTEPS=22346
set WAITLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\eval_pava\mode2_diag.log

echo ===== iter32_kd START %DATE% %TIME% ===== > "%LOG%"

REM /B anchors the match to line start; terminal lines begin with the token and prose never
REM does. An unanchored findstr matched a log line that merely MENTIONED it earlier today.
echo === WAIT for the mode-2 diagnostic to finish %TIME% === >> "%LOG%"
:waitloop
findstr /B /C:"DONE_EXIT_" "%WAITLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto ready
timeout /t 120 /nobreak >nul
goto waitloop
:ready
echo GPU free %TIME% >> "%LOG%"

REM ------------------------------------------------------- TEACHER (d=128, forward only)
setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM The teacher is the ORIGINAL model: it must run under its OWN architecture and none of
REM the trunk's flags. Every student-side flag below is cleared EXPLICITLY rather than left
REM unset, because this .cmd may be re-run in a shell where they linger.
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py
set RWKV_GRU_HEAD=
set RWKV_PAVA_LAMBDA=
set RWKV_MUON=
set RWKV_MUON_LR=
set RWKV_MUON_MOMENTUM=
set RWKV_NO_AHEAD_RESIDUAL=
set RWKV_STRIP_L0_VLORA=
set RWKV_ZERO_FEATURES=
set RWKV_STATE_CLAMP_TAU=
set RWKV_STATE_CLAMP_WINDOW=
set RWKV_STRIP_CMIX=
set RWKV_VPRUNE_REF=
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=
REM ...but the two PROBE vars are DATA-side, not model-side: they change the row layout, and
REM the student inserts probes at density 0.08. Teacher and student must agree or every step
REM fails the shape check. The teacher has no PAVA and does not need it -- probe rows are
REM inserted by prepare_batch and are skip rows; the teacher simply predicts on them too.
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_KD_TEACHER=pretrain/RWKV_trained_on_101_4999.pth

echo === SMOKE DUMP: 5 steps (proves arch+flags+VRAM, projects disk) %TIME% === >> "%LOG%"
if exist "%SMOKE%" rmdir /s /q "%SMOKE%"
mkdir "%SMOKE%" 2>nul
set RWKV_KD_DUMP_OUT=%SMOKE%
set RWKV_KD_STEPS=5
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter32_kd/kd_dump.toml > "%DIR%\smoke_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SMOKEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
.venv\Scripts\python.exe scratchpad/iter32_kd/check_dump.py "%SMOKE%" --expect-steps 5 --planned-steps %WSSTEPS% --max-gb 60 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SMOKECHECK_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 10
)
rmdir /s /q "%SMOKE%"
echo SMOKE OK %TIME% >> "%LOG%"

echo === FULL TEACHER DUMP: %WSSTEPS% steps %TIME% === >> "%LOG%"
if exist "%DUMP%" rmdir /s /q "%DUMP%"
mkdir "%DUMP%" 2>nul
set RWKV_KD_DUMP_OUT=%DUMP%
set RWKV_KD_STEPS=%WSSTEPS%
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter32_kd/kd_dump.toml > "%DIR%\dump_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DUMPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 11
)
if not exist "%DUMP%\step_%WSSTEPS%.pt" (
  echo DONE_EXIT_DUMPSHORT ^(last step file missing^) %DATE% %TIME% >> "%LOG%"
  exit /b 12
)
echo DUMP OK %TIME% >> "%LOG%"
endlocal

REM ---------------------------------------------------------- STUDENT (iter-31 recipe + KD)
setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.02
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
REM VPRUNE DELIBERATELY OFF -- see the header. KD is not at matched regularization vs the
REM champion and validation scores HARD labels, so the prune would be sign-biased against it.
set RWKV_VPRUNE_REF=
REM rsplit(":",1) parses the spec, so the drive colon in the path is safe.
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.5

echo === STEP 0.5: 40-step E2E sanity (KD mixing + checksum alignment) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter32_kd/iter32_kd_ws.toml > "%DIR%\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
set RWKV_MAX_STEPS=
findstr /C:"[kd-mix] step 1:" "%DIR%\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_KDNOTACTIVE ^(no kd-mix line in the sanity log^) %DATE% %TIME% >> "%LOG%"
  exit /b 14
)
echo SANITY OK ^(KD confirmed active^) %TIME% >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/iter32_kd/iter32_kd_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/iter32_kd/iter32_grad_stats_ws.json
del /Q scratchpad\iter32_kd\iter32_kd_ws_trace.jsonl scratchpad\iter32_kd\iter32_kd_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (KD alpha=0.5 fixed, vprune OFF) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter32_kd/iter32_kd_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
if not exist scratchpad\iter32_kd\iter32ws_%WSSTEPS%.pth (
  echo DONE_EXIT_WSNOCKPT %DATE% %TIME% >> "%LOG%"
  exit /b 15
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/iter32_kd/iter32_grad_stats_decay.json
REM !! CLEAR KD BEFORE DECAY: decay re-seeds to 12345 and REPRODUCES the epoch-0 batch stream,
REM so the per-step checksum would pass while the targets belong to different reviews.
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter32_kd iter32ws iter32d scratchpad/iter32_kd/iter32_kd_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY (hard labels) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter32_kd/iter32_kd_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-iter32_kd.jsonl result\RWKV-P-iter32_kd.jsonl result\RWKV-iter32_kd-s0.jsonl result\RWKV-P-iter32_kd-s0.jsonl result\RWKV-iter32_kd.nanskip.jsonl result\RWKV-iter32_kd-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter32_kd iter32d scratchpad/iter32_kd/iter32_kd_eval.toml RWKV-iter32_kd RWKV-P-iter32_kd 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo === EVAL %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter32_kd/iter32_kd_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"
endlocal

echo === GATE: paired vs CHAMPION iter31 (val half, unrectified = primary) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter32_kd.jsonl --cand-imm result/RWKV-P-iter32_kd.jsonl --champ-ahead result/RWKV-iter31_algo.jsonl --champ-imm result/RWKV-P-iter31_algo.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"

echo === REFERENCE: the same candidate vs the d=128 TEACHER (how much of the gap closed) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter32_kd.jsonl --cand-imm result/RWKV-P-iter32_kd.jsonl --champ-ahead result/RWKV-base5k.jsonl --champ-imm result/RWKV-P-base5k.jsonl >> "%DIR%\gate_%STAMP%.log" 2>&1

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
