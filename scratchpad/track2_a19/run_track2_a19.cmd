@echo off
REM ============================================================================
REM TRACK-2 A19: the A18 champion + the three track-1 ALGORITHMIC wins
REM (Andrew 2026-07-26: "Let's accept A18 and continue track 1 with it. Add the
REM algorithmic improvements (PAVA, GRU n_head=3, Muon) to it").
REM   PAVA  = RWKV_PAVA_LAMBDA=0.1 + RWKV_PROBE_DENSITY=0.08   (track-1 iter 23)
REM   GRU   = RWKV_GRU_HEAD 2 -> 3                             (track-1 iter 26)
REM   Muon  = RWKV_MUON=1 / LR 0.02 / MOMENTUM 0.95            (track-1 iter 29)
REM Arch module UNCHANGED (A18's d80_lora4) -> params +~966 only.
REM GATE = ordinary ACCURACY iter vs A18 (0.299302 / 0.268390), NOT the ratio
REM gate: both modes >= 0.0001 after 4-dp rounding AND p < 0.0001, VAL 5001-7500.
REM Bundled on purpose (each independently validated at d=32; together = the
REM iter-29 recipe; ~6 h vs ~20 h). De-bundle precedent if it regresses = A10/A11.
REM STEP 0.5 = 40-step sanity smoke: probe rows inflate batch rows ~30% at
REM MAX=32768, so this also proves the VRAM envelope before committing 4 h.
REM VPRUNE_MIN_STEP=6000 (different optimizer = different early dynamics, as in
REM iter 29, which trailed val all run and still won eval decisively).
REM LOG HYGIENE: the control log is written ONLY by this cmd; every python phase
REM redirects to its own STAMPED sublog. Launch DETACHED via detach.ps1 (CRLF!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a19
set LOG=%DIR%\track2_a19.log
set STAMP=%RANDOM%%RANDOM%
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
set RWKV_MUON=1
set RWKV_MUON_LR=0.02
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_VPRUNE_REF=optimization/champion_5k_track2.json
set RWKV_VPRUNE_DELTA_AHEAD=0.004
set RWKV_VPRUNE_DELTA_IMM=0.006
set RWKV_VPRUNE_MIN_STEP=6000
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25

echo ===== track2_a19 START %DATE% %TIME% ===== > "%LOG%"

echo === STEP 0.5: 40-step E2E sanity (PAVA probes + GRU N=3 + Muon wiring, VRAM) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a19/track2_a19_ws.toml > "%DIR%\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
set RWKV_MAX_STEPS=
echo SANITY OK %TIME% >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/track2_a19/track2_a19_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/track2_a19/t2a19_grad_stats_ws.json
del /Q scratchpad\track2_a19\track2_a19_ws_trace.jsonl scratchpad\track2_a19\track2_a19_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, A18 arch + PAVA + GRU3 + Muon, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a19/track2_a19_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if %ERRORLEVEL%==42 (
  echo DONE_EXIT_PRUNED_42 %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=scratchpad/track2_a19/t2a19_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=32768) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/track2_a19 t2a19ws t2a19d scratchpad/track2_a19/track2_a19_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 32768 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/track2_a19/track2_a19_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-track2_a19.jsonl result\RWKV-P-track2_a19.jsonl result\RWKV-track2_a19-s0.jsonl result\RWKV-P-track2_a19-s0.jsonl result\RWKV-track2_a19.nanskip.jsonl result\RWKV-track2_a19-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/track2_a19 t2a19d scratchpad/track2_a19/track2_a19_eval.toml RWKV-track2_a19 RWKV-P-track2_a19 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo === EVAL (single process, d=80, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a19/track2_a19_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: paired vs CHAMPION A18 (val half) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-track2_a19.jsonl --cand-imm result/RWKV-P-track2_a19.jsonl --champ-ahead result/RWKV-track2_a18.jsonl --champ-imm result/RWKV-P-track2_a18.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
