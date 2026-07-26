@echo off
REM ============================================================================
REM RESEARCH ITER 31 -- the FIRST iteration of the MERGED lineage.
REM Andrew 2026-07-26: "Let's accept A18 and continue track 1 with it. Add the
REM algorithmic improvements (PAVA, GRU n_head=3, Muon) to it" + "it shouldn't
REM be called A19, it should be iter 31 (first table in research 5k)".
REM Base = the track-2 A18 champion trunk (d=80 = 5 heads x K=16, LoRA 4,
REM 557,246 params, 4.95x below 2.76M). Added, from track 1:
REM   PAVA  = RWKV_PAVA_LAMBDA=0.1 + RWKV_PROBE_DENSITY=0.08   (iter 23)
REM   GRU   = RWKV_GRU_HEAD 2 -> 3                             (iter 26)
REM   Muon  = RWKV_MUON=1 / LR 0.02 / MOMENTUM 0.95            (iter 29)
REM Arch module UNCHANGED (A18's d80_lora4) -> 558,212 params (+966).
REM GATE = ordinary research accuracy gate vs A18 (0.299302 / 0.268390): both
REM modes >= 0.0001 after 4-dp rounding AND p < 0.0001, on VAL 5001-7500.
REM Bundled on purpose (each independently validated at d=32; together = the
REM iter-29 recipe; ~10 h vs ~30 h). De-bundle precedent if it regresses = A10/A11.
REM SPEED NOTE (measured on the first launch, 2026-07-26 11:45): this recipe runs
REM ~0.7-0.9 steps/s vs A18's 1.86 -- probes add ~30% rows, PAVA runs eager, Muon
REM adds Newton-Schulz. Expect WS ~7-8 h, total ~10 h. Not a bug.
REM STEP 0.5 = 40-step sanity smoke (also proves the VRAM envelope: 8.8 GB peak
REM reserved measured on the first launch, fits 12 GB).
REM VPRUNE_MIN_STEP=6000 (different optimizer = different early dynamics, as in
REM iter 29, which trailed val all run and still won eval decisively).
REM LOG HYGIENE: the control log is written ONLY by this cmd; every python phase
REM redirects to its own STAMPED sublog. Launch DETACHED via detach.ps1 (CRLF!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM Parked behind iter 32. /B anchored so a line merely MENTIONING the token cannot fire it.
set WAITLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter32_kd\iter32_kd_v2.log
:waitloop
findstr /B /C:"DONE_EXIT_" "%WAITLOG%" >nul 2>&1
if %ERRORLEVEL%==0 goto ready
timeout /t 300 /nobreak >nul
goto waitloop
:ready
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter33_dur
set LOG=%DIR%\iter33_dur.log
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
REM iter 33: EVERY eligible review gets probes, so the duration-zeroed probe path carries
REM the whole ahead objective (76.5%% coverage; the 23.5%% residual is a card first in-chunk
REM review, which has no paired query row so probes are excluded by design).
set RWKV_PROBE_DENSITY=1.0
set RWKV_AHEAD_PROBE_ONLY=1
REM Andrew 2026-07-26: the counterfactual probes carry ZEROED current-row duration,
REM not the train-median constant. Rationale: at deploy Anki must draw the four
REM intervals BEFORE the press, so the current review's duration does not exist yet;
REM recomputing live as it accrues would make the displayed numbers churn while the
REM user reads them AND feed back into their dwell time. Freezing it also makes
REM displayed == scheduled. 0.0 is the pipeline's OWN unknown-encoding (query rows
REM zero scaled_duration, data_processing reject_columns), so the model has seen it
REM on every query row; note scale_duration(x)=(log(10+x)-8.9)/1.07 means a literal
REM 0.0 implies ~7.3 s, very close to the 6,433 ms median it replaces. Real duration
REM still enters the state update via the real row -- only the CURRENT row's duration
REM is withheld from the prediction. Retires duration_median.json from the contract.
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
REM vprune deliberately OFF: the ahead OBJECTIVE changed (scope rule -- prune only at MATCHED
REM regularization vs the reference) and MAX moved 32768->16384, so step pairing against the
REM champion val trace is meaningless. Same call as iter 32.
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25

echo ===== iter33_dur START %DATE% %TIME% ===== > "%LOG%"

echo === STEP 0.5: 40-step E2E sanity (PAVA probes + GRU N=3 + Muon wiring, VRAM) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter33_dur/iter33_dur_ws.toml > "%DIR%\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 9
)
set RWKV_MAX_STEPS=
echo SANITY OK %TIME% >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/iter33_dur/iter33_dur_ws_trace.jsonl
set RWKV_GRAD_STATS=scratchpad/iter33_dur/iter33_grad_stats_ws.json
del /Q scratchpad\iter33_dur\iter33_dur_ws_trace.jsonl scratchpad\iter33_dur\iter33_dur_ws_trace.jsonl.val.jsonl 2>nul

echo === WS 1 epoch (1-5000, A18 trunk + PAVA + GRU3 + Muon, vprune ON min6000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter33_dur/iter33_dur_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
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
set RWKV_GRAD_STATS=scratchpad/iter33_dur/iter33_grad_stats_decay.json

echo === DECAY SETUP (0.25 ep, MAX=16384) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter33_dur iter33ws iter33d scratchpad/iter33_dur/iter33_dur_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 16384 > "%DIR%\decay_setup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter33_dur/iter33_dur_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo DECAY OK %TIME% >> "%LOG%"

del /Q result\RWKV-iter33_dur.jsonl result\RWKV-P-iter33_dur.jsonl result\RWKV-iter33_dur-s0.jsonl result\RWKV-P-iter33_dur-s0.jsonl result\RWKV-iter33_dur.nanskip.jsonl result\RWKV-iter33_dur-s0.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (VAL 5001-7500) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter33_dur iter33d scratchpad/iter33_dur/iter33_dur_eval.toml RWKV-iter33_dur RWKV-P-iter33_dur 5001 7500 > "%DIR%\eval_toml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
REM RECTIFIED eval: the directive makes train/eval/deploy one quantity, so the gate metric is
REM now the rectified one. Compare ONLY against rectified baselines.
set RWKV_EVAL_PAVA=1
echo === EVAL (RECTIFIED, single process, d=80, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter33_dur/iter33_dur_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
echo EVAL OK %TIME% >> "%LOG%"

echo === GATE: paired vs CHAMPION iter31 RECTIFIED (val half) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter33_dur.jsonl --cand-imm result/RWKV-P-iter33_dur.jsonl --champ-ahead result/RWKV-iter31_algo_rect.jsonl --champ-imm result/RWKV-P-iter31_algo_rect.jsonl > "%DIR%\gate_%STAMP%.log" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
