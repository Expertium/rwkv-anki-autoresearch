@echo off
REM ============================================================================
REM MEME RUN "BLIND RWKV" (Andrew 2026-07-19, recorded SEPARATELY from the
REM research loop): train the d=32 model WITHOUT interval-length features and
REM WITHOUT grades -- the two signals every classical SRS algorithm relies on.
REM RWKV_ZERO_FEATURES = dims 0-7 (all six elapsed/interval features) + 9-12
REM (grade one-hot) + 22 (card state, champion-recipe standard). Duration (dim 8)
REM and everything else stays. Question: does blind RWKV still beat FSRS-7?
REM TARGET: FSRS-7-sched_penalties-short-secs-recency on users 5001-10000 =
REM by-user mean LogLoss 0.317933 (RMSE(bins) 0.0591). Compare vs AHEAD mode.
REM Recipe notes: vprune OFF (champion val ref would false-kill a crippled
REM model); PAVA/probes OFF (grade probes meaningless without grade inputs);
REM state clamp ON (completeness insurance -> full n=5000). Standard 64-basis
REM curve head, NO_AHEAD_RESIDUAL per the mandatory recipe. ~3.5 h.
REM STEP 0 waits for iter 25's DONE_EXIT. Launch DETACHED (CRLF file!).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\meme_blind\meme_blind.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_ZERO_FEATURES=0,1,2,3,4,5,6,7,9,10,11,12,22
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STEP_TRACE=scratchpad/meme_blind/meme_blind_ws_trace.jsonl

echo ===== MEME_BLIND START %DATE% %TIME% ===== > "%LOG%"
echo === STEP 0: wait for iter 25 to release the GPU %TIME% === >> "%LOG%"
:waitloop0
findstr /C:"DONE_EXIT" C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter25_gru\iter25_gru.log >nul 2>&1
if not %ERRORLEVEL%==0 (
  timeout /t 120 /nobreak >nul
  goto waitloop0
)
echo iter 25 done -- starting the blind run %TIME% >> "%LOG%"
del /Q scratchpad\meme_blind\meme_blind_ws_trace.jsonl scratchpad\meme_blind\meme_blind_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epoch (1-5000, d=32 BLIND: no intervals no grades, vprune OFF) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/meme_blind/meme_blind_ws.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 2
)
set RWKV_STEP_TRACE=

echo === DECAY SETUP (0.25 ep) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/meme_blind memebws memebd scratchpad/meme_blind/meme_blind_decay.toml train_db_5k_h1 1 5000 0.25 1e-3 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/meme_blind/meme_blind_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)

del /Q result\RWKV-meme_blind.jsonl result\RWKV-P-meme_blind.jsonl result\RWKV-meme_blind-solo.jsonl result\RWKV-P-meme_blind-solo.jsonl result\RWKV-meme_blind-s0.jsonl result\RWKV-P-meme_blind-s0.jsonl result\RWKV-meme_blind-s1.jsonl result\RWKV-P-meme_blind-s1.jsonl result\RWKV-meme_blind.nanskip.jsonl result\RWKV-meme_blind-s0.nanskip.jsonl result\RWKV-meme_blind-s1.nanskip.jsonl result\RWKV-meme_blind-solo.nanskip.jsonl 2>nul
echo === WRITE EVAL TOML (FULL 5001-10000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/meme_blind memebd scratchpad/meme_blind/meme_blind_eval.toml RWKV-meme_blind RWKV-P-meme_blind 5001 10000 >> "%LOG%" 2>&1
echo === EVAL (phased sharded, state-clamp ON) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/meme_blind/meme_blind_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === INFO: paired vs iter23 champion (the cost of blindness) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-meme_blind.jsonl --cand-imm result/RWKV-P-meme_blind.jsonl --champ-ahead result/RWKV-iter23_pava.jsonl --champ-imm result/RWKV-P-iter23_pava.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (FSRS-7 comparison happens at record time) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
