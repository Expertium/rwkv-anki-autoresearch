@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner5k\hp5k_warmup_steps_800.log
set OMP_NUM_THREADS=4
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY=0.01
set RWKV_CLIP=0.25
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt
set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ_LEARN=1
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=student
set RWKV_QAT_ROT_CACHE=1
set RWKV_QAT_FAST_EMB=1
set RWKV_QAT_EMA_FOREACH=1
set RWKV_QAT_NO_MEMFILL=1
set RWKV_STEP_TRACE=scratchpad/tuner5k/hp5k_warmup_steps_800/hp5k_warmup_steps_800_ws_trace.jsonl
set RWKV_PRUNE_REF=optimization/champion_5k.json
set RWKV_PRUNE_MIN_STEP=1600
echo ===== TRIAL hp5k_warmup_steps_800 (param=warmup_steps=800) cfg={"peak_lr": 0.001, "warmup_steps": 800, "weight_decay": 0.01, "clip": 0.25, "decay_ratio": 0.25} START %DATE% %TIME% ===== > "%LOG%"
echo === WS 1 epochs (1-5000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/hp5k_warmup_steps_800/hp5k_warmup_steps_800_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo === WILCOXON-PRUNED - recording estimated logloss %TIME% === >> "%LOG%"
  .venv\Scripts\python.exe optimization/hp_tuner_5k.py record-pruned hp5k_warmup_steps_800 >> "%LOG%" 2>&1
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/tuner5k/hp5k_warmup_steps_800 hp5k_warmup_steps_800ws scratchpad/tuner5k/hp5k_warmup_steps_800/cb_wkv_ws.txt scratchpad/tuner5k/hp5k_warmup_steps_800/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/tuner5k/hp5k_warmup_steps_800/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/tuner5k/hp5k_warmup_steps_800/cb_shift_ws.txt
echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/tuner5k/hp5k_warmup_steps_800 hp5k_warmup_steps_800ws hp5k_warmup_steps_800d scratchpad/tuner5k/hp5k_warmup_steps_800/hp5k_warmup_steps_800_decay.toml train_db_5k_h1 1 5000 0.25 0.001 >> "%LOG%" 2>&1
echo === DECAY 0.25 epoch (ratio 0.25) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/hp5k_warmup_steps_800/hp5k_warmup_steps_800_decay.toml >> "%LOG%" 2>&1
echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/tuner5k/hp5k_warmup_steps_800 hp5k_warmup_steps_800d scratchpad/tuner5k/hp5k_warmup_steps_800/cb_wkv_final.txt scratchpad/tuner5k/hp5k_warmup_steps_800/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/tuner5k/hp5k_warmup_steps_800/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/tuner5k/hp5k_warmup_steps_800/cb_shift_final.txt
del /Q result\RWKV-hp5k_warmup_steps_800.jsonl result\RWKV-P-hp5k_warmup_steps_800.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/tuner5k/hp5k_warmup_steps_800 hp5k_warmup_steps_800d scratchpad/tuner5k/hp5k_warmup_steps_800/hp5k_warmup_steps_800_eval.toml RWKV-hp5k_warmup_steps_800 RWKV-P-hp5k_warmup_steps_800 >> "%LOG%" 2>&1
echo === EVAL 5001-5200 (quant-aware) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/tuner5k/hp5k_warmup_steps_800/hp5k_warmup_steps_800_eval.toml >> "%LOG%" 2>&1
echo === RECORD hp5k_warmup_steps_800 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/hp_tuner_5k.py record hp5k_warmup_steps_800 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
