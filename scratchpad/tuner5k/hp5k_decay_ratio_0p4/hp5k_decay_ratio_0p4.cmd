@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
REM RWKV_QAT_COMPILE needs MSVC cl.exe on PATH or inductor fails into hollow skipped-batch steps
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner5k\hp5k_decay_ratio_0p4.log
set OMP_NUM_THREADS=4
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_N_HEADS=2
set RWKV_HEAD_DIM=16
set RWKV_WEIGHT_DECAY=0.2
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=1
set RWKV_CB_LR_MULT=1
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
set RWKV_STEP_TRACE=scratchpad/tuner5k/hp5k_decay_ratio_0p4/hp5k_decay_ratio_0p4_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/champion_5k.json
echo ===== TRIAL hp5k_decay_ratio_0p4 (param=decay_ratio=0.4) cfg={"peak_lr": 0.001, "warmup_steps": 200, "weight_decay": 0.2, "clip": 0.25, "decay_ratio": 0.4, "adamw_beta2": 0.999, "dropout_scale": 1.0, "cb_lr_mult": 1.0} START %DATE% %TIME% ===== > "%LOG%"
REM re-run hygiene: STEP_TRACE (and its .val sidecar) open in APPEND mode -- a leftover trace/marker
REM from a prior-era run of this same config would pollute this run's files (liveplot + post-hoc).
del /Q scratchpad\tuner5k\hp5k_decay_ratio_0p4\hp5k_decay_ratio_0p4_ws_trace.jsonl scratchpad\tuner5k\hp5k_decay_ratio_0p4\hp5k_decay_ratio_0p4_ws_trace.jsonl.pruned.json scratchpad\tuner5k\hp5k_decay_ratio_0p4\hp5k_decay_ratio_0p4_ws_trace.jsonl.val.jsonl 2>nul
echo === WS 1 epochs (1-5000) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/hp5k_decay_ratio_0p4/hp5k_decay_ratio_0p4_ws.toml >> "%LOG%" 2>&1
if %ERRORLEVEL%==42 (
  echo === WILCOXON-PRUNED - recording estimated logloss %TIME% === >> "%LOG%"
  .venv\Scripts\python.exe optimization/hp_tuner_5k.py record-pruned hp5k_decay_ratio_0p4 >> "%LOG%" 2>&1
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
REM a crashed WS must NOT cascade into decay/eval (hp5k_weight_decay_0p2 decayed a step-50 ckpt)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo === RESOLVE WS CODEBOOKS (feed decay) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/tuner5k/hp5k_decay_ratio_0p4 hp5k_decay_ratio_0p4ws scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_wkv_ws.txt scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_shift_ws.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_WS %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_wkv_ws.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_shift_ws.txt
echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/tuner5k/hp5k_decay_ratio_0p4 hp5k_decay_ratio_0p4ws hp5k_decay_ratio_0p4d scratchpad/tuner5k/hp5k_decay_ratio_0p4/hp5k_decay_ratio_0p4_decay.toml train_db_5k_h1 1 5000 0.4 0.001 >> "%LOG%" 2>&1
echo === DECAY 0.4 epoch (ratio 0.4) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner5k/hp5k_decay_ratio_0p4/hp5k_decay_ratio_0p4_decay.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo === RESOLVE DECAY CODEBOOKS (feed eval) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/resolve_run_cbs.py scratchpad/tuner5k/hp5k_decay_ratio_0p4 hp5k_decay_ratio_0p4d scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_wkv_final.txt scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_shift_final.txt >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_CBFAIL_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 3
)
set RWKV_QAT_PQ=scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_wkv_final.txt
set RWKV_QAT_SHIFT_PQ=scratchpad/tuner5k/hp5k_decay_ratio_0p4/cb_shift_final.txt
del /Q result\RWKV-hp5k_decay_ratio_0p4.jsonl result\RWKV-P-hp5k_decay_ratio_0p4.jsonl 2>nul
echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/tuner5k/hp5k_decay_ratio_0p4 hp5k_decay_ratio_0p4d scratchpad/tuner5k/hp5k_decay_ratio_0p4/hp5k_decay_ratio_0p4_eval.toml RWKV-hp5k_decay_ratio_0p4 RWKV-P-hp5k_decay_ratio_0p4 >> "%LOG%" 2>&1
echo === EVAL 5001-5200 (quant-aware) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/tuner5k/hp5k_decay_ratio_0p4/hp5k_decay_ratio_0p4_eval.toml >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo === RECORD hp5k_decay_ratio_0p4 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/hp_tuner_5k.py record hp5k_decay_ratio_0p4 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
