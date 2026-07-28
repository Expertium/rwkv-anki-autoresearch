@echo off
REM iter 32 RECTIFIED eval (the deploy metric) + the throughput measurement it still owed.
REM Env is iter 31's rectified-eval block verbatim -- KD is train-time only, so iter 32's
REM eval-time architecture and flags are identical. Only the checkpoint and output names differ.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=scratchpad\eval_pava
set LOG=%DIR%\iter32_rect.log
set STAMP=%RANDOM%%RANDOM%

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
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
REM PAVA_LAMBDA must be set: it creates pava_theta, and load_state_dict is strict --
REM without it the checkpoint will not open at all.
set RWKV_PAVA_LAMBDA=0.1
set RWKV_PROBE_DENSITY=0.08
set RWKV_MUON=1
set RWKV_EVAL_PAVA=1
set RWKV_PROBE_DUR=0.0

echo ===== iter32 RECTIFIED eval START %DATE% %TIME% ===== > "%LOG%"

del /Q result\RWKV-iter32_kd_rect.jsonl result\RWKV-P-iter32_kd_rect.jsonl result\RWKV-iter32_kd_rect.nanskip.jsonl 2>nul

echo === write eval toml %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter32_kd iter32d scratchpad/eval_pava/iter32_rect_eval.toml RWKV-iter32_kd_rect RWKV-P-iter32_kd_rect 5001 7500 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 4
)

echo === iter32 rectified eval %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/eval_pava/iter32_rect_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 > "%DIR%\iter32_rect_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo ITER32 RECT OK %TIME% >> "%LOG%"

echo === GATE: iter32 RECT vs iter31 RECT (the deploy metric, both rectified) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --intersect --cand-ahead result/RWKV-iter32_kd_rect.jsonl --cand-imm result/RWKV-P-iter32_kd_rect.jsonl --champ-ahead result/RWKV-iter31_algo_rect.jsonl --champ-imm result/RWKV-P-iter31_algo_rect.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"

echo === THROUGHPUT (the measurement iter 32 owed) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/measure_throughput.py scratchpad/iter32_kd/iter32d_5586.pth >> "%LOG%" 2>&1
echo THROUGHPUT_DONE (exit %ERRORLEVEL%) >> "%LOG%"

endlocal
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
