@echo off
REM Route-R (measure-first): does a smaller COLD chunk (8192, B~4) give the GPU-util speedup without
REM the intricate stateful carry, and at what accuracy cost vs the 65536-chunk (B~1) baseline?
REM Pipeline: WS base65k -> WS sc8k -> eval both on 101-200 -> compare. Fresh same-session runs for a
REM fair speed+accuracy comparison. Detached (Esc/teardown-proof); monitor scratchpad/route_r.log.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\route_r.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_DETERMINISTIC=1

echo ================ ROUTE R START %DATE% %TIME% ================ > "%LOG%"

echo === WS base65k (65536-chunk, B~1) START %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/r_ws_base65k.toml >> "%LOG%" 2>&1
echo === WS base65k DONE %TIME% === >> "%LOG%"

echo === WS sc8k (8192-chunk, B~4) START %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/r_ws_sc8k.toml >> "%LOG%" 2>&1
echo === WS sc8k DONE %TIME% === >> "%LOG%"

REM get_result SKIPS users already present in BOTH output files -> delete stale jsonls so it re-evals.
del /Q result\RWKV-r-base65k.jsonl result\RWKV-P-r-base65k.jsonl 2>nul
del /Q result\RWKV-r-sc8k.jsonl result\RWKV-P-r-sc8k.jsonl 2>nul

echo === EVAL base65k on 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/r_eval_base65k.toml >> "%LOG%" 2>&1
echo === EVAL sc8k on 101-200 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.get_result --config scratchpad/r_eval_sc8k.toml >> "%LOG%" 2>&1

echo === COMPARISON === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/compare_route_r.py >> "%LOG%" 2>&1

echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
