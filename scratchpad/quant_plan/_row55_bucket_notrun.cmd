@echo off
REM Row-55 bucket tax: reuse the COMPLETED prediction dirs (no engine runs).
REM   plain  = rust_grid_preds\row55\plain       (users 107,136 + 40 roster users)
REM   l2i8s4 = rust_grid_preds\row55b\l2i8s4     (40 roster users, run directly)
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe -u scratchpad\quant_plan\bucket_tax.py ^
  --tag row55 --configs plain,l2i8s4 --no-run ^
  --preds-map "plain=scratchpad/quant_plan/rust_grid_preds/row55/plain,l2i8s4=scratchpad/quant_plan/rust_grid_preds/row55b/l2i8s4" ^
  --trace-dir scratchpad/quant_plan/trace_corrupt50 ^
  > scratchpad\quant_plan\bucket_row55b.log 2>&1
echo RC=%ERRORLEVEL%
