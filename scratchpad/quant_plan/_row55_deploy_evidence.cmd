@echo off
REM Row-55 deploy-side evidence, with the SHIPPED weights, on the row-55 traces.
REM Idempotent: every step rewrites its own JSON. All output goes to stdout (redirected by the caller).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set W=scratchpad\row55_l2i8s4e1c_bundle\weights\deploy_iter55_l2i8s4e1c.safetensors
echo === rust_grid (plain vs l2i8s4, users 107/136) ===
.venv\Scripts\python.exe -u scratchpad\quant_plan\rust_grid.py --weights %W% --trace-dir scratchpad\quant_plan\trace_row55_parity --users 107,136 --configs plain,l2i8s4 --tag row55
if errorlevel 1 goto fail
echo === rust_finite strict (users 107/136) ===
.venv\Scripts\python.exe -u scratchpad\quant_plan\rust_finite.py --weights %W% --configs plain,l2i8s4 --trace-dir scratchpad\quant_plan\trace_row55_parity --users 107,136 --tag row55 --strict
if errorlevel 1 goto fail
echo === rust_finite strict (long user 5098) ===
.venv\Scripts\python.exe -u scratchpad\quant_plan\rust_finite.py --weights %W% --configs l2i8s4 --trace-dir scratchpad\quant_plan\trace_corrupt50 --users 5098 --tag row55_long --strict
if errorlevel 1 goto fail
echo === py_rust_parity plain (users 107/136) ===
.venv\Scripts\python.exe -u scratchpad\quant_plan\py_rust_parity.py --weights %W% --trace-dir scratchpad\quant_plan\trace_row55_parity --config plain --users 107,136 --tag row55_plain
if errorlevel 1 goto fail
echo === rust_speed (plain vs l2i8s4, reps 2) ===
.venv\Scripts\python.exe -u scratchpad\quant_plan\rust_speed.py --weights %W% --trace-dir scratchpad\quant_plan\trace_row55_parity --users 107,136 --configs plain,l2i8s4 --tag row55 --reps 2
if errorlevel 1 goto fail
echo ALL ROW55 DEPLOY EVIDENCE DONE
exit /b 0
:fail
echo ROW55 DEPLOY EVIDENCE FAILED at the step above
exit /b 1
