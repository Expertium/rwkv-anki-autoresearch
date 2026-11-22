@echo off
REM Build the Rust deploy engine and run the four checks of verification.md, in order.
REM Run from anywhere; this script finds its own folder. Needs: cargo, python + numpy + scikit-learn.
setlocal
cd /d "%~dp0code"
set PY=python
if exist "..\..\.venv\Scripts\python.exe" set PY=..\..\.venv\Scripts\python.exe

echo === [1/4] build the Rust engine ==========================================
cargo build --release --manifest-path rust\rwkv-infer\Cargo.toml || goto :fail
echo.

echo === [2/4] deploy log loss, plain vs l2i8s4, users 107/136/156 ============
%PY% scratchpad\quant_plan\rust_grid.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors --trace-dir ..\traces --users 107,136,156 --configs plain,l2i8s4 --tag verify || goto :fail
echo.

echo === [3/4] non-finiteness check, strict, users 107/136/156 =================
%PY% scratchpad\quant_plan\rust_finite.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors --configs plain,l2i8s4 --trace-dir ..\traces --users 107,136,156 --tag verify --strict || goto :fail
echo.

echo === [4/4] Python<->Rust parity ==========================================
echo -- 4a: plain engine vs the trace's own Python RNN reference (must be ~1e-6) --
%PY% scratchpad\quant_plan\py_rust_parity.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors --trace-dir ..\traces --config plain --users 107,136,156 --tag verify_plain || goto :fail
echo.
echo -- 4b: same on a 29k-review history (traces_long/user 5098) --
%PY% scratchpad\quant_plan\py_rust_parity.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors --trace-dir ..\traces_long --config plain --users 5098 --tag verify_plain_long || goto :fail
echo.
echo -- 4c: the recipe itself, quantized: the deploy tax on that long history --
%PY% scratchpad\quant_plan\py_rust_parity.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors --trace-dir ..\traces_long --config l2i8s4 --users 5098 --tag verify_l2i8s4_long || goto :fail
echo.
echo ALL FOUR CHECKS RAN. Compare the printed numbers with verification.md.
exit /b 0

:fail
echo.
echo *** A CHECK FAILED (see the output above) ***
exit /b 1
