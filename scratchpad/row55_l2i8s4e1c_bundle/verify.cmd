@echo off
REM Build the Rust deploy engine and run the checks of verification.md, in order.
REM Run from anywhere; this script finds its own folder. Needs: cargo, python + numpy + scikit-learn.
REM The shipped traces are users 107 and 136. The 23 MB long-history trace is not shipped.
setlocal
cd /d "%~dp0code"
set PY=python
if exist "..\..\.venv\Scripts\python.exe" set PY=..\..\.venv\Scripts\python.exe
set W=..\weights\deploy_iter55_l2i8s4e1c.safetensors

echo === [1/5] build the Rust engine ===========================================
cargo build --release --manifest-path rust\rwkv-infer\Cargo.toml || goto :fail
echo.

echo === [2/5] deploy log loss, plain vs l2i8s4, users 107/136 ==================
echo     expect l2i8s4 tax +0.000329 ahead / +0.000338 imm, both inside the 0.0005 tolerance
%PY% scratchpad\quant_plan\rust_grid.py --weights %W% --trace-dir ..\traces --users 107,136 --configs plain,l2i8s4 --tag verify || goto :fail
echo.

echo === [3/5] non-finiteness check, strict, users 107/136 ===================
echo     expect PASS rc=0, 43,474,944 state values, zero non-finite
%PY% scratchpad\quant_plan\rust_finite.py --weights %W% --configs plain,l2i8s4 --trace-dir ..\traces --users 107,136 --tag verify --strict || goto :fail
echo.

echo === [4/5] Python to Rust parity ============================================
echo -- plain engine vs the trace's own Python RNN reference (expect PASS) --
%PY% scratchpad\quant_plan\py_rust_parity.py --weights %W% --trace-dir ..\traces --config plain --users 107,136 --tag verify_plain || goto :fail
echo.
echo -- There is deliberately NO quantized-path parity step: the trace's py_pred_* is the Python
echo    RNN path, which does not apply the state quantization, so it is an UNQUANTIZED reference
echo    and the quantized config fails against it by construction (verification.md section 4b).
echo.

echo === [5/5] deploy speed, informational (the floor is 0.75x) =================
echo     expect about 0.80x rev/s; the RATIO is what matters, not the absolute rate
%PY% scratchpad\quant_plan\rust_speed.py --weights %W% --trace-dir ..\traces --users 107,136 --configs plain,l2i8s4 --tag verify --reps 2
echo.

echo ALL CHECKS RAN. Step 5 is informational. Compare every number with verification.md.
exit /b 0

:fail
echo.
echo *** A CHECK FAILED (see the output above) ***
exit /b 1
