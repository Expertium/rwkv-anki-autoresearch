@echo off
REM Build the Rust deploy engine and run the checks of verification.md, in order.
REM Run from anywhere; this script finds its own folder. Needs: cargo, python + numpy + scikit-learn.
REM The shipped traces are users 107 and 136 (traces\) and user 5098 (traces_long\). User 156 is
REM NOT here: the current label DB puts review 1142 of that user in the equalize set and the
REM exporter's own per-review prediction dict has no entry for it, so it aborts (README section 9).
setlocal
cd /d "%~dp0code"
set PY=python
if exist "..\..\.venv\Scripts\python.exe" set PY=..\..\.venv\Scripts\python.exe
set W=..\weights\deploy_iter17_l2i8s4.safetensors

echo === [1/5] build the Rust engine ===========================================
cargo build --release --manifest-path rust\rwkv-infer\Cargo.toml || goto :fail
echo.

echo === [2/5] deploy log loss, plain vs l2i8s4, users 107/136 ==================
echo     expect l2i8s4 tax +0.000335 ahead / +0.000316 imm, both inside the 0.0005 tolerance
%PY% scratchpad\quant_plan\rust_grid.py --weights %W% --trace-dir ..\traces --users 107,136 --configs plain,l2i8s4 --tag verify || goto :fail
echo.

echo === [3/5] non-finiteness check, strict, users 107/136 ======================
echo     expect both configs PASS rc=0, zero non-finite state values and zero non-finite p(recall)
%PY% scratchpad\quant_plan\rust_finite.py --weights %W% --configs plain,l2i8s4 --trace-dir ..\traces --users 107,136 --tag verify --strict || goto :fail
echo.

echo === [4/5] Python to Rust parity ============================================
echo -- 4a: plain engine vs the trace's own Python RNN reference (expect PASS, ~1e-6) --
%PY% scratchpad\quant_plan\py_rust_parity.py --weights %W% --trace-dir ..\traces --config plain --users 107,136 --tag verify_plain || goto :fail
echo.
echo -- 4b: the same on a 29k-review history (traces_long\user 5098) --
%PY% scratchpad\quant_plan\py_rust_parity.py --weights %W% --trace-dir ..\traces_long --config plain --users 5098 --tag verify_plain_long || goto :fail
echo.
echo -- 4c: the quantized recipe on that long history --
echo     This step prints PARITY: FAIL and that IS the correct result. The trace's py_pred_* was
echo     written by the Python RNN path, which does not apply the state quantization, so it is an
echo     UNQUANTIZED reference. Read the printed d_ll instead (expect imm -0.003950, ahead -0.003295
echo     here) and see verification.md section 4b. This step is NOT fatal.
%PY% scratchpad\quant_plan\py_rust_parity.py --weights %W% --trace-dir ..\traces_long --config l2i8s4 --users 5098 --tag verify_l2i8s4_long
echo.

echo === [5/5] deploy speed, informational (the floor is 0.75x) =================
echo     expect about 0.82x rev/s; the RATIO is what matters, not the absolute rate
%PY% scratchpad\quant_plan\rust_speed.py --weights %W% --trace-dir ..\traces --users 107,136 --configs plain,l2i8s4 --tag verify --reps 2
echo.

echo ALL CHECKS RAN. Steps 4c and 5 are informational. Compare every number with verification.md.
exit /b 0

:fail
echo.
echo *** A CHECK FAILED (see the output above) ***
exit /b 1
