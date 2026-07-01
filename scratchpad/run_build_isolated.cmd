@echo off
REM Build the RWKV CUDA extension into build/lib.../ ONLY (NO --inplace), so the production
REM rwkv/model/RWKV_CUDA.*.pyd (locked by the running export workers) is NEVER overwritten. Used to
REM validate a kernel source change (compile + parity + microbench) while the export holds the prod pyd.
REM Detached. Monitor scratchpad/build_isolated.log (BUILD_EXIT_).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\build_isolated.log
echo ===== BUILD ISOLATED (no --inplace) START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe setup.py build_ext >> "%LOG%" 2>&1
echo BUILD_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
