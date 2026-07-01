@echo off
REM Rebuild the RWKV CUDA extension after the K-dynamic (K=16 support) kernel changes. vcvars64 + CUDA 13.2
REM + setup.py build_ext --inplace. Detached. Monitor scratchpad/build_k16.log (BUILD_EXIT_).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\build_k16.log
echo ===== BUILD K16 START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe setup.py build_ext --inplace >> "%LOG%" 2>&1
echo BUILD_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
