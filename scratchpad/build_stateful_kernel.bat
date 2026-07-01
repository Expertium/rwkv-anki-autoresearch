@echo off
REM Rebuild the RWKV CUDA extension after the stateful-BPTT kernel changes (rwkv7_cuda.cu + rwkv7.cpp).
REM Recipe from the cuda-kernel-build memory: vcvars64 + DISTUTILS_USE_SDK + CUDA 13.2 + build_ext --inplace.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe setup.py build_ext --inplace
echo BUILD_EXIT_%ERRORLEVEL%
