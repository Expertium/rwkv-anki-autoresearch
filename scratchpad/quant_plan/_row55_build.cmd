@echo off
rem Rebuild the iter-55 bundle. Outer lock dir makes a doubled shell launch a no-op;
rem the builder's own lock is cleared first (a killed earlier copy can leave it stale).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
mkdir scratchpad\_build50.lock 2>nul
if errorlevel 1 (
  echo SKIP: another build instance is running
  exit /b 0
)
del /f /q scratchpad\quant_plan\_build_row55_bundle.lock 2>nul
.venv\Scripts\python.exe scratchpad\quant_plan\corruption\_build_row55_bundle.py
echo BUILD RC=%errorlevel%
del /f /q scratchpad\quant_plan\_build_row55_bundle.lock 2>nul
rmdir scratchpad\_build50.lock
