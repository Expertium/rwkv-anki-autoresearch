@echo off
REM Waits for `ordcut DECAY_OK` in ordcut.log (the decayed checkpoint exists from then on), then runs
REM the P3 screen on it at BelowNormal priority so the GPU eval keeps the CPU. Writes wait_screen.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ordcut\wait_screen.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ordcut\ordcut.log
echo ===== ordcut screen waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitdecay
findstr /B /C:"ordcut DECAY_OK" "%G0%" >nul 2>&1
if errorlevel 1 (
  findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
  if not errorlevel 1 (
    echo ordcut ended without DECAY_OK -- not screening %DATE% %TIME% >> "%LOG%"
    echo DONE_EXIT_66 %DATE% %TIME% >> "%LOG%"
    exit /b 66
  )
  ping -n 61 127.0.0.1 >nul
  goto waitdecay
)
if not exist "scratchpad\ordcut\oc_d_10935.pth" (
  echo DECAY_OK but no oc_d_10935.pth -- not screening %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_67 %DATE% %TIME% >> "%LOG%"
  exit /b 67
)
echo decay reported, launching the screen at BelowNormal %DATE% %TIME% >> "%LOG%"
start /B /BELOWNORMAL /WAIT cmd /c C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\ordcut\run_screen_ordcut.cmd
echo screen returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
