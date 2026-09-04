@echo off
REM Waits for the LOO sweep PHASE 2 (scratchpad/feat_loo/loo_p2.log) to write its anchored terminal
REM marker -- ANY DONE_EXIT_: durdrop only needs a free GPU, not a LOO success -- then runs the
REM durdrop iteration (WS -> decay -> eval, ~10.5 h). Polls every 2 minutes. Writes wait_durdrop.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\durdrop\wait_durdrop.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_loo\loo_p2.log
echo ===== durdrop waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitloo
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitloo
)
echo LOO phase 2 reported, launching durdrop %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\durdrop\run_durdrop.cmd
echo run_durdrop returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
