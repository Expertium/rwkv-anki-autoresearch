@echo off
REM Waits for the d=128 teacher timing run (scratchpad/teacher_gen5/t128_timing.log) to write its
REM anchored terminal marker (any DONE_EXIT_: only a free GPU is needed), then runs the LOO sweep's
REM PHASE 2 (arms 8-19; arm 8 died 18:06 on a transient CUDA illegal-memory-access in group_norm
REM after seven identical arms ran clean). Writes wait_loo_p2.log.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_loo\wait_loo_p2.log
set G0=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\teacher_gen5\t128_timing.log
echo ===== loo p2 waiter armed %DATE% %TIME% ===== >> "%LOG%"
:waitt
findstr /B /C:"DONE_EXIT_" "%G0%" >nul 2>&1
if errorlevel 1 (
  ping -n 61 127.0.0.1 >nul
  goto waitt
)
echo timing reported, launching LOO phase 2 %DATE% %TIME% >> "%LOG%"
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_loo\run_loo_p2.cmd
echo loo p2 returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
