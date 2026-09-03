@echo off
REM Launch the gen4base EVAL RESUME when the ablation chain has reported (its waiter writes the
REM marker to wait_ablate.log after run_ablate returns): the ablation took the GPU the moment the
REM eval's failure marker landed, so the resume queues behind it. Anchored findstr.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\wait_evalresume.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_ablate\wait_ablate.log

echo ===== evalresume waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo ablation chain reported %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\run_gen4base_evalresume.cmd
echo run_gen4base_evalresume returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
