@echo off
REM Leak-free run waiter v2 (2026-09-11) -- w10lf directly BEHIND CMIXGRAFT.
REM Phase L said MATERIAL: moving query and probe rows to the previous answer costs arm 1
REM +0.0032 imm / +0.0003 ahead (scratchpad/leak/leak_report.txt). So w10lf moves from the end of
REM the queue to here, and QAT-KD, the budget curve and the decay-length pair are re-based on its WS
REM instead of ws10's. v1 (wait_decaylen_then_lf.cmd, pid 3724) was stopped 08:55 with no marker.
REM
REM GATE 0 -- the cmixgraft waiter's log must carry a terminal marker (anchored). DONE_EXIT_90 there
REM means arm 2's eval failed upstream and the GPU is kept for its resume: REFUSE with 150. Any other
REM marker (0 = cmixgraft ran; 92..95 = its dry run refused) means the GPU is free.
REM
REM T -- fixed at 1800 s: clk_decide.py returned 0 on 2026-09-11 (the 3 h arm adds +0.000129 imm,
REM under the +0.0005 switch). Both runners are regenerated at T=1800 and preflighted first.
REM
REM GATE 1 -- the dry run (0.05 + 0.05 epochs, 20 users) must end DONE_EXIT_0 with no swallowed
REM exception and both result files.
REM
REM STOP THIS WAITER'S cmd FIRST, then its children.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set R=C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=%R%\scratchpad\w10lf\wait_lf2.log
set PREV=%R%\scratchpad\cmixgraft\wait_cmixgraft.log
set DRYDIR=%R%\scratchpad\w10lfdry
set LFT=1800
echo ===== leak-free run waiter v2 armed on the cmixgraft waiter %DATE% %TIME% ===== >> "%WLOG%"

:waitprev
if not exist "%PREV%" goto sleepprev
findstr /B /C:"DONE_EXIT_" "%PREV%" >nul 2>&1
if errorlevel 1 goto sleepprev
goto gotprev
:sleepprev
ping -n 61 127.0.0.1 >nul
goto waitprev

:gotprev
findstr /B /C:"DONE_EXIT_90 " "%PREV%" >nul 2>&1
if not errorlevel 1 (
  echo arm 2 eval FAILED upstream per the cmixgraft waiter -- the GPU is kept for its resume, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_150 %DATE% %TIME% >> "%WLOG%"
  exit /b 150
)
echo cmixgraft chain done -- the leak-free run uses T=%LFT% s, decided by phase L on 2026-09-11 %DATE% %TIME% >> "%WLOG%"
.venv\Scripts\python.exe scratchpad/w10lf/mk_w10lf.py %LFT% >> "%WLOG%" 2>&1
if errorlevel 1 goto regenfail
.venv\Scripts\python.exe scratchpad/preflight_runner.py scratchpad/w10lf/run_w10lf.cmd scratchpad/w10lfdry/run_w10lfdry.cmd >> "%WLOG%" 2>&1
if errorlevel 1 goto regenfail
findstr /C:"set RWKV_CLOCK_AT_PREV_ANSWER=%LFT%" "%R%\scratchpad\w10lf\run_w10lf.cmd" >nul
if errorlevel 1 goto regenfail
findstr /C:"set RWKV_CLOCK_AT_PREV_ANSWER=%LFT%" "%DRYDIR%\run_w10lfdry.cmd" >nul
if errorlevel 1 goto regenfail
goto launch
:regenfail
echo regenerating the leak-free runners at T=%LFT% FAILED -- not launching %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_156 %DATE% %TIME% >> "%WLOG%"
exit /b 156

:launch
echo 120 s for the previous run's workers, then the dry run %DATE% %TIME% >> "%WLOG%"
ping -n 121 127.0.0.1 >nul
call %R%\scratchpad\w10lfdry\run_w10lfdry.cmd
findstr /B /C:"DONE_EXIT_0 " "%DRYDIR%\w10lfdry.log" >nul 2>&1
if errorlevel 1 (
  echo leak-free dry run ended WITHOUT success -- not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_151 %DATE% %TIME% >> "%WLOG%"
  exit /b 151
)
findstr /C:"Traceback" /C:"Exception caught" "%DRYDIR%\ws_*.log" "%DRYDIR%\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo leak-free dry run swallowed an exception -- a HOLLOW run, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_152 %DATE% %TIME% >> "%WLOG%"
  exit /b 152
)
if not exist "%R%\result\RWKV-w10lfdry.jsonl" (
  echo leak-free dry eval produced no ahead result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_153 %DATE% %TIME% >> "%WLOG%"
  exit /b 153
)
if not exist "%R%\result\RWKV-P-w10lfdry.jsonl" (
  echo leak-free dry eval produced no imm result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_154 %DATE% %TIME% >> "%WLOG%"
  exit /b 154
)
echo dry run clean -- launching the leak-free run %DATE% %TIME% >> "%WLOG%"
call %R%\scratchpad\w10lf\run_w10lf.cmd
echo run_w10lf.cmd returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
