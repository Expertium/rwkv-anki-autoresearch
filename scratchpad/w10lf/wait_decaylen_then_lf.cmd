@echo off
REM Leak-free run waiter (2026-09-10). Andrew: "I think we should do a leak-free run anyway.
REM Nothing that cheats should make it into the final version that will be shipped."
REM
REM PLACEMENT, the default: at the END of the queue, behind the decay-length pair. The experiments
REM before it (cmixgraft, QAT-KD, budget curve, decay length) pick the recipe; this run is the
REM leak-free base the shipped model is built on. If phase L says the leak is MATERIAL, the plan is
REM to move this run forward (behind cmixgraft) and re-base the later experiments on it -- that
REM move is done by hand, by stopping this waiter's cmd FIRST and arming another.
REM
REM GATE 0 -- wait_decaylen.log must carry a terminal marker (anchored). DONE_EXIT_120 there means
REM arm 2's eval failed upstream and the GPU is kept for its resume: REFUSE with 140. Any other
REM marker means the GPU is free.
REM
REM T -- clk_decide.py applies the pre-registered phase-L rule: exit 3 = the 3 h window costs
REM >= +0.0005 imm more than the 30 min one, so T=10800; anything else = T=1800, the default.
REM Both runners are regenerated at that T and preflighted before anything runs.
REM
REM GATE 1 -- the dry run (0.05 + 0.05 epochs, 20 users) must end DONE_EXIT_0 with no swallowed
REM exception and both result files.
REM
REM STOP THIS WAITER'S cmd FIRST, then its children.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set R=C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=%R%\scratchpad\w10lf\wait_lf.log
set PREV=%R%\scratchpad\w10d3\wait_decaylen.log
set DRYDIR=%R%\scratchpad\w10lfdry
echo ===== leak-free run waiter armed on the decay-length waiter %DATE% %TIME% ===== >> "%WLOG%"

:waitprev
if not exist "%PREV%" goto sleepprev
findstr /B /C:"DONE_EXIT_" "%PREV%" >nul 2>&1
if errorlevel 1 goto sleepprev
goto gotprev
:sleepprev
ping -n 61 127.0.0.1 >nul
goto waitprev

:gotprev
findstr /B /C:"DONE_EXIT_120 " "%PREV%" >nul 2>&1
if not errorlevel 1 (
  echo arm 2 eval FAILED upstream -- the GPU is kept for its resume, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_140 %DATE% %TIME% >> "%WLOG%"
  exit /b 140
)
.venv\Scripts\python.exe scratchpad/w10clk/clk_decide.py > "%R%\scratchpad\w10lf\decide.log" 2>&1
set RC=%ERRORLEVEL%
type "%R%\scratchpad\w10lf\decide.log" >> "%WLOG%"
set LFT=1800
if "%RC%"=="3" set LFT=10800
echo phase L decision %RC% -- the leak-free run uses T=%LFT% s %DATE% %TIME% >> "%WLOG%"
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
echo DONE_EXIT_146 %DATE% %TIME% >> "%WLOG%"
exit /b 146

:launch
echo previous chain done -- 120 s for its workers, then the dry run %DATE% %TIME% >> "%WLOG%"
ping -n 121 127.0.0.1 >nul
call %R%\scratchpad\w10lfdry\run_w10lfdry.cmd
findstr /B /C:"DONE_EXIT_0 " "%DRYDIR%\w10lfdry.log" >nul 2>&1
if errorlevel 1 (
  echo leak-free dry run ended WITHOUT success -- not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_141 %DATE% %TIME% >> "%WLOG%"
  exit /b 141
)
findstr /C:"Traceback" /C:"Exception caught" "%DRYDIR%\ws_*.log" "%DRYDIR%\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo leak-free dry run swallowed an exception -- a HOLLOW run, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_142 %DATE% %TIME% >> "%WLOG%"
  exit /b 142
)
if not exist "%R%\result\RWKV-w10lfdry.jsonl" (
  echo leak-free dry eval produced no ahead result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_143 %DATE% %TIME% >> "%WLOG%"
  exit /b 143
)
if not exist "%R%\result\RWKV-P-w10lfdry.jsonl" (
  echo leak-free dry eval produced no imm result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_144 %DATE% %TIME% >> "%WLOG%"
  exit /b 144
)
echo dry run clean -- launching the leak-free run %DATE% %TIME% >> "%WLOG%"
call %R%\scratchpad\w10lf\run_w10lf.cmd
echo run_w10lf.cmd returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
