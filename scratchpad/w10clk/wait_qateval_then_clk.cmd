@echo off
REM Leak-free branch chain, INSERTED between arm 2's eval and cmixgraft (2026-09-10 22:30).
REM
REM GATE 0 -- w10qat_eval.log must end DONE_EXIT_0 (anchored); any other terminal marker means arm
REM 2's eval failed: REFUSE with 130, which the cmixgraft waiter (v2) turns into its old 90. A
REM MISSING log means WAIT. Arm 2's eval runner holds phase L (the counterfactual) at its start and
REM phase D (the tax split) at its end, so by its DONE_EXIT_0 both have run.
REM
REM DECIDE -- clk_decide.py applies the PRE-REGISTERED rule to phase L's results: 0 = insert at
REM T=1800, 3 = insert at T=10800 (runner and dry run regenerated first), 1 = below MATERIAL, 2 = no
REM decision (missing results or a failed harness). 1 and 2 skip the branch and end DONE_EXIT_137,
REM on which the queue continues; a human reads decide.log.
REM
REM GATE 1 -- the dry run (0.05 epochs, 20 users) must end DONE_EXIT_0 with no swallowed exception
REM and both result files; its runner already refuses without the clock-fix banner (56) and the
REM chunk-count line (57). New phase shape: the first run with RWKV_CLOCK_AT_PREV_ANSWER.
REM
REM STOP THIS WAITER'S cmd FIRST, then its children.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set R=C:\Users\Andrew\rwkv-anki-autoresearch
set WLOG=%R%\scratchpad\w10clk\wait_clk.log
set PREV=%R%\scratchpad\w10qat\w10qat_eval.log
set DRYDIR=%R%\scratchpad\w10clkdry
echo ===== leak-free branch waiter armed on arm 2's eval %DATE% %TIME% ===== >> "%WLOG%"

:waitprev
if not exist "%PREV%" goto sleepprev
findstr /B /C:"DONE_EXIT_" "%PREV%" >nul 2>&1
if errorlevel 1 goto sleepprev
goto gotprev
:sleepprev
ping -n 61 127.0.0.1 >nul
goto waitprev

:gotprev
findstr /B /C:"DONE_EXIT_0 " "%PREV%" >nul 2>&1
if errorlevel 1 (
  echo arm 2 eval ended WITHOUT success -- refusing, nothing launched %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_130 %DATE% %TIME% >> "%WLOG%"
  exit /b 130
)
.venv\Scripts\python.exe scratchpad/w10clk/clk_decide.py > "%R%\scratchpad\w10clk\decide.log" 2>&1
set RC=%ERRORLEVEL%
type "%R%\scratchpad\w10clk\decide.log" >> "%WLOG%"
if "%RC%"=="0" goto insert
if "%RC%"=="3" goto regen
echo decision %RC% -- the branch is NOT inserted; the queue continues %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_137 %DATE% %TIME% >> "%WLOG%"
exit /b 137

:regen
echo regenerating the branch and its dry run at T=10800 %DATE% %TIME% >> "%WLOG%"
.venv\Scripts\python.exe scratchpad/w10clk/mk_w10clk.py 10800 >> "%WLOG%" 2>&1
if errorlevel 1 goto regenfail
.venv\Scripts\python.exe scratchpad/mk_dryrun.py scratchpad/w10clk/run_w10clk.cmd >> "%WLOG%" 2>&1
if errorlevel 1 goto regenfail
.venv\Scripts\python.exe scratchpad/preflight_runner.py scratchpad/w10clk/run_w10clk.cmd scratchpad/w10clkdry/run_w10clkdry.cmd >> "%WLOG%" 2>&1
if errorlevel 1 goto regenfail
findstr /C:"set RWKV_CLOCK_AT_PREV_ANSWER=10800" "%R%\scratchpad\w10clk\run_w10clk.cmd" >nul
if errorlevel 1 goto regenfail
goto insert
:regenfail
echo regeneration at T=10800 FAILED -- the branch is NOT inserted %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_136 %DATE% %TIME% >> "%WLOG%"
exit /b 136

:insert
echo inserting w10clk -- 120 s for arm 2's workers, then the dry run %DATE% %TIME% >> "%WLOG%"
ping -n 121 127.0.0.1 >nul
call %R%\scratchpad\w10clkdry\run_w10clkdry.cmd
findstr /B /C:"DONE_EXIT_0 " "%DRYDIR%\w10clkdry.log" >nul 2>&1
if errorlevel 1 (
  echo w10clk dry run ended WITHOUT success -- not launching w10clk %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_131 %DATE% %TIME% >> "%WLOG%"
  exit /b 131
)
findstr /C:"Traceback" /C:"Exception caught" "%DRYDIR%\decay_*.log" >nul 2>&1
if not errorlevel 1 (
  echo w10clk dry decay swallowed an exception -- a HOLLOW run, not launching %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_132 %DATE% %TIME% >> "%WLOG%"
  exit /b 132
)
if not exist "%R%\result\RWKV-w10clkdry.jsonl" (
  echo w10clk dry eval produced no ahead result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_133 %DATE% %TIME% >> "%WLOG%"
  exit /b 133
)
if not exist "%R%\result\RWKV-P-w10clkdry.jsonl" (
  echo w10clk dry eval produced no imm result file %DATE% %TIME% >> "%WLOG%"
  echo DONE_EXIT_134 %DATE% %TIME% >> "%WLOG%"
  exit /b 134
)
echo dry run clean -- launching w10clk %DATE% %TIME% >> "%WLOG%"
call %R%\scratchpad\w10clk\run_w10clk.cmd
echo run_w10clk.cmd returned %ERRORLEVEL% %DATE% %TIME% >> "%WLOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%WLOG%"
endlocal & exit /b 0
