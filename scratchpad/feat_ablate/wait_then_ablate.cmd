@echo off
REM =========================================================================================
REM Run the feature ablation once gen4base is out of the GPU.
REM
REM NOT run concurrently, deliberately. gen4base is gate-critical -- it establishes the -id
REM lineage's champion and its size baseline -- and CLAUDE.md's rule is explicit: no co-tenant
REM GPU work during gate-critical runs, because cuBLAS algo selection under memory pressure
REM breaks bit-replay. The ablation is ~1.5 h of GPU and can wait.
REM
REM It does NOT require gen4base to have succeeded: the ablation reads featB's checkpoint and
REM featB's results, so the two runs are independent. It waits only for the GPU.
REM
REM Anchored findstr -- the unanchored form matches this file's own prose and fires instantly.
REM No angle brackets or arrows in REM lines.
REM =========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_ablate\wait_ablate.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\gen4_base\gen4base.log

echo ===== ablation waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo gen4base reported %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\feat_ablate\run_ablate.cmd
echo run_ablate returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
