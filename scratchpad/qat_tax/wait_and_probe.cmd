@echo off
REM Wait for the running 500-user PTQ arm to finish, then run the 4-arm probe matrix.
REM
REM The arm's sequencer cmd was killed deliberately (to stop the second catalog's arm launching),
REM so run_arm.cmd's own completion lines will never be written -- eval_sharded is running orphaned
REM and finishes on its own. Its MERGED output file is therefore the completion signal: it is
REM written only at the very end, after every user is scored and the shards are merged.
REM
REM Waiting on a FILE, not a log token, so the anchored-findstr trap does not apply here.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\wait_probe.log
set DONEFILE=C:\Users\Andrew\rwkv-anki-autoresearch\result\RWKV-P-qtax_m2b12_ptq.jsonl
echo waiting for the 500-user PTQ arm %DATE% %TIME% > "%LOG%"

:wait
if exist "%DONEFILE%" goto ready
ping -n 31 127.0.0.1 >nul
goto wait

:ready
REM the merged file appears at the end of eval_sharded; give the process a moment to exit fully so
REM the probe does not briefly share the GPU with it.
ping -n 61 127.0.0.1 >nul
echo PTQ arm done, starting the probe matrix %DATE% %TIME% >> "%LOG%"
call scratchpad\qat_tax\probe_cbs.cmd
echo probe matrix returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo PROBECHAIN_EXIT_0 %DATE% %TIME% >> "%LOG%"
