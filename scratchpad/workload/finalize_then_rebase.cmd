@echo off
REM ===========================================================================================
REM Wait for the e2s train db copy to verify, swap in the junction, then start the re-base.
REM
REM Andrew authorised deleting train_db_5k_h1 and train_db_5k_h1_fix (2026-08-30), which freed
REM 194.4 GB on C:. The e2s train db is being copied to the SSD because the measured penalty for
REM reading it from the USB drive is 2.2x on every step: the original C:-hosted teacher dump ran
REM at 1.40 steps/s, the same dump on F: managed 0.63, and GPU utilisation during it was 8% --
REM starved on reads, not computing.
REM
REM TWO WITNESSES before finalizing, because one is not enough: the log must say VERIFY OK AND
REM the copy process must be gone. A log line alone could come from a process that then died;
REM an exited process alone proves nothing about whether it succeeded.
REM
REM finalize_lmdb.py is a SEPARATE script on purpose -- move_lmdb.py verified the source by
REM opening an LMDB env on it and then could not rename it, because its own handle was still
REM associated with the directory. This one never opens the source.
REM
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload\finalize_chain.log
set MOVELOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\workload\move_e2s.log

echo ===== finalize chain armed %DATE% %TIME% ===== >> "%LOG%"

:waitcopy
findstr /B /C:"VERIFY OK" "%MOVELOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 61 127.0.0.1 >nul
  goto waitcopy
)
echo copy verified %DATE% %TIME% >> "%LOG%"

REM second witness: the copy process must have exited before we touch either directory
:waitproc
tasklist /FI "IMAGENAME eq python.exe" /FO CSV | findstr /C:"python.exe" >nul 2>&1
if not errorlevel 1 (
  wmic process where "name='python.exe'" get commandline 2>nul | findstr /C:"move_lmdb" >nul 2>&1
  if not errorlevel 1 (
    ping -n 31 127.0.0.1 >nul
    goto waitproc
  )
)
echo copy process gone %DATE% %TIME% >> "%LOG%"

.venv\Scripts\python.exe scratchpad/workload/finalize_lmdb.py train_db_5k_h1_e2s >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo FINALIZE_FAILED -- original restored by the tool, nothing lost >> "%LOG%"
  echo DONE_EXIT_51 %DATE% %TIME% >> "%LOG%"
  exit /b 51
)

REM The junction must resolve and carry the right row count, or the re-base would train on
REM whatever is at that path. Checked THROUGH the F: path, which is what every toml uses.
.venv\Scripts\python.exe scratchpad/features_rebuild/check_db.py F:/rwkv_lmdb/train_db_5k_h1_e2s 1483984 24 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo JUNCTION_CHECK_FAILED >> "%LOG%"
  echo DONE_EXIT_52 %DATE% %TIME% >> "%LOG%"
  exit /b 52
)
echo JUNCTION OK -- train db now reads from the SSD %DATE% %TIME% >> "%LOG%"

call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\e2s_rebase\run_e2s_rebase.cmd
echo re-base returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
