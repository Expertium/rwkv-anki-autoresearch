@echo off
REM Cold-GPU watch (Andrew 2026-09-20: "you don't need an LLM for a check like this"). Run by the
REM Windows task ClaudeGpuColdWatch every 30 minutes through run_gpu_cold_watch_hidden.vbs, so no
REM console window appears. It runs one script and is silent unless that script says COLD.
REM   exit 0 = both GPUs warm or finished -> one line in the log, nothing else
REM   exit 2 = a GPU is idle with work outstanding -> a balloon for Andrew + scratchpad\GPU_COLD.txt
REM The marker file is how the next Claude session learns about an alarm it slept through; the watch
REM never restarts anything, because today proved the repair depends on which failure it was.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u scratchpad\gpu_cold_check.py --low-util-threshold 40 --low-util-minutes 20 > scratchpad\gpu_cold_last.txt 2>&1
set RC=%ERRORLEVEL%
echo %DATE% %TIME% rc=%RC% >> scratchpad\gpu_cold_watch.log
if not "%RC%"=="2" exit /b 0
type scratchpad\gpu_cold_last.txt >> scratchpad\gpu_cold_watch.log
copy /y scratchpad\gpu_cold_last.txt scratchpad\GPU_COLD.txt > nul
powershell -NoProfile -WindowStyle Hidden -Command "Add-Type -AssemblyName System.Windows.Forms; $n = New-Object System.Windows.Forms.NotifyIcon; $n.Icon = [System.Drawing.SystemIcons]::Warning; $n.Visible = $true; $n.ShowBalloonTip(20000, 'GPU utilization alert', ((Get-Content 'C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\GPU_COLD.txt' | Select-Object -Last 3) -join ' '), 'Warning'); Start-Sleep -Seconds 20; $n.Dispose()"
exit /b 0
