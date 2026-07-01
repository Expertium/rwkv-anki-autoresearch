@echo off
REM Detached driver for the 5k-phase HP tuner (proxy: 1500 users, 2 WS + 0.5 decay, H=2/K=16).
REM Runs the resumable coordinate-descent loop -> each trial self-records to optimization/tuner_5k_log.jsonl.
REM Survives Esc/teardown (launched via detach.ps1 -> WmiPrvSE). Monitor scratchpad/hp5k_loop.log +
REM the per-trial scratchpad/tuner5k/<name>.log. The loop is resumable: if killed, just relaunch.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hp5k_loop.log
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
echo ===== HP5K TUNER LOOP START %DATE% %TIME% ===== >> "%LOG%"
.venv\Scripts\python.exe -u optimization/hp_tuner_5k.py loop >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
