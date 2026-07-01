@echo off
REM Self-driving greedy coordinate-descent HP tuner: runs EVERY remaining trial (compute next ->
REM train sc8k WS aug-off -> eval 101-200 -> self-record) until coordinate descent converges.
REM Resumable: replays optimization/tuner_log.jsonl on restart, so an Esc/teardown just continues
REM from the next trial. Launch DETACHED (survives teardown). Monitor scratchpad/tuner/tuner_loop.log.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner\tuner_loop.log
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
echo ===== TUNER LOOP START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u optimization/hp_tuner.py loop >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
