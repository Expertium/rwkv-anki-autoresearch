@echo off
REM Card-scoped delta-rule ablation. Detached via WMI so session teardown cannot kill it.
REM Unbuffered (-u) so partial results survive a kill -- the first attempt lost 7 completed
REM runs because Python buffered stdout to a file and the process was tree-killed.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hybrid100k\card_delta.log
set OMP_NUM_THREADS=1
.venv\Scripts\python.exe -u scratchpad\hybrid100k\card_delta_ablate.py 3 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "%LOG%"
endlocal
