@echo off
REM ============================================================================
REM HP TUNER LOOP -- MAX=65536 era (2026-07-30, Andrew: "Accept it, do compaction
REM and then run the HP tuner").
REM
REM Recovers the -0.000264 ahead / -0.000307 imm that MAX=65536 cost by halving the
REM optimizer steps per epoch (groups 22,346 -> 10,935) at unchanged LR. Lever order
REM leads with the learning rates because that is what the batch change implicates.
REM
REM Self-driving + RESUMABLE: hp_tuner_5k.py loop replays optimization/tuner_5k_log.jsonl
REM on every restart, so a teardown or reboot continues from the next unrun trial.
REM Each trial .cmd self-records; the loop stops if one fails to record.
REM
REM 11 non-default grid points x ~4.4 h = ~48 h if nothing prunes. Val-based early
REM pruning (vs optimization/tuner65k_vprune_ref.json, built from the maxval run) kills
REM a diverging LR trial by ~step 3000 instead of burning the full 4.4 h.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\tuner65k\tuner_loop.log

echo ===== TUNER LOOP START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u optimization/hp_tuner_5k.py loop >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
