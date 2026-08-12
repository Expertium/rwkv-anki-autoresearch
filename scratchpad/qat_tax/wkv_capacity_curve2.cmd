@echo off
REM ===========================================================================================
REM WKV capacity curve, TAKE 2 -- bigger holdout, wider bit range. CPU-only.
REM
REM Take 1 refuted the "flat curve" prediction: bits 8 -> 0.4857, bits 10 -> 0.3973, i.e. 2 bits
REM bought 18%. That is NOT the shift side's behaviour (there, 2.5x the bits bought ~9%), so the
REM WKV joint-uv scheme is NOT saturated at the shipped budget and capacity is a live lever.
REM Take 1 also died at bits=12 fitting the ORACLE (4096 centroids > 3205 holdout points).
REM
REM Two fixes:
REM   holdout user 102 instead of 156 -- ~49k held-out vectors instead of 3,205, so the REFIT
REM     estimate is far less noisy AND the oracle stays meaningful up to bits=12.
REM   bits 8..14 -- 14 (16384 centroids) is well past any budget we would ship; it is there to
REM     show where the curve actually bends, which is what says whether 12 is worth a trade.
REM
REM COST SIDE, to read alongside: index bits are per head per layer. Card = 1 layer x 5 heads, so
REM +2 bits is +10 bits/card ~ +1.25 B on a 9 B/card budget (+14%); note = 1 layer x 5 heads on a
REM 27 B budget. So bits=12 is a real state-size trade for Andrew to weigh, not a free win --
REM which is exactly why the curve needs measuring before anyone proposes it.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\wkv_capacity2.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set LOKY_MAX_CPU_COUNT=6

echo ===== WKV CAPACITY CURVE 2 START %DATE% %TIME% ===== > "%LOG%"
for %%B in (8 10 12 14) do (
  echo --- bits=%%B --- >> "%LOG%"
  .venv\Scripts\python.exe -u scratchpad/qat_tax/wkv_cb_staleness.py "%DIR%\corpus\wkv_*.txt" --bits %%B --h 5 --k 16 --holdout-user 102 >> "%LOG%" 2>&1
  echo BITS_%%B_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"
)
echo CAPCURVE2_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
