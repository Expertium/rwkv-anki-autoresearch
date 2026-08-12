@echo off
REM ===========================================================================================
REM WKV catalog CAPACITY CURVE at d=80 -- bits 8 / 10 / 12, cross-user holdout. CPU-only.
REM
REM The refit at the shipped bits=10 budget lands at ~0.40 held-out cross-user, against the SHIFT
REM side's 0.19. So even a correctly-fitted WKV catalog is the lossier half, and the obvious
REM question before any further tax-reduction work is whether that is a BIT BUDGET problem or an
REM intrinsic one. 1024 centroids in 32 dims is ~1.24 points per axis; 4096 is ~1.30 -- barely
REM different, which predicts the curve is FLAT and capacity is NOT the lever.
REM
REM Stating that prediction up front so the run can refute it. The same reasoning was right on the
REM shift side (2.5x the bits bought only ~9% error) and it is the cheap way to find out here.
REM
REM What each answer would mean:
REM   FLAT (b12 ~ b10)  -> the joint-uv scheme is saturated; more bits are wasted deploy state.
REM                        Reducing the WKV half then needs a different SCHEME (finer chunks,
REM                        rank-2, residual stages), not a bigger catalog.
REM   STEEP (b12 << b10) -> capacity IS the lever, and +2 bits/head/layer costs ~1.25 B/card on a
REM                        9 B/card budget -- a real trade to put to Andrew, not a free win.
REM
REM Cross-user holdout (user 156 entirely out) because the deployed catalog is fitted on a handful
REM of users and applied to thousands. Note the ORACLE column is near-memorization at this holdout
REM size and is NOT a floor -- read REFIT vs OLD only.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\wkv_capacity.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set LOKY_MAX_CPU_COUNT=6

echo ===== WKV CAPACITY CURVE START %DATE% %TIME% ===== > "%LOG%"
for %%B in (8 10 12) do (
  echo --- bits=%%B --- >> "%LOG%"
  .venv\Scripts\python.exe -u scratchpad/qat_tax/wkv_cb_staleness.py "%DIR%\corpus\wkv_*.txt" --bits %%B --h 5 --k 16 --holdout-user 156 >> "%LOG%" 2>&1
  echo BITS_%%B_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"
)
echo CAPCURVE_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
