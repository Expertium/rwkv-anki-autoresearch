@echo off
REM Two extra shift-catalog fits that turn a 2-point extrapolation into a curve. CPU-only.
REM
REM m2b12 (24 b/vec) = 0.1902/0.1601 and m5b12 (60 b/vec) = 0.1734/0.1465, i.e. 2.5x the bits for
REM ~9% less error. Before concluding "capacity is not the lever" off two points, ask two things:
REM
REM   m10b12 (120 b/vec, sub=8): is there ANY headroom at all, or does the scheme saturate? If
REM      5x the bits of m2b12 still lands near 0.17, the error is intrinsic to product-quantizing
REM      an 80-dim direction with 4096-entry catalogs, not a budget problem.
REM   m4b6  (24 b/vec, sub=20, 64 centroids): the SAME budget as m2b12 split differently -- more
REM      chunks, tiny catalogs each. Directly actionable: if it beats m2b12 the deploy config is
REM      simply mis-chunked and we get fidelity for free.
REM
REM Why the curse of dimensionality is the suspect: 4096 centroids in sub=40 dims is 4096^(1/40)
REM ~= 1.2 points per axis; at sub=16 it is ~1.65; at sub=8 it is ~2.7. Coverage is exponential in
REM sub, so bits bought at fixed sub barely move the error while SHRINKING sub should move it a lot.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\fit_shift_probe.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set LOKY_MAX_CPU_COUNT=6

echo ===== PROBE FITS START %DATE% %TIME% ===== > "%LOG%"
echo --- m10b12 (120 b/vector, sub=8): is there headroom? --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/pq_train_shift.py %DIR%\pq_cb_shift_c80_m10b12.txt %DIR%\corpus\shift_*.txt --c 80 --m 10 --bits 12 --minibatch 1 --ninit 3 --maxiter 100 --holdout 2000 >> "%LOG%" 2>&1
echo M10_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"

echo --- m4b6 (24 b/vector, sub=20, 64 cent): same budget, finer chunks --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/pq_train_shift.py %DIR%\pq_cb_shift_c80_m4b6.txt %DIR%\corpus\shift_*.txt --c 80 --m 4 --bits 6 --minibatch 1 --ninit 3 --maxiter 100 --holdout 2000 >> "%LOG%" 2>&1
echo M4B6_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"

echo PROBE_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
