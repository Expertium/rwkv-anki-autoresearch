@echo off
REM ===========================================================================================
REM Fit TWO C=80 shift-PQ catalogs on the SAME corpus (Andrew 2026-08-12: "do both").
REM
REM   m2b12 : m=2 x 12 bits = 24 b per shift vector -- the SAME bit budget as the shipped q72u
REM           catalog spent at C=32. Card state ~23 B. Answers "what does QAT cost at today's
REM           budget?"
REM   m5b12 : m=5 x 12 bits = 60 b per shift vector -- the same bits PER DIMENSION as q72u had
REM           (24 b for 32 dims). Card state ~37 B. Answers "what does QAT cost when the
REM           quantizer is not starved?"
REM
REM The shipped q72u shift catalog is C=32-shaped and hard-fails on this model, so a refit is
REM mandatory regardless; the free choice is capacity, hence the pair.
REM
REM --holdout 2000: 2000 vectors per role are kept OUT of the fit and scored for mean relative L2
REM reconstruction error. At ncent=4096 a fit-set error is meaningless (the catalog can memorize);
REM held-out error is what says whether a bit budget captures this model's shift distribution, and
REM it costs no GPU. This is the number that predicts the logloss arms before we spend ~7 h each.
REM
REM --minibatch 1: full Lloyd at ncent=4096 over ~150k vectors is hours PER CHUNK. MiniBatchKMeans
REM with a fixed seed is the standard substitution; both catalogs get identical treatment, so the
REM CONTRAST between them is unaffected even though absolute error is slightly pessimistic.
REM Defaults in pq_train_shift.py are unchanged, so the shipped catalogs remain reproducible.
REM
REM Thread-limited: the GPU eval is co-tenant and its fetch workers need CPU headroom.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\fit_shift.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set LOKY_MAX_CPU_COUNT=6

echo ===== FIT shift catalogs START %DATE% %TIME% ===== > "%LOG%"

echo --- m2b12 (24 b/vector, ~23 B card) --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/pq_train_shift.py reference/pq_cb_shift_c80_m2b12.txt %DIR%\corpus\shift_*.txt --c 80 --m 2 --bits 12 --minibatch 1 --ninit 3 --maxiter 100 --holdout 2000 >> "%LOG%" 2>&1
echo M2_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"

echo --- m5b12 (60 b/vector, ~37 B card) --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/pq_train_shift.py reference/pq_cb_shift_c80_m5b12.txt %DIR%\corpus\shift_*.txt --c 80 --m 5 --bits 12 --minibatch 1 --ninit 3 --maxiter 100 --holdout 2000 >> "%LOG%" 2>&1
echo M5_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"

echo FIT_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
