@echo off
REM ===========================================================================================
REM Refit the WKV joint-uv codebook for THIS trunk (d=80, H=5, K=16). CPU-only.
REM
REM Replaces reference/pq_cb_wkv_q72u.txt, which was fitted on the d=32/H=2 model and measures
REM WORSE THAN RANDOM here (held-out mean relative L2: old 0.9985, random-1024 0.9576, zero-code
REM 1.0000, refit 0.3032). See wkv_cb_staleness.py and research_5k_notes.md.
REM
REM SAME BIT BUDGET ON PURPOSE: bits=10 -> 1024 centroids -> 10 index bits per head per layer,
REM exactly what q72u spends. So this is a drop-in swap that changes fidelity and NOT deploy state
REM size -- the comparison against the recorded tax stays honest, and champion_5k.json's frozen
REM 9 B/card / 27 B/note deploy truth is unaffected in size terms.
REM
REM Fitted on card AND note states together, because RWKV_QAT_PQ is a single catalog shared by
REM every scoped stream -- fitting them separately would not be deployable under the current env.
REM
REM Full Lloyd (not MiniBatch) is affordable at ncent=1024; the shift side needed MiniBatch only
REM because ncent=4096 over ~150k vectors ran for hours per chunk.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\fit_wkv.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set LOKY_MAX_CPU_COUNT=6

echo ===== FIT WKV CB START %DATE% %TIME% ===== > "%LOG%"

echo --- staleness, HELD-OUT USER 156 (the honest cross-user test) --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/qat_tax/wkv_cb_staleness.py "%DIR%\corpus\wkv_*.txt" --bits 10 --h 5 --k 16 --holdout-user 156 >> "%LOG%" 2>&1
echo STALENESS_USER_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"

echo --- staleness, random-vector holdout (comparable to the single-user preview) --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/qat_tax/wkv_cb_staleness.py "%DIR%\corpus\wkv_*.txt" --bits 10 --h 5 --k 16 --holdout 8000 >> "%LOG%" 2>&1
echo STALENESS_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"

echo --- fit the deployable catalog (all data, bits=10) --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/pq_train_juv.py reference/pq_cb_wkv_c80_b10.txt %DIR%\corpus\wkv_*.txt --k 16 --h 5 --bits 10 >> "%LOG%" 2>&1
echo FIT_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"

echo FIT_WKV_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
