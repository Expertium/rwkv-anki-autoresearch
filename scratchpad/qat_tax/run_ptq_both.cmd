@echo off
REM PTQ arms for BOTH refitted C=80 shift catalogs: evaluate iter 45's PLAIN final UNDER the QAT
REM env, no training, 500 users each (~45 min). This tests the prediction made from the held-out
REM reconstruction errors BEFORE committing ~7 h to a full QAT fine-tune:
REM
REM   m2b12  24 b/shift-vector, ~23 B card, held-out err 0.1902/0.1601
REM   m5b12  60 b/shift-vector, ~37 B card, held-out err 0.1734/0.1465
REM
REM PREDICTION (pre-registered): only ~9% apart in reconstruction error, so their logloss should be
REM close. If it holds, deploy keeps the cheap ~23 B card and the tax is not shift starvation --
REM pointing reduction work at the WKV side or at QAT placement (the decay-only-vs-throughout
REM hypothesis). If they diverge, logloss is far more sensitive to shift fidelity than
REM reconstruction error implies, which is itself worth knowing.
REM
REM Both are compared against iter 45's PLAIN numbers restricted to the same 500 users.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\qat_tax.log
echo ===== PTQ BOTH START %DATE% %TIME% ===== >> "%LOG%"
call scratchpad\qat_tax\run_arm.cmd qtax_m2b12_ptq reference/pq_cb_shift_c80_m2b12.txt ptq
echo PTQ_M2_RC_%ERRORLEVEL% %TIME% >> "%LOG%"
call scratchpad\qat_tax\run_arm.cmd qtax_m5b12_ptq reference/pq_cb_shift_c80_m5b12.txt ptq
echo PTQ_M5_RC_%ERRORLEVEL% %TIME% >> "%LOG%"
echo PTQ_BOTH_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
