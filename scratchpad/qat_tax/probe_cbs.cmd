@echo off
REM ===========================================================================================
REM CHEAP PTQ PROBE MATRIX (10 users, ~3-4 min each) -- where does the quantization cost live?
REM
REM All four arms evaluate the SAME plain iter-45 checkpoint, changing only what is quantized:
REM
REM   oldwkv   old q72u WKV catalog + m2b12 shift   -- today's recipe. Reference: the same 10
REM                                                    users gave +0.009276 ahead / +0.012690 imm
REM   newwkv   REFIT d=80 WKV catalog + m2b12 shift -- the proposed fix, identical bit budget
REM   wkvonly  REFIT WKV, shift NOT quantized       -- isolates the WKV half
REM   shonly   m2b12 shift, WKV NOT quantized       -- isolates the shift half
REM
REM Together these answer, for ~15 min of GPU, what the 11 h three-cell chain cannot: whether the
REM tax is dominated by the WKV side (and therefore by a catalog that measured WORSE THAN RANDOM
REM on this trunk -- old 0.9985 vs random-1024 0.9576) or by the shift side.
REM
REM wkvonly + shonly vs newwkv also tests whether the two costs are additive; if the pair costs
REM much more than the sum, the errors interact and per-side tuning will mislead.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set SRCREL=scratchpad/iter45_kddecay
set LOG=%DIR%\probe_cbs.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set RWKV_NO_JIT=1
set RWKV_EVAL_PAVA=1

echo ===== PROBE MATRIX START %DATE% %TIME% ===== > "%LOG%"

call :arm oldwkv reference/pq_cb_wkv_q72u.txt    reference/pq_cb_shift_c80_m2b12.txt
call :arm newwkv reference/pq_cb_wkv_c80_b10.txt reference/pq_cb_shift_c80_m2b12.txt
call :arm wkvonly reference/pq_cb_wkv_c80_b10.txt NONE
call :arm shonly NONE                            reference/pq_cb_shift_c80_m2b12.txt

echo PROBEMATRIX_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0

REM ---- :arm <TAG> <WKV_CB|NONE> <SHIFT_CB|NONE> ----------------------------------------------
:arm
setlocal
set TAG=%~1
set WCB=%~2
set SCB=%~3
REM NONE means that half is left in fp32: clear BOTH the catalog and the scope, or the scope alone
REM would still fake-quant without a codebook and the arm would not mean what its name says.
if "%WCB%"=="NONE" (set RWKV_QAT_PQ=& set RWKV_QAT_LOWRANK_SCOPE=) else (set RWKV_QAT_PQ=%WCB%& set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4)
if "%SCB%"=="NONE" (set RWKV_QAT_SHIFT_PQ=& set RWKV_QAT_SHIFT_SCOPE=) else (set RWKV_QAT_SHIFT_PQ=%SCB%& set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3)
echo --- arm %TAG%: wkv=%WCB% shift=%SCB% --- >> "%LOG%"
.venv\Scripts\python.exe -u scratchpad/qat_tax/assert_qat_live.py >> "%LOG%" 2>&1
.venv\Scripts\python.exe scratchpad/write_eval_toml.py %SRCREL% i45_d %DIR%\probe_%TAG%.toml RWKV-probe_%TAG% RWKV-P-probe_%TAG% 5001 5010 >> "%LOG%" 2>&1
.venv\Scripts\python.exe -u -m rwkv.get_result --config %DIR%\probe_%TAG%.toml > "%DIR%\probe_%TAG%_%RANDOM%.log" 2>&1
echo ARM_%TAG%_EXIT_%ERRORLEVEL% %TIME% >> "%LOG%"
endlocal & exit /b 0
