@echo off
REM ===========================================================================================
REM THE DECISIVE CHECK for the rank-1 regularizer (qtaxf_r1reg). CPU-ONLY -- safe to run while
REM the decay trains on the GPU (no CUDA, no co-tenancy).
REM
REM (!) COMMENT STYLE: no angle brackets, ampersands, pipes or carets anywhere in REM lines.
REM cmd.exe processes REDIRECTION BEFORE it honours REM, so a comment containing an arrow or a
REM usage string with placeholder brackets is parsed as a redirect and the runner dies with
REM "'M' is not recognized" plus "was unexpected at this time". Cost one dead launch.
REM
REM WHAT IT ANSWERS. The training-loss penalty tells us the PROXY moved (k/v alignment). It does
REM NOT tell us the thing the deploy quantizer actually pays for: the exact rank-1 truncation
REM error of the WKV STATE, which the proxy only bounds indirectly (it ignores the decay
REM weighting). If logloss comes back null, these two branches need separating:
REM   floor MOVED   ==  Andrew's objection confirmed: rank-1-ness is achievable and worthless.
REM   floor UNMOVED ==  proxy and floor are decoupled; the LEVER is wrong, not the dose.
REM
REM USAGE:  run_rank1_check.cmd  CKPT_PATH  LABEL
REM   e.g.  run_rank1_check.cmd  scratchpad\iter45_kddecay\qtaxf_r1reg_d_1000.pth  r1reg_s1000
REM
REM (!) Compare ONLY against the control measured with the SAME tool on the SAME users --
REM     card 0.3733 / note 0.3729 over users 101,102,136. The ladder's 0.4353 / 0.3049 is a
REM     DIFFERENT quantity; comparing to it manufactures a 0.06-sized fake improvement.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set CKPT=%~1
set LABEL=%~2
if "%CKPT%"=="" (echo usage: run_rank1_check.cmd CKPT_PATH LABEL & exit /b 2)
if "%LABEL%"=="" (echo usage: run_rank1_check.cmd CKPT_PATH LABEL & exit /b 2)
if not exist "%CKPT%" (echo MISSING CKPT %CKPT% & exit /b 3)

set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\rank1_check_%LABEL%.log
set OUT=%DIR%\corpus_%LABEL%
set PY=.venv\Scripts\python.exe
set BIN=rust\rwkv-infer\target\release\rwkv-infer.exe
if not exist "%OUT%" mkdir "%OUT%"

REM ---- the arch the traces were exported under. Training-only change, so iter45's traces and
REM      arch env apply unchanged; only the WEIGHTS differ. RWKV_STREAM_ORDER is a RUST-ONLY flag
REM      (verified: no Python file reads it), so setting it here cannot perturb the export.
REM (!) The STRUCTURAL flags below are NOT optional. The checkpoint stores 1x1 DUMMIES for every
REM     stripped component (the 9 stripped channel mixers, the layer-0 v_lora, the old ahead head
REM     that RWKV_GRU_HEAD replaces). Omit them and the RNN model builds those full-size, so
REM     load_state_dict dies with dozens of "copying a param with shape [1, 1]" mismatches that
REM     look like an arch-version problem and are not. Cost one dead launch 2026-08-14.
set RWKV_TRACE_DIR=reference_iter45
set RWKV_INTERLEAVE=1
set RWKV_STREAM_ORDER=card,note,deck,preset,user
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_GRU_HEAD=3
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_PAVA_LAMBDA=0.2
REM NO quantization env -- the corpus must be fp32 ground truth. The floor is a property of the
REM UNQUANTIZED state; quantizing here would measure the quantizer, not the model.

echo ===== RANK1 CHECK %LABEL% START %DATE% %TIME% ===== > "%LOG%"
echo ckpt=%CKPT% >> "%LOG%"

REM ---- phase 1: weights to safetensors (CPU, seconds). Traces are weight-independent, reused.
set RWKV_CHAMP_CKPT=%CKPT%
set RWKV_CHAMP_SFT=r1check_%LABEL%.safetensors
%PY% scratchpad\export_weights_only.py >> "%LOG%" 2>&1
if errorlevel 1 (echo EXPORT_FAIL >> "%LOG%" & echo DONE_EXIT_11 >> "%LOG%" & endlocal & exit /b 11)
if not exist "reference\r1check_%LABEL%.safetensors" (echo EXPORT_NOFILE >> "%LOG%" & echo DONE_EXIT_12 >> "%LOG%" & endlocal & exit /b 12)

REM ---- phase 2: dump fp32 WKV states (CPU, roughly 6 min for 3 users by 2 streams)
set RWKV_WEIGHTS=reference/r1check_%LABEL%.safetensors
for %%U in (101 102 136) do (
  for %%S in (card note) do (
    %BIN% --dump-corpus %%U %%S 4 > "%OUT%\wkv_%%U_%%S.txt" 2>> "%LOG%"
    echo DUMP_%%U_%%S_EXIT_%%ERRORLEVEL%% >> "%LOG%"
  )
)

REM ---- phase 3: the floor, candidate then control, on the SAME users with the SAME tool
echo. >> "%LOG%"
echo ---- CANDIDATE %LABEL% ---- >> "%LOG%"
%PY% scratchpad\qat_tax\rank1_floor.py "%OUT%/wkv_*_card.txt" --label "%LABEL% card" --json "%DIR%\rank1_%LABEL%_card.json" >> "%LOG%" 2>&1
%PY% scratchpad\qat_tax\rank1_floor.py "%OUT%/wkv_*_note.txt" --label "%LABEL% note" --json "%DIR%\rank1_%LABEL%_note.json" >> "%LOG%" 2>&1
echo. >> "%LOG%"
echo ---- CONTROL iter45, same users ---- >> "%LOG%"
%PY% scratchpad\qat_tax\rank1_floor.py "%DIR%/corpus/wkv_101_card.txt" "%DIR%/corpus/wkv_102_card.txt" "%DIR%/corpus/wkv_136_card.txt" --label "control card" >> "%LOG%" 2>&1
%PY% scratchpad\qat_tax\rank1_floor.py "%DIR%/corpus/wkv_101_note.txt" "%DIR%/corpus/wkv_102_note.txt" "%DIR%/corpus/wkv_136_note.txt" --label "control note" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== RANK1 CHECK %LABEL% END %DATE% %TIME% ===== >> "%LOG%"
echo DONE_EXIT_0 >> "%LOG%"
endlocal
exit /b 0
