@echo off
REM CLEAN re-confirm of the both-low-rank PTQ deploy (card rank2:int4 + note rank2:int4 + int4 shifts) on
REM iter45 -- the preliminary number was 0.289137 (measured near an unrelated process kill). Honest gate.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\confirm_bothlr.log
set RWKV_QUANT_SHIFTS=1
echo START %DATE% %TIME% > "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/run_qat_eval.sh reference/rwkv_iter45.safetensors "" 14 "card:2:int4,note:2:int4" >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
