@echo off
REM Launcher: detach.ps1 takes only -Script and passes no arguments, so the parameterised chain
REM needs a zero-arg wrapper. Keeps run_qat_chain.cmd reusable for a second catalog later.
REM
REM Catalogs, both chosen from the 4-arm probe matrix:
REM   WKV   = reference/pq_cb_wkv_c80_b10.txt  (the d=80 refit; set inside run_qat_chain.cmd)
REM   shift = reference/pq_cb_shift_c80_m2b12.txt (the CHEAP one -- the whole shift-side PTQ cost
REM           is +0.000365/+0.000720, ~1/14th of the WKV side, so m5b12's extra bytes buy nothing)
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
call scratchpad\qat_tax\run_qat_chain.cmd qtaxc_m2b12 reference/pq_cb_shift_c80_m2b12.txt
