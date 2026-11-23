@echo off
REM Diagnostic probe: run the row-55 l2i8s4 engine on single users of trace_corrupt50 and time each.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set RWKV_WEIGHTS=scratchpad\row55_l2i8s4e1c_bundle\weights\deploy_iter55_l2i8s4e1c.safetensors
set RWKV_TRACE_DIR=scratchpad\quant_plan\trace_corrupt50
set RWKV_PRED_DIR=scratchpad\quant_plan\_row55_probe
set RWKV_STATE_LOWRANK_SCOPE=card:2:int8,note:1:int8
set RWKV_QUANT_SHIFTS=1
set RWKV_STATE_SHIFT_LEVEL=card:int4,note:int4
set RWKV_FINITE_CHECK=0
echo === users %* ===
rust\rwkv-infer\target\release\rwkv-infer.exe %*
echo probe rc=%errorlevel%
