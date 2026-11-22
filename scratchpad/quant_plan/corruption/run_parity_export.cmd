@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
.venv\Scripts\python.exe -X utf8 -u scratchpad\quant_plan\corruption\export_parity_trace.py --users 107 > scratchpad\quant_plan\corruption\parity_export.log 2>&1
echo exit=%ERRORLEVEL% >> scratchpad\quant_plan\corruption\parity_export.log
