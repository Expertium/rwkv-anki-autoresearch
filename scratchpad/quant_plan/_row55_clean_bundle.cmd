@echo off
REM Remove the artifacts that running the bundle's own verify.cmd creates inside it.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set B=scratchpad\row55_l2i8s4e1c_bundle
if exist "%B%\code\rust\rwkv-infer\target" rmdir /s /q "%B%\code\rust\rwkv-infer\target"
for %%d in (rust_grid_preds parity_preds rust_finite_preds rust_speed_preds _row55_probe) do (
  if exist "%B%\code\scratchpad\quant_plan\%%d" rmdir /s /q "%B%\code\scratchpad\quant_plan\%%d"
)
for /r "%B%\code" %%d in (.) do @if exist "%%d\__pycache__" rmdir /s /q "%%d\__pycache__"
del /q "%B%\code\scratchpad\quant_plan\*_verify*.json" 2>nul
echo cleanup done
