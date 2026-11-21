"""One-shot: append the autorun lock pattern to .gitignore, preserving its CRLF endings."""
import pathlib

P = pathlib.Path(".gitignore")
with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()

block = (
    "\r\n"
    "# The hourly supervisor's own single-instance lock (autorun/run_autorun.cmd holds it while it\r\n"
    "# runs; autorun/autorun_lock.py writes and deletes it).\r\n"
    "scratchpad/quant_plan/autorun/.autorun.lock\r\n"
)
if "autorun/.autorun.lock" in text:
    raise SystemExit("already present")
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text + block)
print("appended; lines now", len(P.read_text(encoding="utf-8").splitlines()))
