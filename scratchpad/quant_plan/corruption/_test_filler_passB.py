"""One-shot test of record_corruption.py's second pass (2026-11-21 23:50).

Sets row 23's corruption cell to `pending`, runs the filler with --dry-run (it must report flipping it
to `-`, because it36l2i4s4 fails the log-loss gate), verifies the file was NOT written, then restores
the cell to `-` and confirms the file is byte-identical to the state before the test.
"""
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
RESULTS = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
FILLER = ROOT / "scratchpad" / "quant_plan" / "corruption" / "record_corruption.py"

before = RESULTS.read_bytes()
lines = before.decode("utf-8").split("\n")
hit = [i for i, l in enumerate(lines) if re.match(r"^\|\s*23\s*\|", l)]
assert len(hit) == 1, hit
i = hit[0]
assert lines[i].split("|")[-2].strip() == "-", lines[i].split("|")[-2]
cr = "\r" if lines[i].endswith("\r") else ""
cells = lines[i].split("|")
cells[-2] = " pending "
lines[i] = "|".join(cells).rstrip("\r") + cr
RESULTS.write_bytes("\n".join(lines).encode("utf-8"))
print("row 23 temporarily set to 'pending'")

env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"
out = subprocess.run([str(PY), str(FILLER), "--dry-run"], capture_output=True, text=True,
                     cwd=str(ROOT), env=env)
print("---- filler output ----")
print((out.stdout or "").strip())
print("---- end ----")

dry_ok = RESULTS.read_bytes() != before and \
    RESULTS.read_bytes().decode("utf-8").split("\n")[i].split("|")[-2].strip() == "pending"
print("--dry-run did not write:", dry_ok)

# restore
lines = RESULTS.read_bytes().decode("utf-8").split("\n")
cells = lines[i].split("|")
cells[-2] = " - "
cr = "\r" if lines[i].endswith("\r") else ""
lines[i] = "|".join(cells).rstrip("\r") + cr
RESULTS.write_bytes("\n".join(lines).encode("utf-8"))
same = RESULTS.read_bytes() == before
print("restored byte-identical to the pre-test state:", same)

ok = dry_ok and same and "gate FAIL" in (out.stdout or "")
print("TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
