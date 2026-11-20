"""Health check for a TA3 (or plain) trainer log. Read-only; does not touch the run.

Usage: .venv/Scripts/python.exe scratchpad/ta3/logcheck.py <log> [<log> ...]
Reports step counts, TA3-vs-plain step lines, aborts/skips/tracebacks, validations, the
last checkpoint written, and an (rough) step rate from the log's own progress lines.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
STEP = re.compile(r"^(\d+) (\d+) (\d+)( \[ta3\])?, all:")


def check(path):
    p = Path(path)
    if not p.exists():
        print(f"MISSING {p}")
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    ta3, plain = [], []
    for line in lines:
        m = STEP.match(line)
        if m:
            (ta3 if m.group(4) else plain).append(int(m.group(3)))
    val = re.findall(r"Mean ahead validation loss:\s*([0-9.eE+-]+) \(([0-9.eE+-]+)\), imm:\s*([0-9.eE+-]+)",
                     text)
    ckpt = re.findall(r"Saved model to (\S+)", text)
    saves = text.count("MODEL SAVED.")
    bad = {k: text.count(k) for k in (
        "Traceback (most recent call last)", "Exception caught", "skipping optimizer step",
        "TA3 step failed", "TA3 measure-only step failed", "Skipping batch", "Non-finite")}
    last = max(ta3 + plain + [0])
    print(f"== {p}")
    print(f"   step lines: ta3={len(ta3)} plain={len(plain)} last_step={last}")
    print(f"   first/last step lines: {ta3[:1] + plain[:1]} .. {ta3[-1:] + plain[-1:]}")
    print(f"   validations={len(val)} last={val[-1] if val else None}")
    print(f"   checkpoints written={len(ckpt)} last={ckpt[-1] if ckpt else None}")
    print(f"   MODEL SAVED. count={saves}  (printed at every validation checkpoint, not only at the end)")
    print("   problem markers: " + ", ".join(f"{k}={v}" for k, v in bad.items()))
    if any(bad.values()):
        print("   !! inspect before trusting this run")
    mtimes = p.stat().st_mtime
    import datetime
    print(f"   log mtime: {datetime.datetime.fromtimestamp(mtimes):%Y-%m-%d %H:%M:%S}")
    tail = [l for l in lines if l.strip()][-3:]
    for l in tail:
        print("   | " + l[:160])


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        check(arg)
