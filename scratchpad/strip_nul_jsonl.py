"""Strip the NUL padding a dirty shutdown leaves at the end of a jsonl (AGENTS.md 2026-08-17: "a dirty shutdown
truncates the chain log's last write to NUL bytes ... strip them before relaunching"). 2026-09-15: the same padding
on result/RWKV-lfd1-s0.jsonl made get_result's sort_jsonl raise JSONDecodeError and the eval resume failed.

For each file: drop every NUL byte and any trailing partial line (one without a newline, or one that is not valid
JSON), after saving a byte-exact backup <file>.nulbak once. Files without NULs and with a valid last line are left
untouched (byte-identical). Prints what it did; exit 0 always unless a file cannot be read.
Usage: python scratchpad/strip_nul_jsonl.py <jsonl> [<jsonl> ...]
"""
import json
import os
import sys

for f in sys.argv[1:]:
    if not os.path.exists(f):
        print(f"{f}: absent, nothing to strip")
        continue
    b = open(f, "rb").read()
    n_nul = b.count(b"\x00")
    body = b.replace(b"\x00", b"")
    lines = body.split(b"\n")
    tail_dropped = 0
    # the last element is "" iff the file ends with a newline; otherwise it is a partial line
    if lines and lines[-1] != b"":
        tail_dropped += 1
        lines = lines[:-1]
    # drop trailing non-JSON lines (a truncated record can still end with a newline if the padding sat inside)
    while len(lines) > 1 and lines[-2].strip():
        try:
            json.loads(lines[-2]); break
        except Exception:
            del lines[-2]; tail_dropped += 1
    out = b"\n".join(lines)
    if out == b:
        print(f"{f}: clean ({len(lines) - 1} lines), untouched")
        continue
    bak = f + ".nulbak"
    if not os.path.exists(bak):
        open(bak, "wb").write(b)
    open(f, "wb").write(out)
    print(f"{f}: removed {n_nul} NUL bytes and {tail_dropped} partial line(s); {len(lines) - 1} lines kept; backup {bak}")
