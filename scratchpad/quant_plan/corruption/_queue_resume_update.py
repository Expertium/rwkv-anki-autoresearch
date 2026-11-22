"""One-shot: rewrite QUEUE.md sections 2 and 5 for the RESUMED experiment queue.

Andrew's 11:21 pause ("stop further runs for now and ONLY do corruption checks") ended when the
corruption chain finished at 11:41 with five PASSes and he resumed experiments ("Since corruption
checks are done, we can resume experiments. Sure, try rank-2 note"). The supervisor contract still
said the four killed chains were PAUSED and that it must never restart them, which would have left the
GPU idle the moment `it36noterank2` ended - the exact failure mode the contract exists to prevent.

Sections 2 and 5 are replaced wholesale (by header), the other sections are left byte-for-byte alone.
"""
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "scratchpad" / "quant_plan" / "autorun" / "QUEUE.md"
text = P.open(encoding="utf-8", newline="").read()
nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"
lines = text.split(nl)

SEC2 = """## 2. LIVE chains, and the restart rule

LIVE chains, in run order (Andrew resumed experiments on 2026-11-22 12:02: "Since corruption checks are
done, we can resume experiments. Sure, try rank-2 note"):

`it36noterank2` -> `it36shrink` -> `it36equalbits` -> `it36combo` -> `it36corrupt2`

All five are LIVE. The 11:21 pause ended when `it36corrupt` finished at 11:41 with five PASSes; the four
chains killed at 11:24 were relaunched at ~12:06 waiting behind `it36noterank2`, and `it36noterank2`
(the rank-2/3 NOTE ladder from Andrew's 12:01 direction) started at 12:05:28.

Each chain takes its own lock file and, while waiting, logs `waiting: prev chain ended=..., matching
processes=N` once a minute. So a WAITING chain's log GROWS and the 45-minute rule below never fires for
it. A chain killed by an earlier stop has no end marker and its log stops growing, which the report
shows as `STALLED?`; for the four relaunched chains that is the OLD kill, not a new stall - read the log
tail before doing anything.

The report prints a `CPU over 5 s:` line (added 2026-11-22, after an F: disk I/O error wedged
`it36shrink` for 8.7 h at 0.000 s CPU while the report still said `GPU BUSY` and every chain log looked
normal). If that line says `WEDGE? live workload processes burned no CPU`, do NOT kill anything: log
`NEEDS ANDREW: wedged workload at <time> - <pid lines>` and STOP. The chains now kill and retry such a
run by themselves (`it36ladder_lib.run`: 20 min with neither log growth nor CPU -> `taskkill /F /T`,
then one retry).

`it36corrupt` (the corruption/finiteness check) is DONE: five points, `=== it36corrupt chain end ===`
at 11:41:38, all five PASS - RESULTS.md rows 17, 15, 16, 18 and the quantization-OFF control of row 0,
50/50 users each, 158,986 raw state observations and 150 p(recall) tensors per quantized point, zero
non-finite. `it36corrupt2` (16 gate-guarded points, four of them `it36noterank2`'s) is the chain that
checks the new rows, and it is last in the queue.

A chain queued BEHIND a corruption chain must add `probe_get_result.py|check_corruption.py` to its own
BUSY_PAT, or it will start on top of the corruption run (the driver's command line matches no other
chain's pattern). `it36shrink`, `it36equalbits`, `it36combo` and `it36corrupt2` all carry those two
patterns.

The chains sequence themselves through their own lock file and their `=== <name> chain end` marker.
Normally you do nothing for them.

Restart a LIVE chain ONLY when all four conditions hold:

1. the status report says `GPU IDLE (no matching workload process)`;
2. that chain's log has no `=== <name> chain end` line;
3. that chain's log file has not changed for more than 45 minutes;
4. host RAM free is at least 10 GiB.

Then run these two commands and log the restart:

    del "scratchpad\\quant_plan\\<name>_chain.lock"
    powershell -NoProfile -File scratchpad\\detach.ps1 -Command ".venv\\Scripts\\python.exe -u scratchpad\\quant_plan\\<name>_chain.py > scratchpad\\quant_plan\\<name>_launcher.log 2>&1"

The chain skips any point whose merged result already exists, so a restart resumes cleanly. A WAITER
that stalled is safe to restart: it re-takes its lock and goes back to waiting for its predecessor.
Restart at most one chain per run. Never restart a chain while any workload process is running."""

SEC5 = """## 5. Pre-authorized launch items

None right now, and none are needed: the five LIVE chains of section 2 already own the queue in run
order, so there is no `- [ ]` item for you to launch. Do NOT launch a chain by hand and do NOT add one -
the queue drains by itself.

When the report says `GPU IDLE` while a LIVE chain is waiting (its log says `waiting: ...`), that is a
normal hand-off: the next chain starts on its own within about a minute. It is NOT an instruction to
start anything.

When `it36corrupt2` logs `=== it36corrupt2 chain end ===` and the GPU is idle, there is nothing left to
run: log `QUEUE EMPTY - needs Andrew's decision` and stop. Do NOT invent a new experiment, do NOT extend
an existing chain, and do NOT restart a finished chain.

The CPU-only Rust checks (`rust_finite.py`, `measure_finite_offpath.py`, `measure_offpath_oldbuild.py`)
also exist now, but they are the interactive session's job, NOT yours: each one ends in an artifact that
a human decision depends on.

Andrew adds an item here when he authorizes one. A real item is one raw, unindented line: the box
marker, then the task, then `::`, then the exact command line, then `::`, then the proof of success.
`status_report.py` counts only raw unindented box lines, so keep any example indented like this
paragraph:

    - [ ] <what to run> :: <exact command line> :: <what proves it worked>"""


def replace_section(lines, header, new_body, next_headers):
    start = next(i for i, l in enumerate(lines) if l.strip() == header)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() in next_headers), len(lines))
    new = new_body.split("\n")
    return lines[:start] + new + [""] + lines[end:]


lines = replace_section(lines, "## 2. LIVE chains, and the restart rule", SEC2,
                        {"## 3. Idle-only deploy measurement (run at most once)"})
lines = replace_section(lines, "## 5. Pre-authorized launch items", SEC5,
                        {"## 6. Forbidden, always"})
P.open("w", encoding="utf-8", newline="").write(nl.join(lines))
print(f"rewrote sections 2 and 5 of QUEUE.md (newline {nl!r}); file now {len(lines)} lines")

kept = [l for l in lines if l.strip().startswith("## ")]
print("section headers now:", " | ".join(h.strip() for h in kept))
if any("PAUSED by Andrew" in l for l in lines):
    print("WARNING: a PAUSED paragraph survived")
    sys.exit(1)
if "it36noterank2` -> `it36shrink" not in " ".join(lines):
    print("WARNING: the LIVE order line is missing")
    sys.exit(1)
print("verified: no PAUSED paragraph left, LIVE order line present")
