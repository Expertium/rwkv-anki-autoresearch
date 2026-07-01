"""Tidy CLAUDE.md: keep sections 0-11 (lines 1-310) byte-exact, archive the bloated optimization-state
section (line 311+) verbatim into optimization/HISTORY.md, and replace it with the authored tight section
in scratchpad/new_state.md. Asserts the boundary so a wrong cut aborts without writing."""
import io

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
CLAUDE = ROOT + r"\CLAUDE.md"
HIST = ROOT + r"\optimization\HISTORY.md"
NEWSTATE = ROOT + r"\scratchpad\new_state.md"

with io.open(CLAUDE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Boundary: line 311 (index 310) must be the optimization-state header.
BOUNDARY = 310
assert lines[BOUNDARY].startswith("## Optimization state"), \
    f"BOUNDARY WRONG: line {BOUNDARY+1} = {lines[BOUNDARY]!r} -- ABORTING, nothing written"
assert lines[BOUNDARY - 1].strip() == "", \
    f"expected blank line before boundary, got {lines[BOUNDARY-1]!r} -- ABORTING"

head = lines[:BOUNDARY]            # lines 1-310 (sections 0-11), byte-exact
old_state = lines[BOUNDARY:]       # line 311+ (the section to archive)

with io.open(NEWSTATE, "r", encoding="utf-8") as f:
    new_state = f.read()

# 1) Archive old_state verbatim into HISTORY.md (additive append).
archive_header = (
    "\n\n---\n\n"
    "## CLAUDE.md optimization-state snapshot (archived 2026-06-30 tidy)\n\n"
    "Verbatim copy of the `## Optimization state` section as it stood in CLAUDE.md before the "
    "2026-06-30 declutter. Superseded plans (NEW PHASE PLAN, deck/preset-grow RESUME, stateful-BPTT "
    "ROUTE-R narrative, step-4 groundwork, old active agenda) and the full iter36/iter45 champion "
    "lineage live here now; CLAUDE.md keeps only the current champion + gate + compact lesson bank.\n\n"
)
with io.open(HIST, "a", encoding="utf-8") as f:
    f.write(archive_header)
    f.write("".join(old_state))

# 2) Rewrite CLAUDE.md = head (1-310) + authored new_state.
with io.open(CLAUDE, "w", encoding="utf-8", newline="") as f:
    f.write("".join(head))
    if not new_state.startswith("\n"):
        pass
    f.write(new_state)

print(f"OK: CLAUDE.md head kept {len(head)} lines (1-{BOUNDARY}); archived {len(old_state)} lines to HISTORY.md")
import os
print(f"CLAUDE.md now {sum(1 for _ in open(CLAUDE, encoding='utf-8'))} lines; "
      f"HISTORY.md now {sum(1 for _ in open(HIST, encoding='utf-8'))} lines")
