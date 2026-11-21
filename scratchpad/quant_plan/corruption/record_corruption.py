"""Fill the `corruption` column of RESULTS.md from the corruption verdict files (deterministic).

Why this exists: the hourly supervisor may edit RESULTS.md (QUEUE.md section 4) and the interactive
session does too, and the main table now has 12 cells per row. A hand edit can drop a cell. This script
is the only writer of the corruption CELLS: it reads `corruption_corrupt_<tag>.json` (written by
`check_corruption.py`), finds that tag's RESULTS.md row, and replaces the LAST cell of that row, and
nothing else.

Row lookup, in order:
  1. the explicit ROWS map below (the rows that existed when the column was added on 2026-11-21);
  2. otherwise, the row whose LINE contains the tag exactly once -- so a row added later must name its
     candidate tag in its Method cell (QUEUE.md section 4 requires that);
  3. otherwise nothing is written for that tag and this script says so, so a missing row is visible.

It refuses to write a row that does not have exactly 12 cells (that would mean the table shape changed
under it), and it skips smoke evidence files.

Usage:
  .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\record_corruption.py            # write
  .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\record_corruption.py --dry-run  # report
"""
import argparse
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../scratchpad/quant_plan/corruption
RUN = HERE.parent                                # .../scratchpad/quant_plan
RESULTS = RUN / "RESULTS.md"
N_CELLS = 12

# verdict tag -> the iteration number of its RESULTS.md row.
ROWS = {
    "ctl_fp32": "0",   # the fp32 control of row 0
    "l2i8": "15",
    "l3i8": "16",
    "l2i8s4": "17",
    "l3i8s4": "18",
    "l2i4s4": "23",
}


def cell_for(rec):
    status = rec.get("status", "?")
    calls = rec.get("state_calls") or 0
    p_calls = rec.get("p_calls") or 0
    mins = (rec.get("secs") or 0) / 60
    users = rec.get("users_scored")
    if status == "PASS":
        if rec.get("state_observed"):
            return (f"**PASS** ({calls:,} state calls, {p_calls} p(recall) tensors, 0 non-finite, "
                    f"{mins:.1f} min on the {users}-user roster)")
        return (f"**PASS** ({p_calls} p(recall) tensors, 0 non-finite; no quantized stream in this "
                f"config, so only p(recall) is observable - the attribution control)")
    if status == "FAIL":
        off = rec.get("offenders") or []
        who = ", ".join(str(o.get("user")) for o in off[:6]) or "?"
        detail = "; ".join(rec.get("problems") or [])
        return f"**FAIL**: {detail}; offenders: {who} ({len(off)} users)"
    detail = "; ".join(rec.get("problems") or [])
    return f"**INCOMPLETE** (not a pass): {detail}"


def find_row(lines, tag):
    num = ROWS.get(tag)
    if num is not None:
        pat = re.compile(rf"^\|\s*{re.escape(num)}\s*\|")
        hits = [i for i, l in enumerate(lines) if pat.match(l)]
        if len(hits) == 1:
            return hits[0]
    hits = [i for i, l in enumerate(lines)
            if l.startswith("|") and re.search(rf"\b{re.escape(tag)}\b", l)]
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    lines = RESULTS.read_bytes().decode("utf-8").split("\n")
    verdicts = sorted(glob.glob(str(HERE / "corruption_corrupt_*.json")))
    changed = 0
    for path in verdicts:
        p = Path(path)
        if "smoke" in p.stem:
            continue                      # smoke evidence, not a roster verdict
        rec = json.loads(p.read_text(encoding="utf-8"))
        tag = rec.get("name") or p.stem.replace("corruption_corrupt_", "")
        idx = find_row(lines, tag)
        if idx is None:
            print(f"{tag:12s} {rec.get('status', '?'):10s} -> no unique RESULTS.md row yet (write the "
                  f"row first, or name the tag in its Method cell)")
            continue
        cells = lines[idx].split("|")
        if len(cells) != N_CELLS + 2:
            print(f"{tag:12s} SKIPPED: row {idx + 1} has {len(cells) - 2} cells, expected {N_CELLS}")
            continue
        new = cell_for(rec)
        if cells[-2].strip() == new.strip():
            print(f"{tag:12s} {rec.get('status', '?'):10s} -> row {idx + 1} already current")
            continue
        cr = "\r" if lines[idx].endswith("\r") else ""
        cells[-2] = f" {new} "
        lines[idx] = "|".join(cells).rstrip("\r") + cr
        changed += 1
        print(f"{tag:12s} {rec.get('status', '?'):10s} -> row {idx + 1}: {new[:100]}")
    if changed and not args.dry_run:
        RESULTS.write_bytes("\n".join(lines).encode("utf-8"))
        print(f"\nwrote {RESULTS} ({changed} cell(s) updated)")
    elif changed:
        print(f"\ndry run: {changed} cell(s) WOULD be updated")
    else:
        print("\nnothing to update")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
