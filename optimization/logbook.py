"""Append-only optimization logbook (protocol point 9).

- log.jsonl : machine-readable, one JSON record per iteration, INCLUDES `comment`.
- log.md    : human-readable table regenerated from log.jsonl, EXCLUDES `comment`.

Usage:
  python optimization/logbook.py add record.json   # append a record, rebuild md
  python optimization/logbook.py rebuild           # rebuild md from jsonl
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSONL = HERE / "log.jsonl"
MD = HERE / "log.md"

COLS = [
    ("number", "#"),
    ("timestamp", "timestamp"),
    ("ahead", "ahead LL"),
    ("imm", "imm LL"),
    ("params", "params"),
    ("state_kib", "state KiB"),
    ("throughput", "throughput (rev/s)"),
    ("wilcoxon_p", "wilcoxon p"),
    ("review_count_check", "size✓"),
    ("logloss_tolerance_check", "LL✓"),
    ("state_size_check", "state✓"),
    ("summary", "summary"),
]


def load():
    if not JSONL.exists():
        return []
    return [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]


def cell(rec, key):
    if key in ("ahead", "imm"):
        v = rec.get("logloss", {}).get(key)
        return f"{v:.6f}" if isinstance(v, (int, float)) else "—"
    v = rec.get(key, "")
    if key == "params" and isinstance(v, int):
        return f"{v:,}"
    if key == "throughput":
        if v in (None, ""):
            return "pending"
        return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
    if key == "wilcoxon_p":
        if v in (None, ""):
            return "n/a"
        # decimal when large, scientific when tiny
        return f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"
    return str(v)


def rebuild_md():
    recs = load()
    header = "| " + " | ".join(h for _, h in COLS) + " |"
    sep = "|" + "|".join("---" for _ in COLS) + "|"
    lines = [
        "# Optimization log (steps 4–5–7)",
        "",
        "Regenerated from `log.jsonl` (do not edit by hand). `comment` is in the jsonl only.",
        "Gates: LL not worse than iter0 by >+0.0015 (both modes); state ≤ iter0; size identical.",
        "",
        header,
        sep,
    ]
    for rec in recs:
        lines.append("| " + " | ".join(cell(rec, k) for k, _ in COLS) + " |")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rebuilt {MD} ({len(recs)} rows)")


def add(record_path):
    rec = json.loads(Path(record_path).read_text())
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"appended iteration {rec.get('number')} to {JSONL}")
    rebuild_md()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "add":
        add(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "rebuild":
        rebuild_md()
    else:
        print(__doc__)
        sys.exit(1)
