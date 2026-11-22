"""One-shot: fix the bundle builder's manifest hash column.

The builder's manifest loop was written with a truncated digest plus a mangled character
(`{dig[:16]}<mojibake>`), which is useless for verification. Replace that whole line with the
full 64-character SHA-256. Matching is line-based, so the mojibake byte cannot break the match.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "_build_row17_bundle.py"

NEW = '        lines.append(f"| `{rel}` | {size:,} | `{dig}` |")\n'


def main() -> int:
    text = P.open(encoding="utf-8", newline="").read()
    lines = text.splitlines(True)
    hits = [i for i, L in enumerate(lines) if "dig[:16]" in L]
    if len(hits) != 1:
        print(f"ABORT: {len(hits)} candidate lines, expected 1")
        return 1
    i = hits[0]
    print(f"old line: {lines[i]!r}")
    lines[i] = NEW
    P.write_text("".join(lines), encoding="utf-8", newline="")
    print(f"new line: {NEW!r}")
    print("patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
