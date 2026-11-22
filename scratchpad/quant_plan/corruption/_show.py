"""Print a numbered slice of a file: _show.py <path> [start] [end]  (1-based, inclusive)."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
end = int(sys.argv[3]) if len(sys.argv) > 3 else start + 200
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
lines = p.open(encoding="utf-8", errors="replace").read().splitlines()
for i in range(start, min(end, len(lines)) + 1):
    print(f"{i:4d} {lines[i - 1]}")
print(f"--- {p} has {len(lines)} lines")
