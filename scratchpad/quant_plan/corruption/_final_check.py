"""Final check: the corruption cell of rows 0 and 15-18 in RESULTS.md, and the git state."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
text = (ROOT / "scratchpad/quant_plan/RESULTS.md").open(encoding="utf-8", errors="replace").read()
for line in text.splitlines():
    parts = line.split("|")
    if len(parts) == 14 and parts[1].strip() in {"0", "15", "16", "17", "18", "23"}:
        print(f"row {parts[1].strip():>2} -> {parts[12].strip()[:95]}")

print()
out = subprocess.run(["git", "log", "--oneline", "-6"], cwd=ROOT, capture_output=True, text=True)
print(out.stdout.strip())
out = subprocess.run(["git", "status", "--short", "--untracked-files=no", "--",
                      "scratchpad/quant_plan", "optimization", "AGENTS.md"],
                     cwd=ROOT, capture_output=True, text=True)
print("uncommitted tracked changes in the relevant paths:", repr(out.stdout.strip()))
