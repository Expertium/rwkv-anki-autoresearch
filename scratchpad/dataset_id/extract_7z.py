"""Extract revlogs.7z (10k per-user .revlog protobufs) with a completion marker for
resume-safety. py7zr cannot resume mid-archive, so an interrupted extraction restarts;
the marker file makes a completed extraction skip on relaunch."""

import sys
from pathlib import Path

import py7zr

archive = Path(sys.argv[1])
dest = Path(sys.argv[2])
marker = dest / ".extract_complete"

if marker.exists():
    print("extraction marker present -- skipping", flush=True)
    sys.exit(0)

dest.mkdir(parents=True, exist_ok=True)
print(f"extracting {archive} -> {dest}", flush=True)
with py7zr.SevenZipFile(archive, mode="r") as z:
    z.extractall(path=dest)
n = len(list(dest.rglob("*.revlog")))
print(f"extracted, {n} .revlog files", flush=True)
if n < 10000:
    print(f"WARNING: expected ~10000 .revlog files, got {n}", flush=True)
    sys.exit(2)
marker.write_text(f"{n} files")
print("EXTRACT_OK", flush=True)
