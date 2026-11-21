"""Single-instance lock for the hourly quant supervisor.

    check   -> exit 0 when free (or the lock is stale), exit 3 when a fresh lock is held
    take    -> write the lock (timestamp + pid)
    release -> delete the lock
    status  -> print the lock state, always exit 0

The interactive session checks this lock before it edits RESULTS.md, NOTES.md or the run board,
so the two never write the same document at the same time.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / ".autorun.lock"
STALE_MIN = 90


def age_min():
    return (time.time() - LOCK.stat().st_mtime) / 60


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "check":
        if LOCK.exists() and age_min() <= STALE_MIN:
            print(f"lock held ({LOCK.read_text(encoding='utf-8', errors='replace').strip()}), "
                  f"age {age_min():.0f} min -- refusing")
            return 3
        if LOCK.exists():
            print(f"stale lock (age {age_min():.0f} min) -- taking over")
        return 0
    if cmd == "take":
        LOCK.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} "
                        f"run=autorun", encoding="utf-8")
        print(f"lock taken: {LOCK}")
        return 0
    if cmd == "release":
        if LOCK.exists():
            LOCK.unlink()
            print("lock released")
        return 0
    if cmd == "status":
        if LOCK.exists():
            print(f"lock HELD age {age_min():.0f} min: "
                  f"{LOCK.read_text(encoding='utf-8', errors='replace').strip()}")
        else:
            print("lock free")
        return 0
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
