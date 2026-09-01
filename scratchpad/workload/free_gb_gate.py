"""Exit 0 if a drive has at least N GB free, non-zero otherwise. A gate, not a report.

Used by runners that want to place a database on the SSD only when there is genuinely room, and
to fall back to the slower drive otherwise rather than failing. A speed optimization must never
be able to stop an experiment.

Usage: free_gb_gate.py <drive_letter> <min_gb>
"""
import ctypes
import sys
from ctypes import wintypes


def free_gb(drive):
    free = ctypes.c_ulonglong(0)
    ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        wintypes.LPCWSTR(drive + ":\\"), ctypes.byref(free), None, None)
    if not ok:
        raise OSError("GetDiskFreeSpaceEx failed for " + drive)
    return free.value / 1024 ** 3


def main():
    drive, need = sys.argv[1].rstrip(":\\"), float(sys.argv[2])
    have = free_gb(drive)
    ok = have >= need
    print("%s: %.1f GB free, need %.1f -- %s" % (drive, have, need, "OK" if ok else "TOO SMALL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
