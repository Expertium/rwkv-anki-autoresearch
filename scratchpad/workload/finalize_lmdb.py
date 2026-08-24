"""Swap an already-copied-and-verified LMDB over to its junction.

SPLIT OUT OF move_lmdb.py FOR A CONCRETE REASON: that script verified the source (opening an
LMDB env on it) and then tried to rename it in the same process. Windows refused with
Access Denied -- the handle from the verification pass was still associated with the
directory. This script never opens the source, so the rename is clean.

Order still protects the data at every step:
    rename original -> .old   (instant, same volume)
    junction at the original path -> C: copy
    read THROUGH the junction and compare the entry count
    only then delete .old; any failure restores the original and exits non-zero.

Usage: .venv/Scripts/python.exe scratchpad/workload/finalize_lmdb.py <name>
"""
import os
import shutil
import subprocess
import sys

import lmdb

SRC_ROOT = r"F:\rwkv_lmdb"
DST_ROOT = r"C:\rwkv_lmdb"
MAP_SIZE = 400_000_000_000


def entries(path):
    env = lmdb.open(path, map_size=MAP_SIZE, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        n = txn.stat()["entries"]
    env.close()
    return n


def free_gb(drive):
    import ctypes
    b = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(drive), None, None, ctypes.pointer(b))
    return b.value / 1e9


def main():
    name = sys.argv[1]
    src, dst, old = os.path.join(SRC_ROOT, name), os.path.join(DST_ROOT, name), \
        os.path.join(SRC_ROOT, name + ".old")
    assert os.path.isdir(dst), "destination copy missing: %s" % dst
    assert os.path.isdir(src), "source missing: %s" % src
    # the destination is the only thing opened before the rename
    n_dst = entries(dst)
    print("destination %s: %d entries" % (name, n_dst), flush=True)

    os.rename(src, old)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", src, dst], capture_output=True, text=True)
    print((r.stdout or r.stderr).strip(), flush=True)
    if not os.path.isdir(src):
        os.rename(old, src)
        print("JUNCTION FAILED -- original restored, nothing lost")
        sys.exit(1)

    n_via = entries(src)
    print("through junction: %d entries  %s"
          % (n_via, "OK" if n_via == n_dst else "*** MISMATCH"), flush=True)
    if n_via != n_dst:
        os.remove(src)
        os.rename(old, src)
        print("JUNCTION READ FAILED -- original restored, nothing lost")
        sys.exit(1)

    shutil.rmtree(old, ignore_errors=True)
    still = " (WARNING: .old not fully removed)" if os.path.isdir(old) else ""
    print("done%s: C: free %.1f GB, F: free %.1f GB" % (still, free_gb("C:\\"), free_gb("F:\\")),
          flush=True)


if __name__ == "__main__":
    main()
