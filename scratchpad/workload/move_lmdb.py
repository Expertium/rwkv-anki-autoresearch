"""Move one LMDB from F: to C: without breaking a single config.

WHY A JUNCTION AND NOT A PATH EDIT. The dbs are addressed by hardcoded absolute
`F:/rwkv_lmdb/...` strings in the runners AND inside their guard assertions
(`findstr /C:"F:/rwkv_lmdb/test_db_5k_fix"`). Editing those is the exact "clone a runner,
update the lever but not everything that depends on it" failure CLAUDE.md keeps recording.
So the bytes move to C: and the F: path becomes a junction: every runner, toml and guard
keeps resolving, unchanged.

WHY env.copy(compact=True) AND NOT A FILE COPY. These are SPARSE. A plain copy materialises
the reservation, so test_db_5k_fix would land as its 232 GB apparent size instead of 103 GB
allocated -- which would not even fit. LMDB's own copy walks live pages only, so the result
is at most the live size and usually smaller.

ORDER IS CHOSEN SO THE DATA IS NEVER UNPROTECTED:
    copy -> verify (entry count + byte-identical sample) -> rename original to .old
         -> junction -> verify reads THROUGH the junction -> only then delete .old
At no point does the sole copy depend on something unverified.

Usage: .venv/Scripts/python.exe scratchpad/workload/move_lmdb.py <name> [--finalize]
       (without --finalize it copies and verifies, then stops and reports)
"""
import os
import subprocess
import sys
import time

import lmdb

SRC_ROOT = r"F:\rwkv_lmdb"
DST_ROOT = r"C:\rwkv_lmdb"
MAP_SIZE = 400_000_000_000
N_SAMPLE = 200


def free_gb(drive):
    import ctypes
    b = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(drive), None, None, ctypes.pointer(b))
    return b.value / 1e9


def entries(path):
    env = lmdb.open(path, map_size=MAP_SIZE, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        n = txn.stat()["entries"]
    env.close()
    return n


def sample_keys(path, n):
    """First n keys plus n from a cursor seeded mid-file -- cheap, and covers both ends
    rather than only the head, which a pure `first n` check would."""
    env = lmdb.open(path, map_size=MAP_SIZE, readonly=True, lock=False, subdir=True)
    out = []
    with env.begin() as txn:
        cur = txn.cursor()
        if cur.first():
            for i, k in enumerate(cur.iternext(keys=True, values=False)):
                out.append(k)
                if i >= n:
                    break
        if cur.last():
            for i, k in enumerate(cur.iterprev(keys=True, values=False)):
                out.append(k)
                if i >= n:
                    break
    env.close()
    return out


def compare(a, b, keys):
    ea = lmdb.open(a, map_size=MAP_SIZE, readonly=True, lock=False, subdir=True)
    eb = lmdb.open(b, map_size=MAP_SIZE, readonly=True, lock=False, subdir=True)
    bad = 0
    nbytes = 0
    with ea.begin() as ta, eb.begin() as tb:
        for k in keys:
            va, vb = ta.get(k), tb.get(k)
            if va != vb:
                bad += 1
            elif va:
                nbytes += len(va)
    ea.close()
    eb.close()
    return bad, nbytes


def main():
    name = sys.argv[1]
    finalize = "--finalize" in sys.argv
    src = os.path.join(SRC_ROOT, name)
    dst = os.path.join(DST_ROOT, name)
    old = src + ".old"

    assert os.path.isdir(src), "source missing: %s" % src
    os.makedirs(DST_ROOT, exist_ok=True)

    if not os.path.isdir(dst):
        n_src = entries(src)
        print("source %s: %d entries; C: free %.1f GB" % (name, n_src, free_gb("C:\\")),
              flush=True)
        os.makedirs(dst, exist_ok=True)
        t0 = time.perf_counter()
        env = lmdb.open(src, map_size=MAP_SIZE, readonly=True, lock=False, subdir=True)
        env.copy(dst, compact=True)
        env.close()
        dt = time.perf_counter() - t0
        print("copied in %.1f min" % (dt / 60), flush=True)
    else:
        print("destination already exists, verifying only", flush=True)
        n_src = entries(src)

    n_dst = entries(dst)
    size_dst = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    print("entries  src %d  dst %d  %s" % (n_src, n_dst, "MATCH" if n_src == n_dst else "*** MISMATCH"),
          flush=True)
    keys = sample_keys(src, N_SAMPLE)
    bad, nbytes = compare(src, dst, keys)
    print("sampled %d keys, %.1f MB, mismatches %d" % (len(keys), nbytes / 1e6, bad), flush=True)
    print("dst on-disk %.1f GB;  C: free %.1f GB" % (size_dst / 1e9, free_gb("C:\\")), flush=True)

    if n_src != n_dst or bad:
        print("VERIFY FAILED -- leaving both copies in place, nothing renamed or deleted")
        sys.exit(1)
    print("VERIFY OK", flush=True)

    if not finalize:
        print("(stopping here; rerun with --finalize to swap in the junction)")
        return

    os.rename(src, old)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", src, dst], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip(), flush=True)
    if not os.path.isdir(src):
        os.rename(old, src)
        print("JUNCTION FAILED -- original restored, nothing lost")
        sys.exit(1)
    n_via = entries(src)          # read through the junction
    print("through junction: %d entries  %s"
          % (n_via, "OK" if n_via == n_dst else "*** MISMATCH"), flush=True)
    if n_via != n_dst:
        os.remove(src)
        os.rename(old, src)
        print("JUNCTION READ FAILED -- original restored, nothing lost")
        sys.exit(1)
    import shutil
    shutil.rmtree(old, ignore_errors=True)
    print("done: %s now lives on C:, F: path is a junction. C: free %.1f GB, F: free %.1f GB"
          % (free_gb("C:\\"), free_gb("F:\\")), flush=True)


if __name__ == "__main__":
    main()
