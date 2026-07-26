"""Extract the first revlog entry's (id, cid) from every user file in FSRS-Anki-20k.

Hand-decodes just the protobuf header instead of parsing the whole file:
  RevlogEntries { repeated RevlogEntry entries = 1; }  -> tag 0x0A, varint len
  RevlogEntry   { int64 id = 1; int64 cid = 2; }       -> tag 0x08 varint, tag 0x10 varint
so we read 64 bytes per user instead of ~2.7 MB.
"""
import os, sys, csv
from concurrent.futures import ThreadPoolExecutor

ROOT = r"F:\FSRS\FSRS-Anki-20k\revlogs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_entries_20k.csv")


def _varint(b, i):
    val = 0
    shift = 0
    while True:
        c = b[i]
        i += 1
        val |= (c & 0x7F) << shift
        if not (c & 0x80):
            return val, i
        shift += 7


def head(path):
    with open(path, "rb") as f:
        b = f.read(64)
    if len(b) < 8 or b[0] != 0x0A:
        return None
    _, i = _varint(b, 1)          # length of the first RevlogEntry
    if b[i] != 0x08:
        return None
    rid, i = _varint(b, i + 1)    # field 1 = id
    if b[i] != 0x10:
        return rid, None
    cid, i = _varint(b, i + 1)    # field 2 = cid
    return rid, cid


def job(args):
    shard, name = args
    p = os.path.join(ROOT, shard, name)
    try:
        r = head(p)
    except Exception:
        r = None
    if r is None:
        return None
    return (shard, name[:-7], r[0], r[1], os.path.getsize(p))


def main():
    tasks = []
    for shard in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, shard)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".revlog"):
                tasks.append((shard, name))
    print(f"{len(tasks)} user files", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for n, r in enumerate(ex.map(job, tasks)):
            if r is not None:
                rows.append(r)
            if (n + 1) % 2000 == 0:
                print(f"  {n+1}/{len(tasks)}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shard", "user", "first_id", "first_cid", "bytes"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
