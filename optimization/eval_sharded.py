"""Run a get_result eval split across N GPU-sharing processes, power-user-aware (wedge-safe).

Users are independent in get_result, so sharding changes NOTHING about per-user numerics -- each
user is processed exactly as in a single-process run (same model, same batches, same reduction
order within the user). Only wall-clock parallelism is added. Approved by Andrew 2026-07-03;
power-user solo phase approved by Andrew 2026-07-14.

WHY THE SOLO PHASE: two parallel shards each hitting a mega-user simultaneously oversubscribe
the 12 GB card (WDDM thrash, both shards frozen at ~11.7 GB -- the 2026-07-12/13 wedges). The
top of the user-size distribution is extreme (p50 57k, p99 ~1M, max ~2.1M); users >= the solo
threshold (default 1,000,000; ~56 users = ~11% of eval work on 5001-10000) therefore run FIRST,
alone, in ONE process with the GPU to themselves. The remaining users are LPT-split across the
parallel shards; the worst concurrent pair is then ~2x below the observed wedge scale.
Wall-clock ~= 0.11*W + 0.89*W/2 ~= 0.56*W (~1.8x over sequential, ~11% off unrestricted 2-way).

How: reads the base eval toml, sizes every user in USER_START..USER_END from the test LMDB's
"{user}_batches" keys, peels off users >= --solo-threshold into a solo phase (suffix "-solo"),
assigns the rest to shards greedily largest-first (LPT -> near-equal work), writes per-phase
tomls (suffixed output files + USERS_FILE lists, get_result's additive selector), runs the solo
process to completion, then the parallel shards, then merges all jsonls into the canonical
result files and prints by-user means.

Resume: rerun with the same args -- each phase's process skips users already in its own output
jsonl. VRAM: d=32 evals only; do NOT shard d=128 evals (one alone peaks ~9 GB).

Usage:
  python optimization/eval_sharded.py --config rwkv/get_result_config_iterN.toml [--shards 2]
      [--solo-threshold 1000000] [--fetch-per-shard 3] [--threads-per-shard 3]
      [--solo-threads 7] [--dry-run]
  --solo-threshold 0 disables the solo phase (old parallel-only behavior).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARD_DIR = os.environ.get("RWKV_EVAL_SHARD_DIR") or os.path.join(ROOT, "scratchpad", "eval_shards")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")


def read_toml_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def toml_value(text, key):
    for line in text.splitlines():
        s = line.split("#")[0].strip()
        if s.startswith(key) and s[len(key):].lstrip().startswith("="):
            v = s.split("=", 1)[1].strip()
            return v.strip('"')
    raise KeyError(f"{key} not found in toml")


def user_sizes(db_path, db_size, users):
    """Approximate per-user work from the stored section metadata (monotone with total length)."""
    import lmdb  # local import: keep the driver importable without the training env

    sizes = {}
    env = lmdb.open(db_path, map_size=db_size, readonly=True, lock=False,
                    max_readers=1024, readahead=False)
    with env.begin() as txn:
        for uid in users:
            raw = txn.get(f"{uid}_batches".encode())
            if raw is None:
                continue  # not in db -> get_result would skip it too
            batches = json.loads(raw)
            sizes[uid] = max(int(b[2]) for b in batches)
    env.close()
    return sizes


def lpt_split(sizes, n_shards):
    shards = [{"users": [], "work": 0} for _ in range(n_shards)]
    for uid, sz in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        tgt = min(shards, key=lambda s: s["work"])
        tgt["users"].append(uid)
        tgt["work"] += sz
    return shards


def merge_jsonl(shard_paths, out_path):
    records = []
    for p in shard_paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    users = [r["user"] for r in records]
    assert len(users) == len(set(users)), f"duplicate users across shards for {out_path}"
    records.sort(key=lambda r: r["user"])
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return records


def write_phase_toml(base, tag, file_ahead, file_imm, users, n_fetch):
    """Write the users file + toml for one phase process; returns the toml path."""
    users_file = os.path.join(SHARD_DIR, f"users_{tag}.json").replace("\\", "/")
    with open(users_file, "w", encoding="utf-8") as fh:
        json.dump(sorted(users), fh)
    text = []
    for line in base.splitlines():
        key = line.split("#")[0].split("=")[0].strip()
        if key == "FILE_AHEAD":
            line = f'FILE_AHEAD = "{file_ahead}-{tag}"'
        elif key == "FILE_IMM":
            line = f'FILE_IMM = "{file_imm}-{tag}"'
        elif key == "NUM_FETCH_PROCESSES":
            line = f"NUM_FETCH_PROCESSES = {n_fetch}"
        text.append(line)
    text.append(f'USERS_FILE = "{users_file}"')
    toml_path = os.path.join(SHARD_DIR, f"shard_{tag}.toml")
    with open(toml_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")
    return toml_path


def launch(toml_path, tag, n_threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
    logf = open(os.path.join(SHARD_DIR, f"shard_{tag}.log"), "w", encoding="utf-8")
    p = subprocess.Popen([PY, "-u", "-m", "rwkv.get_result", "--config", toml_path],
                         stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    print(f"shard {tag} launched pid {p.pid} (log {logf.name})", flush=True)
    return p, logf


def wait_all(procs, what):
    t0 = time.time()
    fails = 0
    for tag, (p, logf) in procs:
        rc = p.wait()
        logf.close()
        print(f"shard {tag} exit {rc} after {time.time() - t0:.0f}s", flush=True)
        fails += rc != 0
    if fails:
        sys.exit(f"{fails} {what} process(es) failed -- fix and rerun "
                 "(completed users are skipped on resume)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base get_result toml")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--solo-threshold", type=int, default=1_000_000,
                    help="users with work >= this run alone first (0 = disable solo phase)")
    ap.add_argument("--fetch-per-shard", type=int, default=3)
    ap.add_argument("--threads-per-shard", type=int, default=3)
    ap.add_argument("--solo-fetch", type=int, default=4)
    ap.add_argument("--solo-threads", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="plan the split, launch nothing")
    args = ap.parse_args()

    os.chdir(ROOT)
    base = read_toml_text(args.config)
    file_ahead = toml_value(base, "FILE_AHEAD")
    file_imm = toml_value(base, "FILE_IMM")
    db_path = toml_value(base, "DATASET_LMDB_PATH")
    db_size = int(toml_value(base, "DATASET_LMDB_SIZE").replace("_", ""))
    u0, u1 = int(toml_value(base, "USER_START")), int(toml_value(base, "USER_END"))
    if toml_value(base, "RAW").lower() != "false":
        sys.exit("RAW=true evals are not supported by the shard merger")

    for mode_file in (file_ahead, file_imm):
        canon = os.path.join("result", f"{mode_file}.jsonl")
        if os.path.exists(canon):
            sys.exit(f"{canon} already exists -- move it away first (merge refuses to clobber)")

    users = list(range(u0, u1 + 1))
    print(f"sizing {len(users)} users from {db_path} ...")
    sizes = user_sizes(db_path, db_size, users)
    print(f"{len(sizes)} users present in db")

    solo = {u: s for u, s in sizes.items() if args.solo_threshold and s >= args.solo_threshold}
    rest = {u: s for u, s in sizes.items() if u not in solo}
    if solo:
        print(f"solo phase: {len(solo)} users >= {args.solo_threshold:,}, "
              f"work {sum(solo.values()):,} ({100 * sum(solo.values()) / sum(sizes.values()):.1f}%)")
    shards = lpt_split(rest, args.shards)
    for i, sh in enumerate(shards):
        print(f"shard {i}: {len(sh['users'])} users, work {sh['work']:,}")

    os.makedirs(SHARD_DIR, exist_ok=True)
    tags = []
    solo_toml = None
    if solo:
        solo_toml = write_phase_toml(base, "solo", file_ahead, file_imm,
                                     list(solo), args.solo_fetch)
        tags.append("solo")
    shard_tomls = []
    for i, sh in enumerate(shards):
        shard_tomls.append(write_phase_toml(base, f"s{i}", file_ahead, file_imm,
                                            sh["users"], args.fetch_per_shard))
        tags.append(f"s{i}")

    if args.dry_run:
        print("dry run -- nothing launched")
        return

    if solo_toml:
        print("=== phase A: solo (power users, one process) ===", flush=True)
        wait_all([("solo", launch(solo_toml, "solo", args.solo_threads))], "solo")
    print("=== phase B: parallel shards ===", flush=True)
    wait_all([(f"s{i}", launch(t, f"s{i}", args.threads_per_shard))
              for i, t in enumerate(shard_tomls)], "shard")

    merged_sets = {}
    for mode_file in (file_ahead, file_imm):
        paths = [os.path.join("result", f"{mode_file}-{t}.jsonl") for t in tags]
        records = merge_jsonl(paths, os.path.join("result", f"{mode_file}.jsonl"))
        mean_ll = statistics.mean(r["metrics"]["LogLoss"] for r in records)
        merged_sets[mode_file] = {r["user"] for r in records}
        print(f"MERGED {mode_file}: {len(records)} users, by-user mean LogLoss {mean_ll:.6f}")

    # Completeness gate (added 2026-07-15 after the A0 eval crashed at user 6701 mid-shard
    # yet exited 0 -> a silent 1700/5000 "success"): every rostered user must be either
    # merged or explicitly NaN-skipped (get_result's get_loss NaN guard writes
    # <FILE_AHEAD>.nanskip.jsonl). Anything else = incomplete eval = nonzero exit.
    nanskips = {}
    for t in tags:
        p = os.path.join("result", f"{file_ahead}-{t}.nanskip.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        nanskips[r["user"]] = r
    if nanskips:
        out = os.path.join("result", f"{file_ahead}.nanskip.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for u in sorted(nanskips):
                fh.write(json.dumps(nanskips[u]) + "\n")
        print(f"NANSKIP: {len(nanskips)} users skipped by the NaN guard -> {out}: {sorted(nanskips)}")
    if merged_sets[file_ahead] != merged_sets[file_imm]:
        print(f"MODE MISMATCH: ahead/imm merged user sets differ "
              f"(sym-diff {sorted(merged_sets[file_ahead] ^ merged_sets[file_imm])[:20]})")
        sys.exit(3)
    missing = set(sizes) - merged_sets[file_ahead] - set(nanskips)
    if missing:
        print(f"INCOMPLETE: {len(missing)} users neither merged nor NaN-skipped: "
              f"{sorted(missing)[:20]}{' ...' if len(missing) > 20 else ''}")
        sys.exit(3)
    print(f"COMPLETE: {len(merged_sets[file_ahead])} merged + {len(nanskips)} nan-skipped "
          f"= {len(sizes)} rostered")


if __name__ == "__main__":
    main()
