"""Run a get_result eval split across N GPU-sharing processes (default 2) for ~1.5-2x wall-clock.

Users are independent in get_result, so sharding changes NOTHING about per-user numerics -- each
user is processed exactly as in a single-process run (same model, same batches, same reduction
order within the user). Only wall-clock parallelism is added. Approved by Andrew 2026-07-03.

How: reads the base eval toml, sizes every user in USER_START..USER_END from the test LMDB's
"{user}_batches" keys, assigns users to shards greedily largest-first (LPT -> near-equal work),
writes per-shard tomls (suffixed output files + USERS_FILE lists, get_result's additive selector),
launches the shards as parallel `python -m rwkv.get_result` processes (QAT/arch env inherited),
then merges the shard jsonls into the canonical result files and prints by-user means.

Resume: rerun with the same args -- each shard skips users already in its own output jsonl.
VRAM: two d=32 champion evals fit easily; do NOT shard d=128 evals (one alone peaks ~9 GB).

Usage:
  python optimization/eval_sharded.py --config rwkv/get_result_config_iterN.toml [--shards 2]
                                      [--fetch-per-shard 3] [--threads-per-shard 3] [--dry-run]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARD_DIR = os.path.join(ROOT, "scratchpad", "eval_shards")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base get_result toml")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--fetch-per-shard", type=int, default=3)
    ap.add_argument("--threads-per-shard", type=int, default=3)
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
    shards = lpt_split(sizes, args.shards)
    for i, sh in enumerate(shards):
        print(f"shard {i}: {len(sh['users'])} users, work {sh['work']:,}")

    os.makedirs(SHARD_DIR, exist_ok=True)
    procs, shard_files = [], []
    for i, sh in enumerate(shards):
        users_file = os.path.join(SHARD_DIR, f"users_s{i}.json").replace("\\", "/")
        with open(users_file, "w", encoding="utf-8") as fh:
            json.dump(sorted(sh["users"]), fh)
        text = []
        for line in base.splitlines():
            key = line.split("#")[0].split("=")[0].strip()
            if key == "FILE_AHEAD":
                line = f'FILE_AHEAD = "{file_ahead}-s{i}"'
            elif key == "FILE_IMM":
                line = f'FILE_IMM = "{file_imm}-s{i}"'
            elif key == "NUM_FETCH_PROCESSES":
                line = f"NUM_FETCH_PROCESSES = {args.fetch_per_shard}"
            text.append(line)
        text.append(f'USERS_FILE = "{users_file}"')
        shard_toml = os.path.join(SHARD_DIR, f"shard_{i}.toml")
        with open(shard_toml, "w", encoding="utf-8") as fh:
            fh.write("\n".join(text) + "\n")
        shard_files.append((f"{file_ahead}-s{i}", f"{file_imm}-s{i}"))
        if args.dry_run:
            continue
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads_per_shard))
        logf = open(os.path.join(SHARD_DIR, f"shard_{i}.log"), "w", encoding="utf-8")
        p = subprocess.Popen([PY, "-u", "-m", "rwkv.get_result", "--config", shard_toml],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        procs.append((p, logf))
        print(f"shard {i} launched pid {p.pid} (log {logf.name})")

    if args.dry_run:
        print("dry run -- nothing launched")
        return

    t0 = time.time()
    fails = 0
    for i, (p, logf) in enumerate(procs):
        rc = p.wait()
        logf.close()
        print(f"shard {i} exit {rc} after {time.time() - t0:.0f}s")
        fails += rc != 0
    if fails:
        sys.exit(f"{fails} shard(s) failed -- fix and rerun (completed users are skipped on resume)")

    for mode_file, idx in ((file_ahead, 0), (file_imm, 1)):
        paths = [os.path.join("result", f"{sf[idx]}.jsonl") for sf in shard_files]
        records = merge_jsonl(paths, os.path.join("result", f"{mode_file}.jsonl"))
        mean_ll = statistics.mean(r["metrics"]["LogLoss"] for r in records)
        print(f"MERGED {mode_file}: {len(records)} users, by-user mean LogLoss {mean_ll:.6f}")


if __name__ == "__main__":
    main()
