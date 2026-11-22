"""Export deploy traces (the Rust engine's input format) for the 50 longest-history users.

Why this exists
---------------
`rust/rwkv-infer/src/finite.rs` + `RWKV_FINITE_CHECK=1` is the DEPLOY twin of the Python corruption
probe in this folder: it watches the engine that actually ships, where a non-finite value is
otherwise invisible (`san()` maps a non-finite p(recall) to 0.5 and `compress_wkv_state` skips
degenerate components). But it has only ever run on the 3-user parity trace (users 107/136/156,
~5-10k reviews each), while Andrew's gate (2026-11-21) is about the users with the LONGEST average
card histories -- the sequences where a recurrent state has the most steps in which to blow up.
A deploy check that has never seen a long history does not answer the gate.

This driver reuses `export_rnn_trace.export_user()` UNCHANGED -- the same function the certified
parity traces come from -- so the trace layout, the per-user id-encoding seed (`torch.manual_seed`)
and the feature capture hook are the certified ones. Only four things are redirected: the output
directory, the roster, the label DB (the eval's `label_filter_db_id_e2s`, not the repo-root default)
and the checkpoint. The checkpoint choice is irrelevant to the trace CONTENT: `feats_imm` /
`feats_proc` are the feature-engineering output the RNN's `review()` receives, and `route` /
`elapsed_seconds` / `review_th` / `rating` come from the frame, not from the weights. Any real
checkpoint of this architecture produces byte-identical trace tensors.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/export_corrupt50_traces.py \
      [--users 7325] [--out scratchpad/quant_plan/trace_corrupt50] [--force]

Writes <out>/trace_user_<u>.safetensors + .json per user, and appends one line per finished user to
<out>/export_progress.jsonl (so a restart can skip what is done).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# The 92-input lineage's arch flags, exactly as quant_sweep.base_env() sets them for the eval. Set
# BEFORE importing the exporter: it reads its env at import time.
os.environ.setdefault("RWKV_ID_FEATURES", "0")
os.environ.setdefault("RWKV_REAL_CYCLES", "0")
os.environ.setdefault("RWKV_FINAL_LAYOUT", "0")
os.environ.setdefault("RWKV_E2S_PUBLISHED", "0")
os.environ["RWKV_NO_JIT"] = "1"
# The eval's label DB (quant_sweep.LABEL_DB). export_rnn_trace defaults to the repo-root
# "label_filter_db", which is a DIFFERENT, older DB -- get_benchmark_info would find no key for
# these users and hand get_stats an empty equalize set.
os.environ["RWKV_EXPORT_LABEL_DB"] = "F:/rwkv_lmdb/label_filter_db_id_e2s"

sys.path.insert(0, str(ROOT))

ROSTER = HERE / "corrupt50.json"
DEFAULT_OUT = ROOT / "scratchpad" / "quant_plan" / "trace_corrupt50"
# A real checkpoint of this architecture. See the docstring: it does not affect the trace tensors.
DEFAULT_CKPT = ROOT / "scratchpad" / "quant_plan" / "qat_it36l2i8s4" / "qat_124.pth"
LABEL_DB_SIZE = 40_000_000_000  # the exporter hardcodes 2 GB; this DB is larger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="", help="comma list; default = every user in corrupt50.json")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--force", action="store_true", help="re-export a user whose trace already exists")
    ap.add_argument("--shard", type=int, default=0, help="shard index (0-based) for parallel export")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3, help="torch intra-op threads per process")
    args = ap.parse_args()

    doc = json.loads(ROSTER.read_text(encoding="utf-8"))
    roster = [int(u) for u in doc["users"]]
    if args.users:
        want = [int(u) for u in args.users.split(",") if u.strip()]
        missing = [u for u in want if u not in roster]
        if missing:
            raise SystemExit(f"not in the corrupt50 roster: {missing}")
        roster = want

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    if args.nshards > 1:
        # Sharding is safe: export_user() seeds torch per USER (`torch.manual_seed(user_id)`), so a
        # user's trace does not depend on which process produced it, and each user writes its own
        # files. The per-review loop is overhead-bound (the profile of user 6326 is 173k module
        # forwards for 6.2k reviews), so one process per shard beats widening the thread pool.
        roster = [u for i, u in enumerate(roster) if i % args.nshards == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(roster)} users", flush=True)

    import export_rnn_trace as e  # noqa: E402

    e.MODEL_PATH = str(ckpt)
    e.OUT_DIR = out
    e.LABEL_DB = os.environ["RWKV_EXPORT_LABEL_DB"]
    e.LABEL_DB_SIZE = LABEL_DB_SIZE
    import torch
    torch.set_num_threads(max(1, args.threads))  # the exporter sets 7; 5 shards would oversubscribe

    # One progress file PER SHARD: several processes appending to one file is not atomic on Windows,
    # and a torn line would make the resume scan drop a finished user. Reads glob every shard.
    prog = out / f"export_progress_shard{args.shard}.jsonl"
    done = set()
    for p in sorted(out.glob("export_progress*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(int(rec["user"]))

    print(f"roster {len(roster)} users  out {out}\nckpt {ckpt}\nlabel_db {e.LABEL_DB}", flush=True)
    n_new = 0
    t_all = time.time()
    for i, u in enumerate(roster, 1):
        sf = out / f"trace_user_{u}.safetensors"
        if u in done and sf.exists() and not args.force:
            print(f"[{i}/{len(roster)}] user {u}: already exported, skip", flush=True)
            continue
        t0 = time.time()
        ok, err, n_reviews, size = True, "", 0, 0
        try:
            meta = e.export_user(u)
            n_reviews = int(meta["n_reviews"])
            size = int(meta["size"])
        except Exception as exc:  # one bad user must not lose the other 49
            ok, err = False, f"{type(exc).__name__}: {exc}"
            import traceback
            traceback.print_exc()
        dt = time.time() - t0
        rec = {"user": u, "ok": ok, "secs": round(dt, 1), "n_reviews": n_reviews,
               "size": size, "error": err, "trace": str(sf) if sf.exists() else ""}
        with open(prog, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        n_new += 1
        rate = f"{n_reviews / dt:.0f} rev/s" if dt > 0 and n_reviews else "-"
        print(f"[{i}/{len(roster)}] user {u}: ok={ok} {dt / 60:.1f} min {rate} "
              f"n={n_reviews} {err}", flush=True)
    print(f"\nexported {n_new} user(s) in {(time.time() - t_all) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
