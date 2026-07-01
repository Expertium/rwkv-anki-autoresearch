"""Profile the DATA-DELIVERY pipeline (NOT the GPU step) to locate the ~499 ms/step stall.

profile_train.py cached batches ON-DEVICE, so its 588 ms/step ceiling EXCLUDES prep + IPC + input H2D.
Real sc8k is ~1087 ms/step => ~499 ms/step is data delivery. This script breaks that down, single-process
(no multiprocessing IPC), so it isolates the CPU costs from the transfer cost:

  (1) get_data  -- LMDB read + tensor deserialize per group
  (2) prepare   -- the CPU batch-assembly (suspected O(B*T) python loop hotspot)
  (3) pin       -- pin_memory() of the prepared CPU batch (cost of enabling async H2D)
  (4) to(cuda)  -- pageable (current, synchronous) input H2D
  (5) to(cuda) non_blocking from PINNED -- async-capable input H2D (what the prefetch will use)
  (6) prepare() inner-loop micro-breakdown (time_shift_select python double loop vs the rest)

Conclusion logic: if prepare dominates -> vectorize prepare and/or fix IPC (mp.Queue); if to(cuda)
dominates -> async pinned double-buffer is the win. (IPC itself needs the real multiprocessing loop;
measured separately by instrumenting train_rwkv. This isolates the per-batch CPU+transfer costs.)

Run (clean machine, after the build frees CPU):
  .venv\\Scripts\\python.exe scratchpad/profile_fetch.py --config scratchpad/train_1500_ws.toml --db train_db_sc8k
"""
import argparse
import time
from argparse import Namespace

import tomli
import torch
import lmdb

from rwkv.prepare_batch import prepare, get_data
from rwkv.train_rwkv import get_groups


def load_config(path):
    with open(path, "rb") as f:
        a = tomli.load(f)
    a["DEVICE"] = torch.device(a.get("DEVICE", "cuda"))
    return Namespace(**a)


def batch_nbytes(pb):
    tot = 0
    for t in [pb.start, pb.labels, pb.label_review_th]:
        tot += t.element_size() * t.nelement()
    for grp in (pb.sub_gather, pb.time_shift_selects, pb.skips):
        for sub in grp:
            for x in sub:
                tot += x.element_size() * x.nelement()
    return tot


def pin_batch(pb):
    from rwkv.model.srs_model import PreparedBatch
    return PreparedBatch(
        num_data=pb.num_data,
        start=pb.start.pin_memory(),
        sub_gather=[[x.pin_memory() for x in sub] for sub in pb.sub_gather],
        sub_gather_lens=pb.sub_gather_lens,
        time_shift_selects=[[x.pin_memory() for x in sub] for sub in pb.time_shift_selects],
        skips=[[x.pin_memory() for x in sub] for sub in pb.skips],
        labels=pb.labels.pin_memory(),
        label_review_th=pb.label_review_th.pin_memory(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--db", default=None, help="override TRAIN_DATASET_LMDB_PATH")
    ap.add_argument("--ngroups", type=int, default=12)
    args = ap.parse_args()

    config = load_config(args.config)
    db = args.db or config.TRAIN_DATASET_LMDB_PATH
    size = getattr(config, "TRAIN_DATASET_LMDB_SIZE", 80_000_000_000)
    MAX = config.MAX_TRAIN_GLOBAL_LEN
    device = torch.device("cuda")
    print(f"db={db} MAX_TRAIN_GLOBAL_LEN={MAX}", flush=True)

    users = list(range(config.TRAIN_USERS_START, config.TRAIN_USERS_END + 1))
    # only the 100-user gate db has these; for sc8k_1500 use 1-100 worth of groups regardless
    groups = get_groups(db, size, MAX, users=users)
    ng = min(args.ngroups, len(groups))
    print(f"profiling {ng} groups (B = {[len(groups[i]) for i in range(ng)]})", flush=True)

    env = lmdb.open(db, map_size=size)
    t_get = t_prep = t_pin = t_h2d_page = t_h2d_pin = 0.0
    nbytes = 0
    with env.begin(write=False) as txn:
        for i in range(ng):
            t = time.perf_counter()
            samples = [get_data(txn, key, device="cpu") for key in groups[i]]
            t_get += time.perf_counter() - t

            t = time.perf_counter()
            pb = prepare(samples, target_len=MAX, seed=1234)
            t_prep += time.perf_counter() - t
            nbytes += batch_nbytes(pb)

            t = time.perf_counter()
            pbp = pin_batch(pb)
            t_pin += time.perf_counter() - t

            torch.cuda.synchronize(); t = time.perf_counter()
            _ = pb.to(device)                       # pageable (current)
            torch.cuda.synchronize(); t_h2d_page += time.perf_counter() - t

            torch.cuda.synchronize(); t = time.perf_counter()
            _ = pbp.to(device)                      # from pinned (sync here, but async-capable)
            torch.cuda.synchronize(); t_h2d_pin += time.perf_counter() - t
    env.close()

    print("\n--- data-delivery ms/group (single process, no IPC) ---", flush=True)
    print(f"  get_data (LMDB read)        {1000*t_get/ng:8.2f} ms")
    print(f"  prepare  (CPU assembly)     {1000*t_prep/ng:8.2f} ms   <-- suspected hotspot")
    print(f"  pin_memory()                {1000*t_pin/ng:8.2f} ms")
    print(f"  to(cuda) pageable (current) {1000*t_h2d_page/ng:8.2f} ms")
    print(f"  to(cuda) from pinned        {1000*t_h2d_pin/ng:8.2f} ms")
    print(f"  avg batch size             {nbytes/ng/1e6:8.2f} MB", flush=True)
    print(f"\n  CPU-side total (get+prepare) {1000*(t_get+t_prep)/ng:8.2f} ms/group", flush=True)
    print(f"  with 7 fetch workers -> ~{1000*(t_get+t_prep)/ng/7:8.2f} ms/step prep throughput", flush=True)
    print(f"  input H2D on critical path   {1000*t_h2d_page/ng:8.2f} ms (pageable) / "
          f"{1000*t_h2d_pin/ng:.2f} ms (pinned)", flush=True)


if __name__ == "__main__":
    main()
