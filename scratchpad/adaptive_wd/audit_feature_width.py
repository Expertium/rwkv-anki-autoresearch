import json
import os
from io import BytesIO

os.environ["RWKV_ID_FEATURES"] = "0"
os.environ["RWKV_REAL_CYCLES"] = "0"

import lmdb
import torch

DATABASES = [
    (r"F:/rwkv_lmdb/train_db_5k_h1_adaptive92", 1, 5000, 400_000_000_000),
    (r"F:/rwkv_lmdb/test_db_5k_adaptive92", 5001, 7500, 250_000_000_000),
]

for path, first, last, map_size in DATABASES:
    env = lmdb.open(path, map_size=map_size, readonly=True, lock=False,
                    readahead=False, max_readers=512)
    with env.begin() as txn:
        # DB completeness and batch counts have a separate audit; keep this width probe
        # bounded to avoid a full database scan before the required adaptive smoke.
        present = [u for u in range(first, min(last, first + 4) + 1)
                   if txn.get(f"{u}_batches".encode()) is not None]
        samples = []
        for user in present:
            batches = json.loads(txn.get(f"{user}_batches".encode()))
            if not batches:
                continue
            start, end, length = batches[0]
            key = f"{user}_{start}-{end}_{length}_card_features"
            value = txn.get(key.encode())
            if value is None:
                raise RuntimeError(f"missing feature tensor: {key}")
            tensor = torch.load(BytesIO(value), weights_only=True, map_location="cpu")
            samples.append((user, tuple(tensor.shape), str(tensor.dtype)))
    env.close()
    widths = sorted({shape[-1] for _, shape, _ in samples})
    print({"database": path, "sampled_users": len(present), "card_feature_widths": widths, "samples": samples})
    if widths != [24]:
        raise RuntimeError(f"expected stored card-feature width 24 for the 92-input layout: {path}; saw {widths}")
