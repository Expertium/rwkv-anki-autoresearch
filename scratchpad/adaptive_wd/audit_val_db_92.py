import json
import lmdb
from io import BytesIO
import torch

path = r"F:/rwkv_lmdb/test_db_5k_adaptive92"
env = lmdb.open(path, map_size=250_000_000_000, readonly=True, lock=False,
                readahead=False, max_readers=512)
missing, multi, bad = [], [], []
widths = set()
with env.begin() as txn:
    for user in range(5001, 7501):
        raw = txn.get(f"{user}_batches".encode())
        if raw is None:
            missing.append(user)
            continue
        batches = json.loads(raw)
        if len(batches) != 1:
            multi.append((user, len(batches)))
        for start, end, length in batches:
            key = f"{user}_{start}-{end}_{length}_card_features"
            value = txn.get(key.encode())
            if value is None:
                bad.append((user, "missing tensor"))
                continue
            tensor = torch.load(BytesIO(value), weights_only=True, map_location="cpu")
            widths.add(tensor.shape[-1])
            if tensor.shape[-1] != 24:
                bad.append((user, tensor.shape[-1]))
print({"users_present": 2500-len(missing), "missing": len(missing),
       "multi_batch_users": len(multi), "sample_multi": multi[:20],
       "card_feature_widths": sorted(widths), "bad_tensors": len(bad),
       "sample_bad": bad[:10]})
env.close()
if missing or bad or widths != {24}:
    raise SystemExit("VAL_92_AUDIT_FAILED")
