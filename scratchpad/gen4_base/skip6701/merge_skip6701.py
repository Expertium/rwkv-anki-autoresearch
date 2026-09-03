"""Merge the -s0 shard files into the canonical gen4base result files (2,499 users: 6701 excluded).
Uses eval_sharded's own merge_jsonl so the format and the duplicate assert are the same."""
import json, sys
sys.path.insert(0, "optimization")
from eval_sharded import merge_jsonl
for a, b in (("result/RWKV-gen4base-s0.jsonl", "result/RWKV-gen4base.jsonl"), ("result/RWKV-P-gen4base-s0.jsonl", "result/RWKV-P-gen4base.jsonl")):
    merge_jsonl([a], b)
    n = sum(1 for l in open(b) if l.strip())
    print(b, n, "users")
    assert n == 2499, n
    assert not any(json.loads(l)["user"] == 6701 for l in open(b) if l.strip())
print("MERGE_OK")
