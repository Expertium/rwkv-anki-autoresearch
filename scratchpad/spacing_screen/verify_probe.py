"""Verify the monotonicity probe's instrument against a parity-verified reference.

The probe reported that predicted retention at a FIXED horizon FALLS over a card's life, which is
the opposite of the naive SRS expectation. Before reading anything into that, check the instrument:
drive the iter-41 checkpoint exactly as export_rnn_trace.py did and require that
predict_func(curve_after_review_n, actual_elapsed) reproduces the stored py_pred_ahead -- the same
numbers the Rust port was certified against at 0.000e+00.

If this matches, the probe is wired correctly and the surprising trend is a property of the model.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ARCH_MODULE", "scratchpad/track2_a18/architecture_d80_lora4_cnd.py")
for k, v in (("RWKV_INTERLEAVE", "1"), ("RWKV_GRU_HEAD", "3"), ("RWKV_PAVA_LAMBDA", "0.2"),
             ("RWKV_NO_AHEAD_RESIDUAL", "1"), ("RWKV_STRIP_L0_VLORA", "1"),
             ("RWKV_ZERO_FEATURES", "22"), ("RWKV_STATE_CLAMP_TAU", "300"),
             ("RWKV_STATE_CLAMP_WINDOW", "32768"),
             ("RWKV_STRIP_CMIX", "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                                 "preset_id:2,deck_id:1,deck_id:2,card_id:1")):
    os.environ.setdefault(k, v)

import numpy as np
import torch

torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod
from pathlib import Path
import pandas as pd

DATA = Path("../anki-revlogs-10k")


def load_user_df(user_id):
    df = pd.read_parquet(DATA / "revlogs" / f"{user_id=}")
    df["review_th"] = range(1, df.shape[0] + 1)
    dc = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", user_id)])
    dc.drop(columns=["user_id"], inplace=True)
    dd = pd.read_parquet(DATA / "decks", filters=[("user_id", "=", user_id)])
    dd.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(dc, on="card_id", how="left", validate="many_to_one")
    df = df.merge(dd, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)
    return df

CKPT = "scratchpad/iter41_ilv/i41_d_10935.pth"
UID = 107

ref = json.load(open(f"reference_iter41/trace_user_{UID}.json"))
ref_ahead = {int(k): v for k, v in ref["py_pred_ahead"].items()}

torch.manual_seed(UID)
df = load_user_df(UID)
srs = rnn_mod.RNNProcess(path=CKPT, device=torch.device("cpu"), dtype=torch.float32)

curves = {}
diffs = []
for _, row in df.iterrows():
    cid = row["card_id"]
    rth = int(row["review_th"])
    if cid in curves and rth in ref_ahead:
        mine = float(srs.predict_func(curves[cid], row["elapsed_seconds"]))
        diffs.append(abs(mine - ref_ahead[rth]))
    curves[cid] = srs.process_row(row)

d = np.array(diffs)
print(f"compared {d.size:,} ahead predictions against the certified trace")
print(f"  max |mine - reference|   {d.max():.3e}")
print(f"  mean                     {d.mean():.3e}")
print("  VERDICT:", "INSTRUMENT OK" if d.max() < 1e-5 else "INSTRUMENT WRONG -- do not use it")
