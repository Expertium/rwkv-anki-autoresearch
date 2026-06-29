"""Export the inputs + reference outputs needed to verify the Rust RWKV port.

Strategy (see step3-rust-plan memory): the Rust port reimplements ONLY the neural
network + curve math, NOT the heavy stateful feature engineering. So we run the
Python RNN reference here and export, per review:
  - the exact 92-dim feature vectors fed to the net (imm forward + proc forward),
  - the dense state-routing ids (card/note/deck/preset),
  - the raw elapsed_seconds used for the ahead prediction,
  - the review_th + rating label,
plus the model weights (-> safetensors) and Python's own per-review predictions and
mean LogLoss (the parity target, must reproduce reference/ref_metrics.json).

Rust then consumes identical inputs and must match within +/-0.0005 mean LogLoss.

Run: python export_rnn_trace.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import save_file as save_np
from safetensors.torch import save_file as save_pt

torch.set_num_threads(7)

import rwkv.run_as_rnn as rnn_mod
from rwkv.get_result import get_benchmark_info, get_stats
from rwkv.model.srs_model_rnn import SrsRWKVRnn
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

DATA = Path("../anki-revlogs-10k")
LABEL_DB = "label_filter_db"
LABEL_DB_SIZE = 2_000_000_000
# Parity target = the current champion. Override with RWKV_CHAMP_CKPT / RWKV_CHAMP_SFT.
# architecture.py must match this checkpoint (it is the single arch source).
MODEL_PATH = os.environ.get("RWKV_CHAMP_CKPT", "pretrain/rwkv/ref_100/rwkv_ref_558.pth")
WEIGHTS_SFT = os.environ.get("RWKV_CHAMP_SFT", "rwkv_ref_558.safetensors")
REF_USERS = [107, 136, 156]
OUT_DIR = Path("reference")


class CapturingRNN(rnn_mod.RNNProcess):
    """RNNProcess that records the routing ids used for state lookup on each .run()."""

    def run(self, row, skip):
        # Exactly the ids the parent uses to index card/note/deck/preset states.
        self._route = (
            row["card_id"],
            row["note_id"],
            row["deck_id"],
            row["preset_id"],
        )
        return super().run(row, skip)


def load_user_df(user_id):
    """Replicates the dataframe construction in rwkv.run_as_rnn.run()."""
    df = pd.read_parquet(DATA / "revlogs" / f"{user_id=}")
    df["review_th"] = range(1, df.shape[0] + 1)
    df_cards = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(DATA / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)
    return df


def export_user(user_id):
    torch.manual_seed(user_id)  # reproducible id-encodings, matches make_reference.py

    df = load_user_df(user_id)
    equalize_review_ths, rmse_bins = get_benchmark_info(LABEL_DB, LABEL_DB_SIZE, user_id)
    rmse_bins_dict = {
        equalize_review_ths[i]: rmse_bins[i] for i in range(len(equalize_review_ths))
    }

    srs = CapturingRNN(
        path=MODEL_PATH, device=torch.device("cpu"), dtype=torch.float32
    )
    # Hook review() to capture the exact 92-dim feature vector it receives.
    _orig_review = srs.rnn.review

    def _cap_review(card_features, *a, **k):
        srs._last_features = (
            card_features.detach().cpu().numpy().reshape(-1).astype(np.float32).copy()
        )
        return _orig_review(card_features, *a, **k)

    srs.rnn.review = _cap_review

    # Dense per-user id maps (encounter order) for compact state routing in Rust.
    maps = {"card": {}, "note": {}, "deck": {}, "preset": {}}

    def densify(kind, raw):
        m = maps[kind]
        if raw not in m:
            m[raw] = len(m)
        return m[raw]

    feats_imm, feats_proc, route = [], [], []
    elapsed_list, review_th_list, rating_list, has_ahead_list = [], [], [], []

    # Tensor dicts feed get_stats (it calls .tolist()); float dicts feed JSON.
    pred_imm_t = {}
    pred_ahead_t = {}
    pred_ahead_curve = {}
    label_rating = {}

    for i, row in df.iterrows():
        i = int(i)
        if (i + 1) % 500 == 0:
            print(f"  user {user_id}: {i + 1}/{len(df)}", flush=True)
        card_id = row["card_id"]
        review_th = int(row["review_th"])

        has_ahead = card_id in pred_ahead_curve
        if has_ahead:
            pred_ahead_t[review_th] = srs.predict_func(
                pred_ahead_curve[card_id], row["elapsed_seconds"]
            )

        # Immediate prediction (no rating/duration sent), state read-only.
        imm_info = row.copy()
        imm_info.drop(columns=["rating", "duration"], inplace=True)
        pred_imm_t[review_th] = srs.imm_predict(imm_info)
        fi = srs._last_features.copy()
        route_raw = srs._route  # routing ids are identical for imm and proc

        # Ahead-of-time forward, updates state, returns the stored forgetting curve.
        pred_ahead_curve[card_id] = srs.process_row(row)
        fp = srs._last_features.copy()

        label_rating[review_th] = int(row["rating"]) - 1

        feats_imm.append(fi)
        feats_proc.append(fp)
        route.append(
            [
                densify("card", route_raw[0]),
                densify("note", route_raw[1]),
                densify("deck", route_raw[2]),
                densify("preset", route_raw[3]),
            ]
        )
        elapsed_list.append(float(row["elapsed_seconds"]))
        review_th_list.append(review_th)
        rating_list.append(int(row["rating"]))
        has_ahead_list.append(1 if has_ahead else 0)

    # Python reference metrics (must reproduce reference/ref_metrics.json).
    imm_stats, _ = get_stats(
        user_id, equalize_review_ths, rmse_bins_dict, pred_imm_t, label_rating
    )
    ahead_stats, _ = get_stats(
        user_id, equalize_review_ths, rmse_bins_dict, pred_ahead_t, label_rating
    )
    pred_imm = {k: float(v.item()) for k, v in pred_imm_t.items()}
    pred_ahead = {k: float(v.item()) for k, v in pred_ahead_t.items()}

    tensors = {
        "feats_imm": np.stack(feats_imm).astype(np.float32),
        "feats_proc": np.stack(feats_proc).astype(np.float32),
        "route": np.asarray(route, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_list, dtype=np.float32),
        "review_th": np.asarray(review_th_list, dtype=np.int64),
        "rating": np.asarray(rating_list, dtype=np.int64),
        "has_ahead": np.asarray(has_ahead_list, dtype=np.int64),
    }
    save_np(tensors, str(OUT_DIR / f"trace_user_{user_id}.safetensors"))

    meta = {
        "user": user_id,
        "n_reviews": len(df),
        "equalize_review_ths": [int(x) for x in equalize_review_ths],
        "rmse_bins": [int(x) for x in rmse_bins],
        "label_rating": {int(k): int(v) for k, v in label_rating.items()},
        "py_pred_imm": {int(k): float(v) for k, v in pred_imm.items()},
        "py_pred_ahead": {int(k): float(v) for k, v in pred_ahead.items()},
        "py_imm_logloss": imm_stats["metrics"]["LogLoss"],
        "py_ahead_logloss": ahead_stats["metrics"]["LogLoss"],
        "size": imm_stats["size"],
    }
    with open(OUT_DIR / f"trace_user_{user_id}.json", "w") as f:
        json.dump(meta, f)

    print(
        f"user {user_id}: size {meta['size']}, "
        f"imm {meta['py_imm_logloss']:.6f}, ahead {meta['py_ahead_logloss']:.6f}",
        flush=True,
    )
    return meta


def export_weights():
    model = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    flat = {k: v.detach().cpu().contiguous().float() for k, v in model.state_dict().items()}
    save_pt(flat, str(OUT_DIR / WEIGHTS_SFT))
    names = sorted(flat.keys())
    with open(OUT_DIR / "weight_names.json", "w") as f:
        json.dump({"n": len(names), "names": names,
                   "shapes": {k: list(flat[k].shape) for k in names}}, f, indent=1)
    print(f"weights: {len(names)} tensors -> reference/{WEIGHTS_SFT}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    export_weights()
    metas = [export_user(u) for u in REF_USERS]

    mean_imm = sum(m["py_imm_logloss"] for m in metas) / len(metas)
    mean_ahead = sum(m["py_ahead_logloss"] for m in metas) / len(metas)
    print(f"\nMEAN imm={mean_imm:.6f}  ahead={mean_ahead:.6f}")

    # This export is now the canonical, EVAL-MODE reference (dropout off). It replaces
    # the earlier dropout-on numbers. Show the shift for transparency.
    old = json.load(open("reference/ref_metrics.json")) if (
        OUT_DIR / "ref_metrics.json"
    ).exists() else None
    ref = {
        "model": "rwkv_ref_558.pth",
        "mode": "RNN eval (dropout off), float32",
        "seed_scheme": "torch.manual_seed(user_id) per user (id-encodings only)",
        "users": REF_USERS,
        "per_user": [
            {"user": m["user"], "size": m["size"],
             "imm_logloss": m["py_imm_logloss"], "ahead_logloss": m["py_ahead_logloss"]}
            for m in metas
        ],
        "mean_imm_logloss": round(mean_imm, 6),
        "mean_ahead_logloss": round(mean_ahead, 6),
        "tolerance": 0.0005,
        "note": "Rust RNN port must match mean_*_logloss within +/-tolerance.",
    }
    with open("reference/ref_metrics.json", "w") as f:
        json.dump(ref, f, indent=2)
    if old is not None:
        print(
            f"prev ref_metrics (dropout-on): imm={old['mean_imm_logloss']:.6f} "
            f"ahead={old['mean_ahead_logloss']:.6f}  -> now eval-mode (dropout off)"
        )
    print("wrote eval-mode reference/ref_metrics.json")


if __name__ == "__main__":
    main()
