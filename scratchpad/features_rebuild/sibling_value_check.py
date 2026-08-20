"""Are the two new columns actually ALIVE on real users? Finite is not the same as informative --
a boolean that is always 0 is a dead dim and it is better to learn that now than after a 7.75 h arm."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
os.environ["RWKV_ID_FEATURES"] = "1"
import rwkv.id_features as idf

D = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
for uid in (1, 2, 101):
    r = pd.read_parquet(D / "revlogs" / f"user_id={uid}")
    c = pd.read_parquet(D / "cards" / f"user_id={uid}")
    d = pd.read_parquet(D / "decks" / f"user_id={uid}")
    df = r.merge(c.drop(columns=["user_id"], errors="ignore"), on="card_id", how="left")
    df = df.merge(d.drop(columns=["user_id", "parent_id"], errors="ignore"), on="deck_id", how="left")
    df = df.sort_values("review_time", kind="stable").reset_index(drop=True)
    raw = idf.sibling_gap_seconds(df)
    idf.add_id_features(df, c, d)
    g = df["scaled_sibling_gap"].to_numpy()
    b = df["card_predates_first_review"].to_numpy()
    dfn = raw >= 0
    print("user %-4d n=%-7d | gap defined %6.2f%%  scaled[min,max]=[%.2f,%.2f]  raw days p50=%.2f p90=%.2f"
          % (uid, len(df), 100 * dfn.mean(), g.min(), g.max(),
             np.median(raw[dfn]) / 86400 if dfn.any() else -1,
             np.percentile(raw[dfn], 90) / 86400 if dfn.any() else -1))
    print("            card_predates_first_review: %.2f%% ones (%d rows)" % (100 * b.mean(), int(b.sum())))
