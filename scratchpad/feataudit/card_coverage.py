"""Is the 35% NaN-metadata rate a MERGE failure (recoverable) or missing source rows (not)?"""
import pandas as pd, numpy as np, glob, os
base = r"C:\Users\Andrew\anki-revlogs-10k-id"
pub = r"C:\Users\Andrew\anki-revlogs-10k"
for u in (333, 315, 472, 101, 1):
    fr = glob.glob(os.path.join(base, "revlogs", f"user_id={u}", "*.parquet"))
    fc = glob.glob(os.path.join(base, "cards", f"user_id={u}", "*.parquet"))
    if not fr:
        print(f"user {u}: no revlogs"); continue
    r = pd.read_parquet(fr[0], columns=["card_id"])
    rc = set(r["card_id"].unique())
    if not fc:
        print(f"user {u}: rows {len(r):>8}  cards-table ABSENT -> 100% NaN metadata")
        continue
    c = pd.read_parquet(fc[0])
    cc = set(c["card_id"].unique())
    hit = r["card_id"].isin(cc)
    # same question on the PUBLISHED set, whose ids are factorized small ints
    fpr = glob.glob(os.path.join(pub, "revlogs", f"user_id={u}", "*.parquet"))
    fpc = glob.glob(os.path.join(pub, "cards", f"user_id={u}", "*.parquet"))
    pubtxt = ""
    if fpr and fpc:
        pr = pd.read_parquet(fpr[0], columns=["card_id"])
        pc = pd.read_parquet(fpc[0])
        ph = pr["card_id"].isin(set(pc["card_id"].unique()))
        pubtxt = (f"   | published: cards {len(pc):>7}  distinct-in-revlogs "
                  f"{pr['card_id'].nunique():>7}  rows-matched {100*ph.mean():6.2f}%")
    print(f"user {u:>5}: revlog rows {len(r):>8}  distinct cards {len(rc):>7}  "
          f"cards-table {len(cc):>7}  covered {100*len(rc & cc)/len(rc):6.2f}%  "
          f"rows-matched {100*hit.mean():6.2f}%{pubtxt}")
