import pandas as pd, glob, os
base = r"C:\Users\Andrew\anki-revlogs-10k-id"
for kind in ("revlogs", "cards", "decks"):
    f = sorted(glob.glob(os.path.join(base, kind, "user_id=333", "*.parquet")))
    if not f:
        f = sorted(glob.glob(os.path.join(base, kind, "**", "*.parquet"), recursive=True))[:1]
    if not f:
        print(f"{kind}: no parquet found"); continue
    df = pd.read_parquet(f[0])
    print(f"\n{kind}  ({f[0].split(chr(92))[-2]})  rows={len(df)}")
    print("   columns:", list(df.columns))
    print(df.head(2).to_string(max_colwidth=22))
