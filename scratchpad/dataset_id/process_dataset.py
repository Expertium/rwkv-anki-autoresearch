import pandas as pd  # type: ignore

user_id = 42
filters = [("user_id", "=", user_id)]
df_revlogs = pd.read_parquet(f"../anki-revlogs-10k/revlogs", filters=filters)
# the revlogs have been sorted by timestamp
df_revlogs["review_th"] = range(1, df_revlogs.shape[0] + 1)
# user_id is not needed
del df_revlogs["user_id"]

df_cards = pd.read_parquet("../anki-revlogs-10k/cards", filters=filters)
del df_cards["user_id"]

df_decks = pd.read_parquet("../anki-revlogs-10k/decks", filters=filters)
del df_decks["user_id"]

# join the three tables, the order of the joins is important
# if cards were deleted by the user, the revlogs still exist
df_join = df_revlogs.merge(df_cards, on="card_id", how="left").merge(
    df_decks, on="deck_id", how="left"
)

print(df_join.head())
print(df_join.describe())
df_join.to_csv(f"{user_id}.csv", index=False)
