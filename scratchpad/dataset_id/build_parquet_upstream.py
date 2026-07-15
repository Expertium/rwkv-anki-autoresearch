from typing import Iterable
from stats_pb2 import CardEntry, Dataset, DeckEntry, RevlogEntry
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # type: ignore
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from multiprocessing import Pool


def filter_revlog(entries: Iterable[RevlogEntry]):
    return filter(
        lambda entry: entry.button_chosen >= 1
        and (entry.review_kind != 3 or entry.ease_factor != 0),
        entries,
    )


def convert_revlog(entries: Iterable[RevlogEntry]):
    return map(
        lambda entry: {
            "review_time": entry.id,
            "card_id": entry.cid,
            "rating": entry.button_chosen,
            "state": entry.review_kind,
            "duration": entry.taken_millis,
        },
        filter_revlog(entries),
    )


def convert_card(entries: Iterable[CardEntry]):
    return map(
        lambda entry: {
            "card_id": entry.id,
            "note_id": entry.note_id,
            "deck_id": entry.deck_id,
        },
        entries,
    )


def convert_deck(entries: Iterable[DeckEntry]):
    return map(
        lambda entry: {
            "deck_id": entry.id,
            "parent_id": entry.parent_id,
            "preset_id": entry.preset_id,
        },
        entries,
    )


class IdMapper:
    def __init__(self):
        self._mappings = {}

    def get_mapping(self, column_name):
        if column_name not in self._mappings:
            self._mappings[column_name] = {}
        return self._mappings[column_name]

    def factorize(self, series, column_name):
        mapping = self.get_mapping(column_name)
        result = series.map(lambda x: mapping.setdefault(x, len(mapping)))
        return result


def process_and_save(file_path: Path):
    data = open(file_path, "rb").read()
    dataset = Dataset()
    dataset.ParseFromString(data)

    id_mapper = IdMapper()

    df_revlogs = process_revlogs(
        dataset, pd.DataFrame(convert_revlog(dataset.revlogs)), id_mapper
    )
    df_cards = process_cards(pd.DataFrame(convert_card(dataset.cards)), id_mapper)
    df_decks = process_decks(pd.DataFrame(convert_deck(dataset.decks)), id_mapper)

    user_id = int(file_path.stem)
    save_to_parquet(df_revlogs, "revlogs", user_id)
    save_to_parquet(df_cards, "cards", user_id)
    save_to_parquet(df_decks, "decks", user_id)


def process_revlogs(dataset, df, id_mapper):
    if df.empty:
        return df

    df["i"] = df.groupby("card_id").cumcount() + 1
    df["is_learn_start"] = (df["state"] == 0) & (
        (df["state"].shift() != 0) | (df["i"] == 1)
    )
    df["sequence_group"] = df["is_learn_start"].cumsum()
    last_learn_start = (
        df[df["is_learn_start"]].groupby("card_id")["sequence_group"].last()
    )
    df["last_learn_start"] = (
        df["card_id"].map(last_learn_start).fillna(0).astype("int64")
    )
    df["mask"] = df["last_learn_start"] <= df["sequence_group"]
    df = df[df["mask"] == True]
    df.loc[:, "state"] += 1
    df.loc[df["is_learn_start"], "state"] = 0
    df = df.groupby("card_id").filter(lambda group: group["state"].iloc[0] == 0)

    df["review_time"] = df["review_time"].astype("int64")
    df["day_offset"] = df["review_time"].apply(
        lambda x: int((x / 1000 - dataset.next_day_at) / 86400)
    )
    df["day_offset"] = df["day_offset"] - df["day_offset"].min()
    df["elapsed_days"] = df["day_offset"].diff().fillna(0).astype("int64")
    df["elapsed_seconds"] = (df["review_time"].diff().fillna(0) / 1000).astype("int64")
    df.loc[df["state"] == 0, "elapsed_days"] = -1
    df.loc[df["state"] == 0, "elapsed_seconds"] = -1
    df["card_id"] = id_mapper.factorize(df["card_id"], "card_id")
    df.sort_values(by="review_time", inplace=True)
    return df[
        [
            "card_id",
            "day_offset",
            "rating",
            "state",
            "duration",
            "elapsed_days",
            "elapsed_seconds",
        ]
    ]


def process_cards(df, id_mapper):
    if df.empty:
        return df

    df["card_id"] = id_mapper.factorize(df["card_id"], "card_id")
    df["note_id"] = id_mapper.factorize(df["note_id"], "note_id")
    df["deck_id"] = id_mapper.factorize(df["deck_id"], "deck_id")
    return df


def process_decks(df, id_mapper):
    if df.empty:
        return df

    df["deck_id"] = id_mapper.factorize(df["deck_id"], "deck_id")
    df["parent_id"] = id_mapper.factorize(df["parent_id"], "deck_id")
    df["preset_id"] = id_mapper.factorize(df["preset_id"], "preset_id")
    return df


def save_to_parquet(df, table_name, user_id):
    if df.empty:
        return

    df["user_id"] = user_id
    table = pa.Table.from_pandas(df)
    output_path = Path(f"../anki-revlogs-10k/{table_name}")

    pq.write_to_dataset(
        table,
        output_path,
        partition_cols=["user_id"],
        existing_data_behavior="delete_matching",
    )

    # rename the file to user_id=xxx
    for file in (output_path / f"user_id={user_id}").glob("*.parquet"):
        new_name = file.with_name("data.parquet")
        file.rename(new_name)


def test():
    process_and_save(Path(f"./revlogs/{70}.revlog"))


def main():
    revlogs_folder = Path("revlogs")
    revlog_files = tuple(revlogs_folder.glob("*.revlog"))

    with Pool() as pool:
        list(
            tqdm(
                pool.imap_unordered(process_and_save, revlog_files),
                total=len(revlog_files),
            )
        )


if __name__ == "__main__":
    main()
    # test()
