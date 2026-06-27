"""Trimmed vendor of srs-benchmark's top-level utils.py.

The RWKV preprocessing pipeline only uses three helpers from this module:
- get_bin / count_lapse  -> RMSE(bins) bucketing in find_equalize_test_reviews.py
- cum_concat             -> history accumulation in features/base.py

The original utils.py also pulled in matplotlib, models.trainable, fsrs_optimizer
and a large Collection/evaluation harness — none of which the RWKV path needs.
Those were dropped to keep this repo RWKV-only. Function bodies below are verbatim
from the source so behavior matches the benchmark exactly.
"""

from itertools import accumulate

import numpy as np


def cum_concat(x):
    """Concatenate a list of lists using accumulate.

    Args:
        x: A list of lists to be concatenated

    Returns:
        A list of accumulated concatenated lists
    """
    return list(accumulate(x))


def count_lapse(r_history, t_history):
    lapse = 0
    for r, t in zip(r_history.split(","), t_history.split(",")):
        if t != "0" and r == "1":
            lapse += 1
    return lapse


def get_bin(row):
    raw_lapse = count_lapse(row["r_history"], row["t_history"])
    lapse = (
        round(1.65 * np.power(1.73, np.floor(np.log(raw_lapse) / np.log(1.73))), 0)
        if raw_lapse != 0
        else 0
    )
    delta_t = round(
        2.48 * np.power(3.62, np.floor(np.log(row["delta_t"]) / np.log(3.62))), 2
    )
    i = round(1.99 * np.power(1.89, np.floor(np.log(row["i"]) / np.log(1.89))), 0)
    return (lapse, delta_t, i)
