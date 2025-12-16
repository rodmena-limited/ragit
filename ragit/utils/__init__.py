#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit utilities module.
"""

from collections import deque
from collections.abc import Hashable, Sequence
from datetime import datetime
from math import floor
from typing import Any

import pandas as pd


def get_hashable_repr(dct: dict[str, object]) -> tuple[tuple[str, object, float, int | None], ...]:
    """
    Returns a hashable representation of the provided dictionary.
    """
    queue: deque[tuple[str, object, float, int | None]] = deque((k, v, 0.0, None) for k, v in dct.items())
    dict_unpacked: list[tuple[str, object, float, int | None]] = []
    while queue:
        key, val, lvl, p_ref = queue.pop()
        if hasattr(val, "items"):  # we have a nested dict
            dict_unpacked.append((key, "+", lvl, p_ref))
            if hash(key) != p_ref:
                lvl += 1
            queue.extendleft((k, v, lvl, hash(key)) for k, v in val.items())
        elif isinstance(val, Hashable):
            dict_unpacked.append((key, val, lvl, p_ref))
        elif isinstance(val, Sequence):
            dict_unpacked.append((key, "+", lvl, p_ref))
            queue.extendleft((key, vv, floor(lvl) + ind * 0.01, hash(key)) for ind, vv in enumerate(val, 1))
        else:
            raise ValueError(f"Unsupported type in dict: {type(val)}")

    return tuple(sorted(dict_unpacked, key=lambda it: (it[2], it[0])))


def remove_duplicates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicates list of provided dictionary items.

    Parameters
    ----------
    items : list[dict]
        List of items to deduplicate.

    Returns
    -------
    list[dict]
        A deduplicated list of input items.
    """
    duplicate_tracker = set()
    deduplicated_items = []
    for ind, elem in enumerate(map(get_hashable_repr, items)):
        if elem not in duplicate_tracker:
            duplicate_tracker.add(elem)
            deduplicated_items.append(items[ind])
    return deduplicated_items


def handle_missing_values_in_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in experiment data combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data with combinations being explored.

    Returns
    -------
    pd.DataFrame
        Data with NaN values properly replaced.
    """
    if "chunk_overlap" in df.columns:
        df["chunk_overlap"] = df["chunk_overlap"].map(lambda el: 0 if pd.isna(el) else el)

    return df


def datetime_str_to_epoch_time(timestamp: str | int) -> str | int:
    """
    Convert datetime string to epoch time.

    Parameters
    ----------
    timestamp : str | int
        Either a datetime string or a unix timestamp.

    Returns
    -------
    int
        Unix timestamp or -1 if parsing fails.
    """
    if not isinstance(timestamp, str):
        return timestamp
    try:
        iso_parseable = datetime.fromisoformat(timestamp)
    except ValueError:
        return -1
    return int(iso_parseable.timestamp())
