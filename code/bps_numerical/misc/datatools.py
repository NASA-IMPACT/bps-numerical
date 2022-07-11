#!/usr/bin/env python3

import random
from typing import Optional, Union

import pandas as pd


def load_csv(
    fname: Union[str, pd.DataFrame],
    p: float = 0.1,
    random_sampling: bool = False,
    n_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads csv partially or fully.
    Also allows random sampling

    Args:
        fname: `str`
            Input csv path

        p: `float`
            How many rows to load if random_sampling is enabled

    """
    if isinstance(fname, pd.DataFrame):
        return fname
    if not random_sampling:
        if isinstance(n_samples, int) and n_samples > 0:
            return pd.read_csv(fname, nrows=n_samples)
        else:
            return pd.read_csv(fname)
    return pd.read_csv(fname, header=0, skiprows=lambda i: i > 0 and random.random() > p)


def main():
    # sanity check
    import os

    src = os.get("BPS_METADATA_CSV")
    df = load_csv(src, p=0.1, n_samples=None, random_sampling=True)
    print(df.shape)


if __name__ == "__main__":
    main()
