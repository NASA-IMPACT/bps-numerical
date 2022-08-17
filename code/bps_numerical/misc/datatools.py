#!/usr/bin/env python3

import itertools
from pathlib import Path
import pickle
import random
import tempfile
from typing import Optional, Union, Any, Type

import numpy as np
import pandas as pd
from sklearn.utils import _safe_indexing

from loguru import logger


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


def train_test_indexed_split(
    *data: pd.DataFrame, test_size: float = 0.2, shuffle: bool = True
) -> dict:
    """
    This function is used to return:
        - splitted data
        - indices for the split
    """
    assert len(data) > 0
    n_samples = data[0].shape[0]

    if not all(_data.shape[0] == n_samples for _data in data):
        raise ValueError("Shape mismatch between all the input data.")
    indices = np.arange(0, n_samples)
    if shuffle:
        random.shuffle(indices)

    n_samples_train = int((1 - test_size) * n_samples)
    train_indices, test_indices = indices[:n_samples_train], indices[n_samples_train:]

    return dict(
        data=list(
            itertools.chain.from_iterable(
                (_safe_indexing(_data, train_indices), _safe_indexing(_data, test_indices))
                for _data in data
            )
        ),
        indices=dict(train=train_indices, test=test_indices),
    )


class LoadSaveMixin:
    def save(self, fname: Union[str, Path] = "tmp/classifier.pkl") -> Type[Any]:
        """
        Save the classifier to a binary file using pickle

        Args:
            `fname`: ```Union[str, Path]```
                Where to save the file?
                    - If directory, a random unique filename is added
                        in the format:
                            <classname>__<len(cols_genes)>__<random_string>
                    - If file, that name is used

        Returns:
            The class object itself.

        """
        path = Path(fname)
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            tmp_name = next(tempfile._get_candidate_names())
            path = path.joinpath(f"{self.__classname__}__{len(self.cols_genes)}__{tmp_name}.pkl")
        logger.info(f"Saving classifier to {path}")
        with open(path, "wb") as fp:
            pickle.dump(self, fp)
        return self

    @classmethod
    def load(cls, fname: Union[str, Path]) -> Type[Any]:
        """
        Load the classifier object

        Args:
            `fname`: ```Union[str, Path]```
                Path to the file to load

        Returns:
            The classifier object

        """
        with open(fname, "rb") as fp:
            return pickle.load(fp)


def main():
    # sanity check
    import os

    src = os.get("BPS_METADATA_CSV")
    df = load_csv(src, p=0.1, n_samples=None, random_sampling=True)
    print(df.shape)


if __name__ == "__main__":
    main()
