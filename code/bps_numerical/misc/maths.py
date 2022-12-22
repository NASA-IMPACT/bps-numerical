from copy import deepcopy
from functools import reduce
from typing import Sequence

import numpy as np


def min_max_normalization(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes incoming array using min-max
    """
    arr = np.array(arr)
    minx = np.min(arr)
    maxx = np.max(arr)
    if minx == maxx == 0:
        return arr
    return (arr - minx) / (maxx - minx)


def chain(*funcs):
    """
    Chain a sequence of functions
    """

    def chained_call(arg):
        return reduce(lambda r, f: f(r), funcs, arg)

    return chained_call


def shuffle_copy(vals: Sequence) -> Sequence:
    """
    This shuffles the incoming sequence (list/tuple)
    without changing inplace by doing deepcopy and shuffling.
    """
    vals = deepcopy(vals)
    np.random.shuffle(vals)
    return vals
