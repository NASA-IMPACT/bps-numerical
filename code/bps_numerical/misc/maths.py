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
