#!/usr/bin/env python3

import itertools
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .clustering import CorrelationClusterer
from .misc.datatools import LoadSaveMixin


class FeatureSelector(LoadSaveMixin, ABC):
    """
    This component is used for selecting features
    after clustering.

    Note:
        `_select_features` abstractmethod should be implemented
        by dowstream children.
    """

    def __init__(self, clusterer: CorrelationClusterer, debug: bool = False) -> None:
        """
        Args:
            clusterer: ```CorrelationClusterer```
                clustering object of the type CorrelationClusterer.
                If provided, and if the input `cluster_map` to
                `select_features(...)` method is invalid, this object
                is used to compute the cluster
        """
        self.clusterer = clusterer
        self.debug = bool(debug)

    def __call__(
        self,
        df: Optional[pd.DataFrame] = None,
        cluster_map: Optional[Dict[int, List[str]]] = None,
        **kwargs,
    ) -> List[str]:
        return self.select_features(df, cluster_map, **kwargs)

    def select_features(
        self,
        df: Optional[pd.DataFrame] = None,
        cluster_map: Optional[Dict[int, List[str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        This is the entrypoint method to do feature selection

        Args:
            df: ```Optional[pd.DataFrame]```
                Pandas dataframe (df) with only genes features

                If `cluster_map` is not provided, this dataframe
                is used to recompute all the clusters and then do the selection.

            cluster_map: ```Optional[dict]```
                A dictionary cache for final cluster.
                See `bps_numerica.clustering.CorrelationClusterer` for more reference

                If this is provided, we try to use it directly to remove any
                re-calculation

        Returns:
            List of selected gene feature string
        """
        cluster_map = self._get_cluster_map(df, cluster_map, **kwargs)
        if not isinstance(cluster_map, dict):
            raise TypeError(f"Invalid type for cluster_map. Expected dict. Got {type(cluster_map)}")
        return self._select_features(cluster_map, **kwargs)

    @abstractmethod
    def _select_features(
        self, cluster_map: Optional[Dict[int, List[str]]] = None, **kwargs
    ) -> List[str]:
        raise NotImplementedError()

    def _get_cluster_map(
        self,
        df: Optional[pd.DataFrame] = None,
        cluster_map: Optional[Dict[int, List[str]]] = None,
        **kwargs,
    ) -> Dict[int, List[str]]:
        """
        This method sanity-checks the input dataframe / cluster_map cache.
        If cluster_map is present, it will *not* compute the clusters and
        return the same cluster map.

        Args:
            df: ```Optional[pd.DataFrame]````
                Dataframe got after pre-processing gene/phenotype CSVs

            cluster_map: ```Optional[Dict[int, List[str]]]```
                A dictionary cache for the cluster.
                It is normally the result from `bps_numerical.clustering.CorrelationClusterer`

        Returns:
            The cluster_map (same data structure as that of `cluster_map` input dict)

        """
        if isinstance(cluster_map, dict) and len(cluster_map) > 0:
            return cluster_map
        if df is not None and isinstance(df, pd.DataFrame):
            cluster_map = self.clusterer.cluster(df, **kwargs)
        return cluster_map


class FirstFeatureSelector(FeatureSelector):
    """
    This selector selects only the "first" feature/gene from each cluster
    """

    def _select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        return list(map(lambda clusters: clusters[0], cluster_map.values()))


class LastFeatureSelector(FeatureSelector):
    """
    This selector selects only the "last" feature/gene from each cluster
    """

    def select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        return list(map(lambda clusters: clusters[-1], cluster_map.values()))


class RandomizedSingleFeatureSelector(FeatureSelector):
    """
    This selector selects only "one" random gene from each cluster
    """

    def _select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        return list(map(lambda clusters: random.choice(clusters), cluster_map.values()))


class KRandomizedFeatureSelector(FeatureSelector):
    """
    This selector selects **k** random genes from each cluster
    """

    def __init__(self, clusterer: CorrelationClusterer, k_features: int = 1) -> None:
        super().__init__(clusterer)
        self.k_features = k_features

    def _select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        res = map(
            lambda clusters: random.choices(clusters, k=self.k_features), cluster_map.values()
        )
        return list(set(itertools.chain(*res)))


class RandomFeatureSelector(FeatureSelector):
    """
    Randomly select features.

    Args:
        `n_random`: `Union[int, float]`
            If int, it represents total number of genes to get randomly.
            If float, it represents fraction of genes to get randomly.
    """

    def __init__(self, n_random: Union[int, float] = 1.0) -> None:
        if not isinstance(n_random, (int, float)):
            raise TypeError(
                f"Invalid type for n_random. Expected (float, int). Got {type(n_random)}"
            )
        self.n_random = n_random

    def select_features(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> List[str]:
        """
        This is the entrypoint method to do feature selection

        Args:
            df: ```Optional[pd.DataFrame]```
                Pandas dataframe (df) with only genes features

                If `cluster_map` is not provided, this dataframe
                is used to recompute all the clusters and then do the selection.


        Returns:
            List of selected gene feature string
        """
        columns = list(df.columns)
        n_samples = (
            int(self.n_random * len(columns)) if isinstance(self.n_random, float) else self.n_random
        )
        return random.sample(columns, k=n_samples)

    def _select_features(self) -> None:
        raise NotImplementedError("This method is not required in RandomFeatureSelector")


class BestCandidateFeatureSelector(FeatureSelector):
    """
    This selector tries to get the single best candidates from each cluster.
    The best candidate is tentatively that point for which the distance
    to all the other points in the cluster is minimum.
    """

    def _select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        assert (
            self.clusterer.dist_matrix is not None
        ), "CorrelationClusterer.dist_matrix should be NxN matrix"
        cols = self.clusterer.column_names
        gene_index_map = {cols: idx for idx, cols in enumerate(cols)}
        res = []
        for idx, cluster in cluster_map.items():
            indices = list(map(lambda c: gene_index_map[c], cluster))
            candidate = self._get_best_candidate(indices, self.clusterer.dist_matrix)
            res.append(cols[candidate])
        return res

    def _get_best_candidate(self, indices: List[int], dist_matrix: np.ndarray):
        """
        Find best candidate based on distance matrix.

        Args:
            `indices`: ```List[int]```
                List of feature index for which best candidate is to be computed

            `dist_matrix`: ```np.ndarray```
                This is actually the `1-abs(corr)` matrix.
                (Two points have distance of 0 if highly correlated.)

        Returns:
            Column index of the candidate


        """
        dists = dist_matrix[indices][:, indices]
        return indices[np.argmin(dists.sum(axis=1))]


def main():
    pass


if __name__ == "__main__":
    main()
