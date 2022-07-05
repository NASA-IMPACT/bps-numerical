#!/usr/bin/env python3

import itertools
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from .clustering import CorrelationClusterer


class FeatureSelector(ABC):
    def __init__(self, clusterer: CorrelationClusterer) -> None:
        self.clusterer = clusterer

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
        if isinstance(cluster_map, dict) and len(cluster_map) > 0:
            return cluster_map
        if df is not None and isinstance(df, pd.DataFrame):
            cluster_map = self.clusterer.cluster(df, **kwargs)
        return cluster_map


class FirstFeatureSelector(FeatureSelector):
    def _select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        return list(map(lambda clusters: clusters[0], cluster_map.values()))


class LastFeatureSelector(FeatureSelector):
    def select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        return list(map(lambda clusters: clusters[-1], cluster_map.values()))


class RandomizedSingleFeatureSelector(FeatureSelector):
    def _select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        return list(map(lambda clusters: random.choice(clusters), cluster_map.values()))


class KRandomizedFeatureSelector(FeatureSelector):
    def __init__(self, clusterer: CorrelationClusterer, k_features: int = 1) -> None:
        super().__init__(clusterer)
        self.k_features = k_features

    def _select_features(self, cluster_map: Dict[int, List[str]], **kwargs) -> List[str]:
        res = map(
            lambda clusters: random.choices(clusters, k=self.k_features), cluster_map.values()
        )
        return list(set(itertools.chain(*res)))


def main():
    pass


if __name__ == "__main__":
    main()
