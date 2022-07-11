#!/usr/bin/env python3

import time
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from loguru import logger
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


class CorrelationClusterer:
    """
    This component is used to cluster input genes/features
    based on correlation.
    Correlation can be:
        - pearson
        - spearman
    """

    def __init__(
        self,
        column_names: List[str],
        cutoff_threshold: float = 0.75,
        debug: bool = False,
        correlation_type: str = "pearson",
    ):
        self.column_names = column_names
        self.cutoff_threshold = cutoff_threshold
        self.labels = []
        self.cluster_map = {}
        self.debug = debug
        self.correlation_type = correlation_type

    @staticmethod
    def compute_pearson_correlation(df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(df, pd.DataFrame):
            df = df.to_numpy().T.astype(np.float32)
        return np.corrcoef(df)

    @staticmethod
    def compute_spearman_correlation(df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(df, pd.DataFrame):
            df = df.rank().to_numpy().T.astype(np.float32)
            return np.corrcoef(df)
        return spearmanr(df)

    def cluster(self, arr: Union[np.ndarray, pd.DataFrame], **kwargs) -> Dict[int, List[str]]:
        """
        Computes cluster map as a dictionary:
            - each key is a integer label index
            - each value is a list of feature/column names
        """
        if self.cluster_map:
            logger.info(f"Using cached cluster_map of size {len(self.cluster_map)}")
            return self.cluster_map

        if isinstance(arr, pd.DataFrame):
            logger.debug(f"Computing correlation for {arr.shape}")
            arr = CorrelationClusterer._CORRELATION_FUNCS.get(
                self.correlation_type, CorrelationClusterer.compute_pearson_correlation
            )(arr)

        start = time.time()
        assert arr.shape[0] == arr.shape[1]

        n_features = kwargs.get("n_features", arr.shape[0])
        if self.debug:
            logger.debug(f"Using n_features = {n_features} | Slicing array...")
        arr = arr[:n_features, :n_features]

        cols = self.column_names[:n_features]
        labels = self._cluster(arr, cutoff_threshold=self.cutoff_threshold, columns=cols)
        assert len(labels) == len(cols)

        self.labels = labels
        self.cluster_map = self._group_by_labels(cols, labels)

        logger.debug(
            f"Took {time.time()-start} seconds to form {len(self.cluster_map)} clusters..."
        )
        return self.cluster_map

    def _cluster(
        self, arr: np.ndarray, columns: List[str], cutoff_threshold: float = 0.5
    ) -> List[int]:
        """
        Computes flattened clusters using agglomerative clustering.

        Args:
            arr: ```np.ndarray```
                Input correlation array of shape: NxN

            cutoff_threshold: ```float```
                The distance cutoff to separate two clusters.

                Note: This value doesn't mean "correlation score".
                It is meant as a geometric distance cutoff.

            columns: ```List[str]```
                Column/gene names to be assigned for the cluster plot
                Note: is only used for debug mode

        Returns:
            List of integer labels. Each label has one-to-one correspondence
            to the gene columns.
        """
        logger.info("Clustering in progress...")
        dists = 1 - np.round(abs(arr), 3)
        hierarchy = sch.linkage(squareform(dists), method='average')
        labels = sch.fcluster(hierarchy, cutoff_threshold, criterion='distance')

        if self.debug:
            logger.debug("Plotting dendrogram...")
            plt.figure(figsize=(100, 45))
            plt.axhline(y=self.cutoff_threshold, c='grey', lw=5, linestyle='dashed')
            plt.subplots_adjust(left=0.07, bottom=0.3, right=0.98, top=0.95, wspace=0, hspace=0)
            plt.xlabel("Genes")
            plt.ylabel("Dissimilarity")
            sch.dendrogram(
                hierarchy,
                color_threshold=self.cutoff_threshold,
                leaf_rotation=90.0,
                leaf_font_size=1.0,
                labels=columns,
            )
            plt.tight_layout()
            fname = "tmp/dendrogram.jpg"
            logger.debug(f"Saving dendrogram plot to {fname}")
            plt.savefig(fname, dpi=200)
        return labels

    def _cluster2(self, arr: np.ndarray, cutoff_threshold: float = 0.5):
        dissimilarity = 1 - np.round(abs(arr), 3)
        hierarchy = sch.ward(squareform(dissimilarity))
        labels = sch.fcluster(hierarchy, cutoff_threshold, criterion='distance')
        return labels

    def _group_by_labels(self, columns: List[str], labels: List[int]) -> Dict[int, List[int]]:
        """
        This method creates the cluster map (dict) where:
            - key represents cluster label
            - value is a list of column/gene strings
        """
        assert len(labels) == len(columns)
        cluster_map = {}
        for label, col in zip(labels, columns):
            group = cluster_map.get(label, [])
            group.append(col)
            cluster_map[label] = group
        return cluster_map

    _CORRELATION_FUNCS = dict(
        pearson=compute_pearson_correlation.__func__,
        spearman=compute_spearman_correlation.__func__,
    )


def main():
    pass


if __name__ == "__main__":
    main()
