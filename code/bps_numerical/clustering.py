#!/usr/bin/env python3

import random
import time
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from loguru import logger
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from tqdm import tqdm

from .classification.classifiers import SinglePhenotypeClassifier


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


class SamplingBasedClusterAnalyzer:
    """
    This component is used to compute the "goodness" of our cluster
    formed from `CorrelationClusterer`. This is done by selecting n random
    genes and replacing them with another random genes from the same cluster.

    The algorithm tentatively works as:
        - Out of `cols_genes`, get `n_replacement` number of genes to replace
        - For each genes to be replaced, generate `max_sampling` number of
            random genes
        - So, for each gene, we pick 1 random gene at a time from the list
        - We take the original dataset and replace the values in the original
        genes with new ones

    Args:
        `clusterer`: ```CorrelationClusterer```
            CorrelationClusterer object that performs clustering
            Note: if clustering is already performed, it uses cached result

        `cols_genes`: ```List[str]```
            Input list of genes

        `trainer`: ```SinglePhenotypeClassifier```
            Trainer object for only single phenotype that is to be tested

        `n_replacement`: ```int```
            Total number of genes we're replacing with new ones
            Note:
                If the cluster size for the genes is less than some threshold
                (`min_cluster_size`), then such genes are replaced.
                So, `total_genes <= n_replacement` in general

        `max_sampling`: ```int```
            For each gene to be replaced, how many random genes from the same
            cluster to use?

        `min_cluster_size`: `int`
            Minimum cluster size from which gene sampling is to be considered.

        `debug`: ```bool```
            If enabled, some debugging logs will be printed

    Usage:

        .. code-block: python

            clf_condition = SinglePhenotypeClassifier(cols_genes, "condition", debug=False)
            cluster_analyzer = SamplingBasedClusterAnalyzer(
                clusterer,
                cols_genes,
                clf_condition,
                n_replacement=1500,
                max_sampling=100,
                debug=True,
            )
            metrics = cluster_analyzer.analyze(
                data_merged=df_merged,
                data_genes=df_genes,
            )



    """

    def __init__(
        self,
        clusterer: CorrelationClusterer,
        cols_genes: List[str],
        trainer: SinglePhenotypeClassifier,
        n_replacement: int = 500,
        max_sampling: int = 25,
        min_cluster_size: int = 5,
        debug: bool = False,
    ) -> None:
        self.clusterer = clusterer
        self.cols_genes = cols_genes
        self.trainer = trainer
        self.debug = debug
        self.n_replacement = int(n_replacement)
        self.max_sampling = int(max_sampling)
        self.min_cluster_size = int(min_cluster_size)

    def analyze(self, data_merged: pd.DataFrame, data_genes: pd.DataFrame, **kwargs) -> List[dict]:
        """
        This is the entrypoint method to analyze the cluster.
        First, the model is trained with input data and then analyzed with
        sampling-based algorithm (see `evaluate(...)`)

        Args:
            `data_merged`: ```pd.DataFrame```
                Input dataframe consisting of both genes and phenotypes
                Note:
                    - This data is used for training/testing the model
                    - This doesn't necessarily imply full dataset
                        (merging of data can happen after feature selection)

            `data_genes`: ```pd.DataFrame```
                Full original dataframe only for genes
                Note:
                    - This is used to replace the gene values in the input data
        """
        cluster_map = self.clusterer.cluster(data_genes, **kwargs)
        cluster_map = self._restructure_cluster_map(cluster_map)

        train_results = self.trainer.train(data_merged)
        eval_results = self.evaluate(
            data_merged,
            data_genes,
            train_results["indices"]["train"],
            train_results["indices"]["test"],
            cluster_map,
            train_results["labels"],
        )
        return eval_results

    def evaluate(
        self,
        data_merged: pd.DataFrame,
        data_genes: pd.DataFrame,
        train_indices: List[int],
        test_indices: List[int],
        cluster_map: Dict[str, Tuple[str]],
        labels: List[str],
    ) -> List[Dict[str, float]]:
        """
        This is the main method where we perform the replacement->evaluation.

        Args:
            `data_merged`: ```pd.DataFrame```
                Input dataframe consisting of both genes and phenotypes
                Note:
                    - This data is used for training/testing the model
                    - This doesn't necessarily imply full dataset
                        (merging of data can happen after feature selection)

            `data_genes`: ```pd.DataFrame```
                Full original dataframe only for genes
                Note:
                    - This is used to replace the gene values in the input data

            `train_indices`: ```List[int]```
                Indices referencing to training samples in `data_merged`

            `test_indices`: ```List[int]```
                Indices referencing to test samples in `data_merged`

            `cluster_map`: ```Dict[str, Tuple[str]]```
                Cluster map to use for sampling

            `labels`: ```List[str]```
                Label-encoded labels that is used for target data.

        Returns:
            List of dictionary where each dict has train/test score from the model
            performance once we sample the genes.
        """

        # Step 1: Seeding
        # get initial sets of genes
        genes_to_replace = random.sample(
            self.cols_genes, k=min(self.n_replacement, len(self.cols_genes))
        )

        # Step 2: Sampling
        # for each target genes to replace, get new genes from the same cluster
        # Note: if the new gene list is empty, we don't consider the sampled gene in the analysis
        genes_to_replace = dict(
            filter(
                lambda x: len(x[1]) > 1,
                zip(
                    genes_to_replace,
                    map(
                        lambda gene: self._sample_columns_from_cluster(
                            cluster_map,
                            gene,
                            min_cluster_size=self.min_cluster_size,
                            n_samples=self.max_sampling,
                        ),
                        genes_to_replace,
                    ),
                ),
            )
        )

        # We replace these genes from the input data with random genes from the
        # same cluster
        source_genes = list(genes_to_replace.keys())
        if self.debug:
            logger.debug(f"Total genes to be replaced => {len(source_genes)}")

        # Step 3: Replacement
        # Since for each gene we could have k number of genes to replace
        # So, we replace with different genes iteratively
        metrics_tracker = []
        for replacers in tqdm(zip(*genes_to_replace.values())):
            data_to_use = data_merged.copy()
            data_to_use[source_genes] = data_genes[list(replacers)]

            train_data = data_to_use.iloc[train_indices]
            test_data = data_to_use.iloc[test_indices]
            X_train, X_test = train_data[self.cols_genes], test_data[self.cols_genes]
            Y_train, Y_test = (
                train_data[[self.trainer.phenotype]],
                test_data[[self.trainer.phenotype]],
            )
            Y_train, Y_test = pd.get_dummies(train_data[[self.trainer.phenotype]]), pd.get_dummies(
                test_data[[self.trainer.phenotype]]
            )
            Y_train = Y_train[labels]
            Y_test = Y_test[labels]
            test_score = self.trainer.model.score(X_test, Y_test)
            train_score = self.trainer.model.score(X_train, Y_train)
            if self.debug:
                logger.debug(f"train_score={train_score} | test_score={test_score}")
            metrics_tracker.append(dict(test_score=test_score, train_score=train_score))
        return metrics_tracker

    def _sample_columns_from_cluster(
        self,
        cluster_map: Dict[str, Tuple[str]],
        gene: str,
        min_cluster_size: int = 3,
        n_samples: int = 5,
    ) -> List[str]:
        """
        For a given 'gene', it generates `n_samples` number of random gene
        from the same cluster. If the cluster size <`min_cluster_size`, empty
        list is returned

        Args:
            `cluster_map`: ```Dict[str, Tuple[str]]```
                Cluster map dict that you get from `_restructure_cluster_map`

            `gene`: ```str```
                Which gene to replace?

            `min_cluster_size`: ```int```
                Minimum size of cluster for the given gene to generate random genes

            `n_samples`: ```int```
                How many genes to sample from the given cluster where 'gene' lies?

        Returns:
            List of sampled gene strings

        """
        cluster = list(cluster_map.get(gene, []))
        if len(cluster) <= min_cluster_size:
            return []

        # Trivial: we don't want the current gene to be replaced by itself
        cluster.remove(gene)
        n_samples = min(len(cluster), n_samples)
        return random.sample(cluster, k=(min(len(cluster), n_samples)))

    def _restructure_cluster_map(self, cluster_map: Dict[int, List[str]]) -> Dict[str, Tuple[str]]:
        """
        This restructures the input cluster_map dict in the format:

            .. code-block: python

                {
                    "geneA": (geneA, geneB),
                    "geneB": (geneA, geneB),
                    "geneC": (geneC, geneD),
                    "geneD": (geneC, geneD),
                    ...
                }

        Args:
            cluster_map: ```Dict[int, List[str]]```
                Original cluster map to restructure

        Returns:
            A re-structured dict
        """
        resultant = {}
        for idx, cluster in cluster_map.items():
            cluster = tuple(cluster)
            for candidate in cluster:
                resultant[candidate] = cluster
        return resultant


def main():
    pass


if __name__ == "__main__":
    main()
