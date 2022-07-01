#!/usr/bin/env python3

import sys

from bps_numerical.clustering import CorrelationClusterer
from bps_numerical.feature_selection import FirstFeatureSelector
from bps_numerical.preprocessing import standardize_gene_data

sys.path.append("./")
sys.path.append("../code/")
sys.path.append("./code/")


def main():
    gene_csv = (
        "/Users/nishparadox/dev/uah/nasa-impact/gene-experiments/data/OneDrive_1_3-21-2022/gen.csv"
    )

    df_genes = standardize_gene_data(gene_csv)
    df_genes.pop("Sample")
    clusterer = CorrelationClusterer(list(df_genes.columns), cutoff_threshold=0.3, debug=False)
    # cluster_map = clusterer.cluster(df_genes, n_features=10)
    # print(len(cluster_map))

    fs = FirstFeatureSelector(clusterer=clusterer)
    features = fs.select_features(df_genes)
    print(features[:10])
    print(len(features))


if __name__ == "__main__":
    main()