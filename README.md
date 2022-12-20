# bps_numerical

This is the repository for the gene ranking algorithm we have developed.

# Installation

- `pip install -e .`
- `python setup.py install`

# Code Structure

Tentatively, there are a few components that can be used

## Data Loader

`bps_numerical.preprocessing.DataLoader` class can be used to load data using the gene expression and metadata CSVs.

```python
from bps_numerical.preprocessing import DataLoader

CSV_GENE = "../data/data-v3/expr-balanced-expanded.csv"
CSV_PHENOTYPE = "../data/data-v3/meta-balanced-expanded.csv"

dataloader = DataLoader(
    csv_gene=CSV_GENE_ORIGINAL_BALANCED,
    csv_phenotype=CSV_PHENOTYPE_ORIGINAL_BALANCED
)

# get the gene expression data
df_genes = dataloader.gene_data

# get all the gene names
cols_genes = dataloader.cols_genes_all

# or you can use selected genes using some feature selection method (see `bps_numerical.feature_selection` module)

# get full data with genes + metadata columns
df_merged = dataloader.get_merged_data(cols_genes=cols_genes)
```

## Clustering

`bps_numerical.clustering.CorrelationClusterer` uses correlation-based hierarchical clustering to group genes

```python
from bps_numerical.clustering import CorrelationClusterer

clusterer = CorrelationClusterer(
    df_genes.columns.to_list(),
    cutoff_threshold=0.2,
    correlation_type="spearman",
    debug=False
)

# get the dictionary of clusters
cluster_map = clusterer.cluster(df_genes)
```

## Feature Selection

`bps_numerical.feature_selection` module has a handful of gene selection algorithm using the clusters. These genes then could be used for any downstream tasks.

- `bps_numerical.feature_selection.FirstFeatureSelector` selects only the first gene from each cluster
- `bps_numerical.feature_selection.LastFeatureSelector` selects only the last gene from each cluster
- `bps_numerical.feature_selection.RandomizedSingleFeatureSelector` selects a random single gene from each cluster
- `bps_numerical.feature_selection.KRandomizedFeatureSelector` selects  **k** random genes from each cluster
- `bps_numerical.feature_selection.BestCandidateFeatureSelector` uses distance-based metric to best select a candidate gene from each cluster

All these feature selector uses the `CorrelationClusterer` object.

These feature selectors also provide `FeatureSelector.load()` and `FeatureSelector.save()` method


```python
from bps_numerical.feature_selection import (
    FirstFeatureSelector,
    LastFeatureSelector,
    RandomizedSingleFeatureSelector,
    KRandomizedFeatureSelector,
    BestCandidateFeatureSelector
)


clusterer = CorrelationClusterer(
    df_genes.columns.to_list(),
    cutoff_threshold=0.2,
    correlation_type="spearman",
    debug=False
)

# feature_selector = FirstFeatureSelector(clusterer=clusterer)
# feature_selector = LastFeatureSelector(clusterer=clusterer)
# feature_selector = KRandomizedFeatureSelector(clusterer=clusterer, k_features=3)


# feature_selector = BestCandidateFeatureSelector.load("tmp/fs.pkl")
feature_selector = BestCandidateFeatureSelector(clusterer=clusterer)

cols_genes = feature_selector.select_features(df_genes)

feature_selector.save("tmp/fs.pkl")
```

## Bayesian Tuner

This module is used for performing parameter search (bayesian) using `scikit-optimize` library.

```python
from skopt.callbacks import DeltaYStopper, DeadlineStopper, CheckpointSaver
from bps_numerical.classification.tuner import BayesTuner

tuner = BayesTuner(
    columns=cols_genes,
    target_column=phenotype,
    n_iter=12,
    n_jobs=12,
    k_folds=3,
    objective="multi:softmax", # if binary use "binary:logistic", else "multi:softmax"
    callbacks=[
        DeltaYStopper(delta=1e-4),
        DeadlineStopper(12*60*60), #12 hours
        CheckpointSaver(f"tmp/checkpoints/checkpoint-{phenotype}-nov-16.pkl", compress=9)
    ], # if None -> no callbacks are used, if List -> use that, if empty list -> default callbacks
    debug=True
)

# we could remove unwanted search param like this
tuner.search_spaces.pop("subsample", None)

# find it
params = tuner.search(df_merged)
```

## Gene Ranking Methodlogies

`bps_numerical.classification.feature_scorers.GeneRanker` component is the main ranking algorithm to be used.
We can also apply MRR-based algorithm `bps_numerical.classification.feature_scorers.MeanReciprocalRanker`.

It happens in multiple stages:
a. Perform bayesian search (using `bps_numerical.classification.tuner.BayesTuner`) to get best params
b. Split datasets into multiple train/test views (and also shuffle columns) and train N xgboost models using the best param
c. Get top-k features from each model (normalization, 0-score feature removal, etc are also performed)
d. Find common (ranked) genes (we can use `bps_numerical.classification.MeanReciprocalRanker` to rank features)

```python
from bps_numerical.classification.feature_scorers import GeneRanker, MeanReciprocalRanker
from bps_numerical.classification.tuner import BayesTuner

cols_genes = [...]
phenotype = "gender"

# setup tuner from scratch
tuner = BayesTuner(
    columns=cols_genes,
    target_column=phenotype,
    n_iter=12,
    n_jobs=12,
    k_folds=3,
    objective="multi:softmax", # if binary use "binary:logistic", else "multi:softmax"
    callbacks=[
        DeltaYStopper(delta=1e-4),
        DeadlineStopper(12*60*60), #12 hours
        CheckpointSaver(f"tmp/checkpoints/checkpoint-{phenotype}-nov-16.pkl", compress=9)
    ], # if None -> no callbacks are used, if List -> use that, if empty list -> default callbacks
    debug=True
)
tuner.search_spaces.pop("subsample", None)

# if you want extra xgboost params, you can provide as kwargs at the end
ranker = GeneRanker(
    cols_genes=cols_genes,
    phenotype=phenotype,
    n_runs = 100,
    params_finder=tuner,
    debug=True,
    shuffle_columns=True,
    objective="multi:softmax",
    num_class=5,
)

# get common genes
features_ranked = ranker.get_features(
    df_merged,
    test_size=0.1,
    top_k=500,
    ignore_zeros=True,
    normalize=True
)

# get MRR-based genes
features_mrr = MeanReciprocalRanker(
    *[
        _clf
        for _clf, _result in zip(ranker.classifiers, ranker.results)
        if _result["train_score"]==1.0 and _result["test_score"] == 1.0
    ],
    score_cutoff=0.1,
    rank_cutoff=0.1,
    debug=True,
).get_features()

```
