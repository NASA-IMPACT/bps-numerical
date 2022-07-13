#!/usr/bin/env python3

import itertools
import os
import random
from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
from bps_numerical.classification import (
    MultiPhenotypeIsolatedClassifier,
    SinglePhenotypeClassifier,
)
from bps_numerical.classification.feature_scorers import PhenotypeFeatureScorer
from bps_numerical.clustering import CorrelationClusterer
from bps_numerical.feature_selection import FirstFeatureSelector
from bps_numerical.preprocessing import merge_gene_phenotype, standardize_gene_data
from loguru import logger

random.seed(42)


def compute_permuted_scores(
    *classifiers,
    top_k: int,
    ignore_zeros: bool = True,
    normalize: bool = True,
) -> Dict[Tuple[str, ...], List[Tuple[str, float]]]:
    def _powerset(items):
        for sl in itertools.product(*[[[], [i]] for i in items]):
            yield {j for i in sl for j in i}

    res = {}
    for objs in _powerset(classifiers):
        if len(objs) < 2:
            continue
        labels = tuple(map(lambda clf: clf.phenotype, objs))
        res[labels] = PhenotypeFeatureScorer(*objs).get_features(
            top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
        )
    return res


def main():
    # feature selection
    gene_csv = os.getenv("BPS_GENE_CSV")
    phenotype_csv = os.getenv("BPS_PHENOTYPE_CSV")
    # merge_column = os.getenv("BPS_MERGE_COLUMN", "Sample")

    df_genes = standardize_gene_data(gene_csv)
    samples = df_genes.pop("Sample")
    df_genes = df_genes.astype(float)

    clusterer = CorrelationClusterer(list(df_genes.columns), cutoff_threshold=0.3, debug=False)
    feature_selector = FirstFeatureSelector(clusterer=clusterer)
    cols_genes = feature_selector.select_features(df_genes)
    logger.debug(f"Total features selected = {len(cols_genes)}")

    # prepr
    df_merged = merge_gene_phenotype(
        pd.concat([samples, df_genes[cols_genes]], axis=1),
        phenotype_csv,
        "Sample",
    )

    # trainer
    ## single phenotype
    # model = xgboost.XGBClassifier()
    # clf = SinglePhenotypeClassifier(
    #     cols_genes=cols_genes,
    #     phenotype="condition",
    #     #     model = model
    # )
    # tracker_single = clf.train(df_merged)

    ## multiple phenotypes
    clf_condition = SinglePhenotypeClassifier(cols_genes, "condition")
    clf_strain = SinglePhenotypeClassifier(cols_genes, "strain")
    clf_gender = SinglePhenotypeClassifier(cols_genes, "gender")
    clf_mission = SinglePhenotypeClassifier(cols_genes, "mission")
    clf_animal_return = SinglePhenotypeClassifier(cols_genes, "animalreturn")
    trainer = MultiPhenotypeIsolatedClassifier(
        cols_genes=cols_genes,
        classifiers=[clf_condition, clf_strain, clf_gender, clf_mission, clf_animal_return],
        debug=True,
    )

    tracker_multi = trainer.train(df_merged)
    pprint(tracker_multi)

    # feature scorer
    top_k = 500
    ignore_zeros = True
    normalize = True
    # get top-k non-zero features for each model
    for clf in trainer.classifiers:
        print(
            clf.phenotype,
            len(
                PhenotypeFeatureScorer(clf).get_features(
                    top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
                )
            ),
        )

    # get common features top_k
    permuted_ = compute_permuted_scores(
        *trainer.classifiers, top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
    )

    pprint(permuted_)
    pprint(dict(map(lambda p: (p[0], len(p[1])), permuted_.items())))


if __name__ == "__main__":
    main()
