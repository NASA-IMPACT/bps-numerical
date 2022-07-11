#!/usr/bin/env python3

from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import numpy as np
from sklearn.base import BaseEstimator

from ..misc.maths import min_max_normalization
from .classifiers import (
    AbstractPhenotypeClassifier,
    MultiPhenotypeIsolatedClassifier,
    SinglePhenotypeClassifier,
)


class FeatureScorer(ABC):
    @abstractmethod
    def get_features(
        self,
        top_k: int = 100,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError()

    @staticmethod
    def get_top_k_features(
        model: Type[BaseEstimator],
        top_k: int = 25,
        ignore_zeros: bool = False,
        normalize: bool = False,
    ) -> List[Tuple[str, float]]:
        cols = list(model.feature_names_in_)
        importances = model.feature_importances_
        if normalize:
            importances = min_max_normalization(importances)
        indices = np.argsort(-importances)[:top_k]
        cols = np.array(cols)[indices].tolist()
        features = zip(cols, importances[indices].tolist())
        if ignore_zeros:
            features = filter(lambda f: f[1] > 0, features)
        return list(features)


class PhenotypeFeatureScorer(FeatureScorer):
    def __init__(
        self, *classifiers: Tuple[Union[Type[AbstractPhenotypeClassifier], Type[BaseEstimator]]]
    ):
        clfs = []
        for clf in classifiers:
            if isinstance(clf, BaseEstimator):
                clf = [clf]
            elif isinstance(clf, SinglePhenotypeClassifier):
                clf = [clf.model]
            elif isinstance(clf, MultiPhenotypeIsolatedClassifier):
                clf = list(map(lambda m: m.model, clf.classifiers))
            clfs.extend(clf)

        self.models = clfs

    def get_features(
        self,
        top_k: int = 100,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        ignore_zeros = kwargs.get("ignore_zeros", False)
        normalize = kwargs.get("normalize", False)
        if len(self.models) == 1:
            return self.get_top_k_features(
                self.models[0], top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
            )
        _get_top_k = partial(
            self.get_top_k_features, top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
        )
        features = list(map(lambda model: _get_top_k(model=model), self.models))
        return self._get_common_features(*features)

    def _get_common_features(
        self, *features_list: List[List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """
        Get common features across multiple feature list.

        Score is computed as an average between all the common feature scores.
        """
        if len(features_list) == 1:
            return features_list[0]

        _mapify = lambda fts: dict(fts)

        features_first = _mapify(features_list[0])
        common = set(features_first.keys())
        scoremap = features_first.copy()

        for features in features_list[1:]:
            fmap = _mapify(features)
            common = common.intersection(fmap.keys())
            scoremap = dict(map(lambda c: (c, (scoremap[c] + fmap[c])), common))

        scoremap = {name: score / len(features_list) for name, score in scoremap.items()}
        return sorted(list(scoremap.items()), key=lambda x: x[1], reverse=True)


def main():
    pass


if __name__ == "__main__":
    main()
