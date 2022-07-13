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
    """
    This represents a type of FeatureScorer.
    Any downstream children should implement `get_features` method.
    """

    @abstractmethod
    def get_features(
        self,
        top_k: int = 500,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        This is an abstract method to implement.

        Args:
            `top_k`: ```int```
                Top k most important features to return
            `columns`: ```Optional[List[str]]```
                List of input columns used for mapping the feature names

        Returns:
            List[Tuple[str, float]] where each tuple is of:
                - first element -> feature name
                - second element -> feature score value

            Eg:
                .. code-block: python
                    [
                        ('ENSMUSG00000020889', 1.0),
                        ('ENSMUSG00000115420', 0.25354915857315063)
                    ]
        """
        raise NotImplementedError()

    @staticmethod
    def get_top_k_features(
        model: Type[BaseEstimator],
        top_k: int = 25,
        ignore_zeros: bool = False,
        normalize: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k important features in descending order
        (first element as the most important feature)

        Args:
            `model`: ```Type[BaseEstimator]```
                Models derived from sklearn's BaseEstimator class
                (eg: `xgboost.XGBClassifier`)
            `top_k`: ```int```
                Top k most important features to return
            `ignore_zeros`: ```bool```
                If enabled, all the features which has *zero* (0) scores
                will be removed
            `normalize`: ```bool```
                If enabled, feature scores will be normalized in [0, 1] range
                using min-max normalization.
                (Highest feature will have 1.0 score)
        Returns:
            List[Tuple[str, float]] where each tuple is of:
                - first element -> feature name
                - second element -> feature score value

            Eg:
                .. code-block: python
                    [
                        ('ENSMUSG00000020889', 1.0),
                        ('ENSMUSG00000115420', 0.25354915857315063)
                    ]


        """
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
    """
    This encapsulates "important" feature computation
    for any classifier/models.

    Input can be of any type:
        - `Type[AbstractPhenotypeClassifier]`
        - `Type[BaseEstimator]`

    All the components derived from AbstractPhenotypeClassifier and BaseEstimator
    represent trained model.

    If the type is `SinglePhenotypeClassifier`, then `SinglePhenotypeClassifier.model`
    is used to compute the feature.

    If the type is `MultiPhenotypeIsolatedClassifier`, then `MultiPhenotypeIsolatedClassifier.classifiers`
    is used to compute the feature, where we can access `model` attribute for eahc of
    those clasifiers.
    """

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
        """
        This is the entrypoint to compute feature importance
        Args:
            `top_k`: ```int```
                Top k most important features to return
            `columns`: ```Optional[List[str]]```
                List of input columns used for mapping the feature names

        Returns:
            List[Tuple[str, float]] where each tuple is of:
                - first element -> feature name
                - second element -> feature score value

            Eg:
                .. code-block: python
                    [
                        ('ENSMUSG00000020889', 1.0),
                        ('ENSMUSG00000115420', 0.25354915857315063)
                    ]
        """
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