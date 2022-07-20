#!/usr/bin/env python3

import itertools
import operator
from abc import ABC, abstractmethod
from functools import partial, reduce
from pprint import pprint
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator

from ..misc.maths import min_max_normalization
from .classifiers import (
    AbstractPhenotypeClassifier,
    BulkTrainer,
    MultiPhenotypeIsolatedClassifier,
    SinglePhenotypeClassifier,
)


class FeatureScorer(ABC):
    """
    This represents a type of FeatureScorer.
    Any downstream children should implement `get_features` method.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)

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

    Args:
        `*classifiers`: Union[
                BulkTrainer,
                Type[BaseEstimator],
                SinglePhenotypeClassifier,
                MultiPhenotypeIsolatedClassifier
            ]
            Vardiac arguments for N number of `BulkTrainer` instances
            to unravel

        `debug`: `bool`
            If enabled, some debug mode logs will be printed
    """

    def __init__(
        self,
        *classifiers: Union[Type[AbstractPhenotypeClassifier], Type[BaseEstimator]],
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.classifiers = classifiers
        classifiers = self._unravel_bulk_trainer(*classifiers) + self._unravel_others(*classifiers)
        self.models = self._unravel_others(*classifiers)

    def _unravel_bulk_trainer(
        self, *classifiers: Tuple[BulkTrainer]
    ) -> Tuple[Type[AbstractPhenotypeClassifier]]:
        """
        This unravel/falttens all the classifiers within `classifiers.BulkTrainer`
        into a tuple of sub-classifiers

        Args:
            `*classifiers`: `BulkTrainer`
                Vardiac arguments for N number of `BulkTrainer` instances
                to unravel
        Returns:
            `Tuple[Union[SinglePhenotypeClassifier, MultiPhenotypeIsolatedClassifier]]`
            Note:
                If the incoming classifier arguments doesn't exist, it will return
                empty tuple
        """
        clfs = tuple(
            filter(
                lambda clf: isinstance(clf, BulkTrainer),
                classifiers,
            )
        )
        if not clfs:
            return tuple()
        clfs = map(lambda clf: reduce(operator.concat, clf.classifiers), clfs)
        clfs = reduce(operator.concat, clfs)
        return tuple(clfs)

    def _unravel_others(
        self, *classifiers: Union[Type[AbstractPhenotypeClassifier], Type[BaseEstimator]]
    ) -> tuple:
        """
        This unravel/falttens all the classifiers into a tuple of sub-classifiers

        Args:
            `*classifiers`: `Union[Type[AbstractPhenotypeClassifier], Type[BaseEstimator]]`
                Vardiac arguments for N number of classifier instances (not BulkTrainer)
                to unravel
        Returns:
            `Tuple[Type[BaseEstimator]]`
        """
        clfs = []
        for clf in classifiers:
            if isinstance(clf, BaseEstimator):
                clf = [clf]
            elif isinstance(clf, SinglePhenotypeClassifier):
                clf = [clf.model]
            elif isinstance(clf, MultiPhenotypeIsolatedClassifier):
                clf = list(map(lambda m: m.model, clf.classifiers))
            else:
                continue
            clfs.extend(clf)
        return tuple(clfs)

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

        _mapify = lambda fts: dict(fts)  # noqa: E731

        features_first = _mapify(features_list[0])
        common = set(features_first.keys())
        scoremap = features_first.copy()

        for features in features_list[1:]:
            fmap = _mapify(features)
            common = common.intersection(fmap.keys())
            scoremap = dict(map(lambda c: (c, (scoremap[c] + fmap[c])), common))

        scoremap = {name: score / len(features_list) for name, score in scoremap.items()}
        return sorted(list(scoremap.items()), key=lambda x: x[1], reverse=True)


class GeneRanker(FeatureScorer):
    """
    This is used for performing feature ranking/scoring for a single genotype
    by fitting in independent models and getting the common features.

    Args:
        `cols_genes`: `List[str]`
            Input list of columns/genes to be considered

        `phenotype`: `str`
            Which phenotype to use?

        `n_runs`: `int`
            How many times to run the training separately?
            (Each training run will be independent)

        `debug`: `bool`
            If enabled, some debug mode logs/diagrams will be rendered
    """

    def __init__(
        self, cols_genes: List[str], phenotype: str, n_runs: int = 3, debug: bool = False
    ) -> None:

        super().__init__(debug=debug)
        self.classifiers = [
            SinglePhenotypeClassifier(cols_genes=cols_genes, phenotype=phenotype, debug=debug)
            for _ in range(n_runs)
        ]
        self.results = []
        self.phenotype = phenotype

    def get_features(
        self, data: pd.DataFrame, test_size: float = 0.2, **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Get list of important features using all the training runs

        Args:
            `data`: `pd.DataFrame`
                Input dataframe with genes and targets
            `test_size`: `float`
                How much portion of data is used for splitting to test?
        """
        self.results = [clf.train(data, test_size) for clf in self.classifiers]

        ignore_zeros = kwargs.get("ignore_zeros", True)
        normalize = kwargs.get("normalize", True)
        top_k = kwargs.get("top_k", 500)

        if self.debug:
            pprint(self.results)
            self._debug_plot_hist(**kwargs)

        return PhenotypeFeatureScorer(*self.classifiers).get_features(
            top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
        )

    def _debug_plot_hist(self, **kwargs):
        if not self.debug:
            return
        ignore_zeros = kwargs.get("ignore_zeros", True)
        normalize = kwargs.get("normalize", True)
        top_k = kwargs.get("top_k", 500)
        nfeatures = []
        for clf in self.classifiers:
            features = PhenotypeFeatureScorer(clf).get_features(
                top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
            )
            nfeatures.append(len(features))
            logger.debug(f"{clf.phenotype} | {len(features)}")

        try:
            import plotly.express as px

            _df_counter = pd.DataFrame(enumerate(nfeatures), columns=["run", "n_feature"])
            fig = px.bar(
                _df_counter,
                x="run",
                y="n_feature",
                title=f"run vs n_feature for {self.phenotype}",
                text_auto=True,
            )
            fig.update_layout(yaxis=dict(tickfont=dict(size=kwargs.get("debug_font_size", 7))))
            fig.show()
        except:
            logger.warning("Cannot plot histogram. plotly might not be installed.")


class UnifiedFeatureScorer(PhenotypeFeatureScorer):
    """
    This feature scorer is used to combine (union) all the features
    from given classifiers

    Args:
        `*classifiers`: Union[
                BulkTrainer,
                Type[BaseEstimator],
                SinglePhenotypeClassifier,
                MultiPhenotypeIsolatedClassifier
            ]
            Vardiac arguments for N number of `BulkTrainer` instances
            to unravel

        `top_k`: `int`
            How many top features to consider?

        `at_model_level`: `bool`
            If enabled, all the instances of `Type[AbstractPhenotypeClassifier]`
            will be flattened to `Type[BaseEstimator]` (eg: XGBClassifier).
            This gives a list of BaseEstimator for which feature importance
            is computed independently and then combined. Else, we go with usual Classifier-level computation.

            Note: This might give us
            a larger feature set as we are performing union operation
            across all the sklearn models.
    """

    def get_features(
        self,
        top_k: int = 100,
        at_model_level: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:

        ignore_zeros = kwargs.get("ignore_zeros", True)
        normalize = kwargs.get("normalize", True)

        # only unravel bulk trainer and then add other classifers without change
        clfs = (
            self.models
            if at_model_level
            else self._unravel_bulk_trainer(*self.classifiers)
            + tuple(filter(lambda clf: not isinstance(clf, BulkTrainer), self.classifiers))
        )
        if not clfs:
            logger.warning("No classifiers detected. Returning empty list!")
            return []
        features = map(
            lambda clf: PhenotypeFeatureScorer(clf).get_features(
                top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
            ),
            clfs,
        )
        features = reduce(
            operator.concat,
            features,
        )
        if self.debug:
            logger.debug(f"All features (n={len(features)}) => {features}")

        res = []
        # need to sort to perform itertools.groupby
        # behaviour is just like `uniq` from Unix
        features = sorted(features)
        for key, group in itertools.groupby(features, operator.itemgetter(0)):
            scores = tuple(map(lambda g: g[1], group))
            res.append((key, sum(scores) / len(scores)))
        return res


def main():
    pass


if __name__ == "__main__":
    main()
