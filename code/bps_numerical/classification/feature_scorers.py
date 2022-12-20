#!/usr/bin/env python3

import functools
import itertools
import operator
from abc import ABC, abstractmethod
from functools import partial, reduce
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..misc.datatools import LoadSaveMixin
from ..misc.maths import min_max_normalization, shuffle_copy
from .classifiers import (
    AbstractPhenotypeClassifier,
    BulkTrainer,
    MultiPhenotypeIsolatedClassifier,
    SinglePhenotypeClassifier,
)
from .tuner import BayesTuner


class FeatureScorer(LoadSaveMixin, ABC):
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
        columns: Optional[List[str]] = None,
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
            `columns`: `List[str]`
                If provided, should be list of input feature names.
                Else, this list will be extracted from the input model.
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
        cols = columns or list(model.feature_names_in_)
        importances = model.feature_importances_
        if normalize:
            importances = min_max_normalization(importances)
        indices = np.argsort(-importances)[:top_k]
        cols = np.array(cols)[indices].tolist()
        features = zip(cols, importances[indices].tolist())
        if ignore_zeros:
            features = filter(lambda f: f[1] > 0, features)
        return list(features)

    @staticmethod
    def _get_common_features(
        *features_list: List[List[Tuple[str, float]]]
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
        clfs = tuple(map(lambda clf: reduce(operator.concat, clf.classifiers), clfs))
        clfs = reduce(operator.concat, clfs) if clfs else clfs
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


class ShapBasedPhenotypeFeatureScorer(PhenotypeFeatureScorer):
    """
    This FeatureScorer uses SHAPely values to get the top_k features

    Args:
        `*classifiers`: Union[
                BulkTrainer,
                Type[BaseEstimator],
                SinglePhenotypeClassifier,
                MultiPhenotypeIsolatedClassifier
            ]
            Vardiac arguments for N number of `BulkTrainer` instances
            to unravel

        `explainer`: ```Type[shap.explainer._explainer.Explainer]````
            What type of explainer to use? Defaults to `TreeExplainer`

        `debug`: `bool`
            If enabled, some debug mode logs will be printed

    Note:
        Unlike xgboost-based feature scorer, this one needs input data
        to figure out the interaction between input/output.
        So, an extra parameter to `get_top_k_features` is `data: pd.DataFrame`
    """

    def __init__(
        self,
        *classifiers: Union[Type[AbstractPhenotypeClassifier], Type[BaseEstimator]],
        explainer: Optional[Type[shap.explainers._explainer.Explainer]] = None,
        debug: bool = False,
    ):
        super().__init__(*classifiers, debug=debug)
        self.explainer = explainer or shap.TreeExplainer

    @staticmethod
    def get_top_k_features(
        model: Type[BaseEstimator],
        explainer: Type[shap.explainers._explainer.Explainer],
        data,
        top_k: int = 500,
        normalize: bool = True,
        ignore_zeros: bool = True,
        columns: List[str] = None,
    ):
        """
        Get top-k important features in descending order
        (first element as the most important feature)

        Args:
            `model`: ```Type[BaseEstimator]```
                Models derived from sklearn's BaseEstimator class
                (eg: `xgboost.XGBClassifier`)
            `explainer`: ```Type[shap.Explainer]```
                Which shap explainer to use?
            `data`: ```pd.DataFrame```
                Input data (X) to generate shapely values
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
        cols = columns or list(model.feature_names_in_)
        data = data[cols]
        c_idxs = []
        for column in cols:
            c_idxs.append(
                data.columns.get_loc(column)
            )  # Get column locations for desired columns in given dataframe

        shap_values = explainer(model).shap_values(data)
        if isinstance(
            shap_values, list
        ):  # If shap values is a list of arrays (i.e., several classes)
            means = [
                np.abs(shap_values[class_][:, c_idxs]).mean(axis=0)
                for class_ in range(len(shap_values))
            ]  # Compute mean shap values per class
            shap_means = np.sum(np.column_stack(means), 1)  # Sum of shap values over all classes
        else:  # Else there is only one 2D array of shap values
            assert len(shap_values.shape) == 2, 'Expected two-dimensional shap values array.'
            shap_means = np.abs(shap_values).mean(axis=0)

        if normalize:
            shap_means = min_max_normalization(shap_means)
        rankings = zip(cols, shap_means)
        if ignore_zeros:
            rankings = filter(lambda x: x[1] > 0, rankings)
        return sorted(rankings, key=lambda x: x[1], reverse=True)[:top_k]

    def get_features(
        self,
        data: pd.DataFrame,
        top_k: int = 500,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        This is the entrypoint to compute feature importance
        Args:
            `data`: ```pd.DataFrame```
                Input data (X) from which shapely values are to be generated
                for each model.
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
        ignore_zeros = kwargs.get("ignore_zeros", True)
        normalize = kwargs.get("normalize", True)
        if len(self.models) == 1:
            return self.get_top_k_features(
                model=self.models[0],
                explainer=self.explainer,
                data=data,
                top_k=top_k,
                ignore_zeros=ignore_zeros,
                normalize=normalize,
            )
        _get_top_k = partial(
            self.get_top_k_features,
            data=data,
            explainer=self.explainer,
            top_k=top_k,
            ignore_zeros=ignore_zeros,
            normalize=normalize,
        )
        features = list(map(lambda model: _get_top_k(model=model), self.models))
        return self._get_common_features(*features)


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
        return sorted(res, key=lambda x: x[1], reverse=True)


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

        `shuffle_columns`: `bool`
            If enabled, different models will see shuffled version of column order

        `xgboost_params`:
            Extra parameters that is passed to each `xgboost` model to provide
            fine-grained path
    """

    def __init__(
        self,
        cols_genes: List[str],
        phenotype: str,
        params_finder: Optional[BayesTuner] = None,
        n_runs: int = 3,
        debug: bool = False,
        shuffle_columns: bool = False,
        **xgboost_params,
    ) -> None:

        super().__init__(debug=debug)
        self.shuffle_columns = bool(shuffle_columns)
        self.cols_genes = list(cols_genes)
        self.phenotype = phenotype
        self.n_runs = int(n_runs)

        # first run is original order
        # later runs are shuffled if enabled
        self.classifiers = []
        self.results = []
        self.phenotype = phenotype
        self.features_ = []
        self.params_finder = params_finder
        self.xgboost_params = xgboost_params

    def build_classifiers(self, **xgboost_params) -> List[SinglePhenotypeClassifier]:
        return [
            SinglePhenotypeClassifier(
                cols_genes=self.cols_genes,
                phenotype=self.phenotype,
                debug=self.debug,
                **xgboost_params,
            )
        ] + [
            SinglePhenotypeClassifier(
                cols_genes=self.cols_genes
                if not self.shuffle_columns
                else shuffle_copy(self.cols_genes),
                phenotype=self.phenotype,
                debug=self.debug,
                **xgboost_params,
            )
            for _ in range(self.n_runs - 1)
        ]

    def get_features(
        self, data: pd.DataFrame, test_size: float = 0.2, top_k: int = 500, **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Get list of important features using all the training runs

        Args:
            `data`: `pd.DataFrame`
                Input dataframe with genes and targets
            `test_size`: `float`
                How much portion of data is used for splitting to test?
            `top_k`: `int`
                How many number of top features to return from each model?
        """
        if self.features_:
            return self.features_

        xgboost_params = self.xgboost_params

        # find params and rebuilt classifiers
        if self.params_finder is not None:
            logger.info("Finding best params...")
            res = self.params_finder.search(data)
            xgboost_params.update(dict(self.params_finder.best_params))
            logger.debug(f"Best params = {xgboost_params}")
            logger.debug(f"Best score at {self.params_finder.best_score}")

        logger.debug(f"xgboost_ params in use = {xgboost_params}")
        self.classifiers = self.build_classifiers(**xgboost_params)

        self.results = [clf.train(data, test_size) for clf in self.classifiers]

        ignore_zeros = kwargs.get("ignore_zeros", True)
        normalize = kwargs.get("normalize", True)
        top_k = int(top_k)

        if self.debug:
            pprint(self.results)
            self._debug_plot_hist(**kwargs)

        self.features_ = PhenotypeFeatureScorer(*self.classifiers).get_features(
            top_k=top_k, ignore_zeros=ignore_zeros, normalize=normalize
        )
        return self.features_

    def _debug_plot_hist(self, **kwargs):
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
        except ModuleNotFoundError:
            logger.warning("Cannot plot histogram. plotly might not be installed.")


class MeanReciprocalRanker(FeatureScorer):
    """
    This ranker use MRR algorithm inspired from document query system but for feature relevance.
    (See: https://en.wikipedia.org/wiki/Mean_reciprocal_rank)

    Once all the N models are trained (using `GeneRanker`), use these models to compute
    mean-reciprocal ranks for all the gene features aggregated.

    Args:
        `*classifiers`: SinglePhenotypeClassifier
                Vardiac arguments for N number of `SinglePhenotypeClassifier` instances
                to unravel
        `score_cutoff`: `float`
            Cut-off threshold for feature score we get from each model (from `classifiers`).
            Defaults to 0.05
        `rank_cutoff`: `float`
            Cut-off threshold for final ranks. `>=rank_cutoff` genes will only be returned
        `debug`: `bool`
            Flag to denote debug mode
    """

    def __init__(
        self, *classifiers, score_cutoff: float = 0.05, rank_cutoff: float = 0, debug: bool = False
    ) -> None:
        super().__init__(debug=debug)
        self.classifiers = classifiers
        self.score_cutoff = score_cutoff
        self.rank_cutoff = rank_cutoff
        self.rankmap = {}

    def get_features(
        self,
        top_k: int = 500,
        normalize: bool = True,
        ignore_zeros: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Compute list of ranked features using  the trained models
        based on  MRR algorithm.

        Args:
            `top_k`: `int`
                How many number of top features to return from each model?
            `normalize`: `bool`
                Normalize ranks in the [0, 1] range or not?
            `ignore_zeros`: `bool`
                Ignore zero-value features?
        """

        if self.debug:
            logger.debug(f"{len(self.classifiers)} in use.")
        if not self.classifiers:
            return []

        features = map(
            lambda clf: list(
                filter(
                    lambda f: f[1] >= self.score_cutoff,
                    PhenotypeFeatureScorer(clf).get_features(
                        top_k=top_k, ignore_zeros=ignore_zeros, normalize=True
                    ),
                )
            ),
            self.classifiers,
        )
        features = list(features)
        _features_flattened = functools.reduce(operator.concat, features)
        _unified = set(map(lambda x: x[0], _features_flattened))
        self.rankmap = {gene: self._compute_ranks(gene, features) for gene in tqdm(_unified)}

        res = [(gene, self._compute_mrr(gene, ranks)) for gene, ranks in self.rankmap.items()]
        if normalize:
            fts, mrrs = zip(*res)
            res = zip(fts, min_max_normalization(mrrs))
        res = filter(lambda x: x[1] > self.rank_cutoff, res)
        return sorted(res, key=lambda x: x[1], reverse=True)

    def _compute_mrr(self, feature, ranks) -> float:
        return sum(map(lambda x: 1 / x if x > 0 else 0, ranks)) / len(ranks)

    def _compute_ranks(self, feature, features_list) -> Dict[str, List[int]]:
        res = []
        for _fts in features_list:
            _fts, _ = zip(*_fts)
            rank = 0
            try:
                rank = _fts.index(feature) + 1
            except ValueError:
                rank = 0
            res.append(rank)
        return res


def main():
    pass


if __name__ == "__main__":
    main()
