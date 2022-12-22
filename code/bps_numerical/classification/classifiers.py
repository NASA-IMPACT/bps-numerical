#!/usr/bin/env python3

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import xgboost
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tqdm import tqdm

from ..misc.datatools import LoadSaveMixin
from ..misc.datatools import (
    train_test_indexed_split_stratified as train_test_indexed_split,
)


class AbstractPhenotypeClassifier(LoadSaveMixin, ABC):
    """
    This represents the component type for training the ML models
    """

    def __init__(self, cols_genes: List[str], debug: bool = False) -> None:
        self.cols_genes = list(cols_genes)
        self.debug = debug

    @abstractmethod
    def train(self, data: pd.DataFrame, test_size: float = 0.2, **kwargs) -> dict:
        """
        This method is used for performing overall training using the
        incoming pre-processed dataframe.

        Args:
            `data`: ```pd.DataFrame```
                Incoming pre-processed dataframe to use for training
            `test_size`: ```float```
                What's the size of test dataset?

        Returns:
            A dictionary whose structure depends on the implementation of
            child components.
        """
        raise NotImplementedError()

    def fit(
        self,
        model: Type[BaseEstimator],
        X_train,
        Y_train,
        X_test,
        Y_test,
        **kwargs,
    ) -> Dict[str, float]:
        """
        This is an internal method used for fitting the incoming model.

        Args:
            `model`: ```Type[BaseEstimator]```
                sklearn model which needs to be fitted
            `X_train`: ```np.ndarray```
                Input samples used for training
            `Y_train`: ```np.ndarray```
                Target samples used for training
            `X_test`: ```np.ndarray```
                Input samples used for test
            `Y_test`: ```np.ndarray```
                Target samples used for test

        """
        start = time.time()

        if self.debug:
            logger.debug(
                f"X_train={X_train.shape} | Y_train={Y_train.shape} | X_test={X_test.shape} | Y_test={Y_test.shape}"
            )
            logger.debug(f"Model = {model}")

        model.fit(X_train, Y_train)
        metrics = {
            "train_score": model.score(X_train, Y_train),
            "test_score": model.score(X_test, Y_test),
        }

        end = time.time()
        logger.debug(f"Training took {end-start} seconds.")

        preds = model.predict(X_test)
        preds = np.asarray(preds).argmax(axis=1) if preds.ndim > 1 else preds

        Y_test = np.asarray(Y_test).argmax(axis=1) if Y_test.ndim > 1 else Y_test
        metrics["confusion_matrix"] = confusion_matrix(Y_test, preds)
        metrics["classification_report"] = classification_report(
            Y_test,
            preds,
            target_names=kwargs.get("labels"),
            output_dict=True,
        )
        return metrics

    @property
    def __classname__(self) -> str:
        return self.__class__.__name__


class SinglePhenotypeClassifier(AbstractPhenotypeClassifier):
    """
    This classifer is used for training a single model for
    a single specific phenotype.

    Note:
        We use `MultiPhenotypeIsolatedClassifier` to perform N-different
        classification independently, each for separate phenotype.
        Then, we use `bps_numerical.classification.feature_scorers.PhenotypeFetureScorer`
        for computing common genes between different phenotypes.

    Args:
        `cols_genes`: ```List[str]```
            Input gene/feature names to be used for classification
        `phenotype`: ```str```
            Name of target phenotype to be used for classification
        `model`: ```Optional[BaseEstimator]```
            Model to be used for classification.
            If not provided, defaults to default `xgboost.XGBClassifier`
        `debug`: ```bool```
            If enabled, debugging logs will be printed
    """

    def __init__(
        self,
        cols_genes: List[str],
        phenotype: str,
        model: Optional[Type[BaseEstimator]] = None,
        debug: bool = False,
        target_encoder: Optional[Union[LabelEncoder, LabelBinarizer]] = None,
        **xgboost_params,
    ) -> None:
        super().__init__(cols_genes, debug)
        self.phenotype = phenotype
        self.model = model or xgboost.XGBClassifier(**xgboost_params)
        self.target_encoder = target_encoder or LabelEncoder()

    def train(self, data: pd.DataFrame, test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        This method is used for performing overall training using the
        incoming pre-processed dataframe.

        Args:
            `data`: ```pd.DataFrame```
                Incoming pre-processed dataframe to use for training
            `test_size`: ```float```
                What's the size of test dataset?

        Returns:
            A dictionary of the form:

            Sample output dict:

                .. code-block: python
                    {
                        'classification_report': {
                            'animalreturn_isst': {'f1-score': 1.0, 'precision': 1.0, 'recall': 1.0, 'support': 973},
                            'animalreturn_lar': {'f1-score': 1.0, 'precision': 1.0, 'recall': 1.0, 'support': 394},
                            'macro avg': {'f1-score': 1.0, 'precision': 1.0, 'recall': 1.0, 'support': 1367},
                            'micro avg': {'f1-score': 1.0, 'precision': 1.0, 'recall': 1.0, 'support': 1367},
                            'samples avg': {'f1-score': 1.0, 'precision': 1.0, 'recall': 1.0, 'support': 1367},
                            'weighted avg': {'f1-score': 1.0, 'precision': 1.0, 'recall': 1.0, 'support': 1367},
                        },
                        'confusion_matrix': array([[973, 0], [0, 394]]),
                        'labels': ['animalreturn_isst', 'animalreturn_lar'],
                        'test_score': 1.0,
                        'train_score': 1.0,
                    }
        """
        target_counts = data[self.phenotype].value_counts()
        data = data[self.cols_genes + [self.phenotype]]

        # labels = sorted(list(set(data.columns) - set(self.cols_genes) - set([self.phenotype])))
        Y = self.target_encoder.fit_transform(data[self.phenotype])

        labels = self.target_encoder.classes_

        if self.debug:
            logger.debug(f"Target phenotype stats:: {target_counts}")
            logger.debug(f"n genes = {len(self.cols_genes)} || Labels -> {labels}")

        splitted_data = train_test_indexed_split(
            data[self.cols_genes],
            # data[labels],
            Y,
            test_size=test_size,
            shuffle=kwargs.get("shuffle", True),
        )
        X_train, X_test, Y_train, Y_test = splitted_data.pop("data")
        tracker = self.fit(self.model, X_train, Y_train, X_test, Y_test, labels=labels)
        tracker["labels"] = labels
        tracker["indices"] = splitted_data.pop("indices")
        return tracker


class MultiPhenotypeIsolatedClassifier(AbstractPhenotypeClassifier):
    """
    Trains N different independent multi-class model.
    Same train/test data is used for each of the model.

    Args:
        `cols_genes`: ```List[str]```
            Input gene/feature names to be used for classification
        `phenotype`: ```str```
            Name of target phenotype to be used for classification
        `model`: ```Optional[BaseEstimator]```
            Model to be used for classification.
            If not provided, defaults to default `xgboost.XGBClassifier`
        `debug`: ```bool```
            If enabled, debugging logs will be printed
    """

    def __init__(
        self,
        cols_genes: List[str],
        classifiers: List[SinglePhenotypeClassifier],
        debug: bool = False,
    ) -> None:
        self.cols_genes = list(cols_genes)
        self.classifiers = classifiers
        self.debug = debug

    def train(self, data: pd.DataFrame, test_size: float = 0.2, **kwargs):
        """
        This method is used for performing overall training using the
        incoming pre-processed dataframe.

        Args:
            `data`: ```pd.DataFrame```
                Incoming pre-processed dataframe to use for training
            `test_size`: ```float```
                What's the size of test dataset?

        Returns:
            A dictionary whose key is the name of phenotype class and
            value is another `dict` of the form returned by any `SinglePhenotypeClassifier.train`

        Sample output dict:
            .. code-block: python
                {
                    "animalreturn": {
                        "classification_report": {
                            "animalreturn_ISST": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 973,
                            },
                            "animalreturn_LAR": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 394,
                            },
                            "macro avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                            "micro avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                            "samples avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                            "weighted avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                        },
                        "confusion_matrix": array([[973, 0], [0, 394]]),
                        "labels": ["animalreturn_ISST", "animalreturn_LAR"],
                        "test_score": 1.0,
                        "train_score": 1.0,
                    },
                    "condition": {
                        "classification_report": {
                            "condition_FLT": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 710,
                            },
                            "condition_GC": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 657,
                            },
                            "macro avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                            "micro avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                            "samples avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                            "weighted avg": {
                                "f1-score": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "support": 1367,
                            },
                        },
                        "confusion_matrix": array([[710, 0], [0, 657]]),
                        "labels": ["condition_FLT", "condition_GC"],
                        "test_score": 1.0,
                        "train_score": 1.0,
                    },
                }

        """

        target_columns_full = list(set(data.columns) - set(self.cols_genes))
        data_x = data[self.cols_genes]
        data_y_full = data[target_columns_full]

        splitted_data = train_test_indexed_split(
            data_x,
            data_y_full,
            test_size=test_size,
            shuffle=kwargs.get("shuffle", True),
        )
        X_train, X_test, Y_train_full, Y_test_full = splitted_data.pop("data")

        tracker = dict(indices=splitted_data.pop("indices"))
        for clf in tqdm(self.classifiers):
            logger.info(f"Training for phenotype={clf.phenotype}")
            tracker[clf.phenotype] = {}
            Y_train = pd.get_dummies(Y_train_full[[clf.phenotype]])
            Y_test = pd.get_dummies(Y_test_full[[clf.phenotype]])

            labels = list(Y_train.columns)
            tracker[clf.phenotype]["labels"] = labels

            if self.debug:
                logger.debug(
                    f"Target phenotype stats:: {Y_train_full[clf.phenotype].value_counts()}"
                )
                logger.debug(f"n genes = {len(self.cols_genes)} || Labels -> {labels}")
            metrics = clf.fit(clf.model, X_train, Y_train, X_test, Y_test, labels=labels)
            tracker[clf.phenotype].update(metrics)

        return tracker


class BulkTrainer(AbstractPhenotypeClassifier):
    """
    A wrapper around `SinglePhenotypeClassifier` and `MultiPhenotypeIsolatedClassifier`
    to train them N different times.
    That is: it performs training N times separately, independently.

    Args:
        `cols_genes`: ```List[str]```
            Input gene/feature names to be used for classification
        `classifiers`: ```List[Union[SinglePhenotypeClassifier, MultiPhenotypeIsolatedClassifier]]```
            Classifiers to be used for classification.
        `debug`: ```bool```
            If enabled, debugging logs will be printed
    """

    def __init__(
        self,
        cols_genes: List[str],
        classifiers: List[SinglePhenotypeClassifier],
        n_runs: int = 3,
        debug: bool = False,
    ) -> None:
        self.cols_genes = list(cols_genes)
        self.n_runs = n_runs
        self.debug = debug
        self.results = []

        # each input classifier is copied for each run
        # to avoid re-using same model across different runs
        self.classifiers = [copy.deepcopy(classifiers) for _ in range(n_runs)]

    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> List[List[dict]]:
        """
        This method is used for training the input classifiers N different times

        Args:
            `data`: ```pd.DataFrame```
                Incoming pre-processed dataframe to use for training
            `test_size`: ```float```
                What's the size of test dataset?

        Note:
            Every time this method is run, the `self.results`
            will be overridden by the new results.
        """
        results = []
        for run in tqdm(range(self.n_runs)):
            res = []
            for clf in self.classifiers[run]:
                res.append(clf.train(data, test_size))
            results.append(res)

        # replace the old cache
        self.results = results
        return results


def main():
    pass


if __name__ == "__main__":
    main()
