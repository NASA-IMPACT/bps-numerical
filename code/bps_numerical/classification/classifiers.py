#!/usr/bin/env python3

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import xgboost
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class AbstractPhenotypeClassifier(ABC):
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

        metrics["confusion_matrix"] = confusion_matrix(
            np.asarray(Y_test).argmax(axis=1), np.asarray(preds).argmax(axis=1)
        )
        metrics["classification_report"] = classification_report(
            Y_test,
            preds,
            target_names=kwargs.get("labels"),
            output_dict=True,
        )
        return metrics


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
    ) -> None:
        super().__init__(cols_genes, debug)
        self.phenotype = phenotype
        self.model = model or xgboost.XGBClassifier()

    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
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
        data = pd.get_dummies(data)

        labels = list(set(data.columns) - set(self.cols_genes) - set([self.phenotype]))

        if self.debug:
            logger.debug(f"Target phenotype stats:: {target_counts}")
            logger.debug(f"n genes = {len(self.cols_genes)} || Labels -> {labels}")

        X_train, X_test, Y_train, Y_test = train_test_split(
            data[self.cols_genes],
            data[labels],
            test_size=test_size,
        )
        tracker = self.fit(self.model, X_train, Y_train, X_test, Y_test, labels=labels)
        tracker["labels"] = labels
        return tracker


class MultiPhenotypeIsolatedClassifier:
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

    def train(self, data: pd.DataFrame, test_size: float = 0.2):
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

        X_train, X_test, Y_train_full, Y_test_full = train_test_split(
            data_x, data_y_full, test_size=test_size
        )

        tracker = {}
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


def main():
    pass


if __name__ == "__main__":
    main()
