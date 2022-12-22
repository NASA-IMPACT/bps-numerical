#!/usr/bin/env python3

from typing import List, Optional, Type, Union

import pandas as pd
import xgboost
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.callbacks import (
    CheckpointSaver,
    DeadlineStopper,
    DeltaYStopper,
    EarlyStopper,
)
from skopt.space import Integer, Real


class BayesTuner:
    """
    This module is used to find the best parameter for xgboost model.

    Args:
        `columns`: `List[str]`
            List of feature column names to be used
        `target_column`: `str`
            The target column for classification task
        `n_iter`: `int`
            Total number of iteration to perform for search.
            Defaults to `10`
        `n_jobs`: `int`
            Total number of jobs in parallel. Defaults to 4.
        `objective`: `str`
            Objective function used for xgboost training.
            Can be either of:
                - 'multi:softmax' (for multi-class)
                - 'binary:logistic' (for binary classification)
        `k_folds`: `int`
            Total number of folds to use for search
        `callbacks`: `Optional[List[Union[Type[EarlyStopper], CheckpointSaver]]]`
            List of `skopt` callbacks object.
            If this parameter is `None`, no callbacks will be added.
            If this parameter is an empty list (`[]`), defaults will be generated.
        `debug`: `bool`
            Flag for debugging mode

    Properties & Methods:
        - `search`: The entry-point method to start the search
        - `best_params`: `dict` that has the best params once search has
        ended
        - `best_score`: `float` that represents the best model found
    """

    # early stopping params (see _get_callbacks(...))
    _ES_DELTA_Y = 1e-4  # the search doesn't progress by this delta, stop
    _ES_DEADLINE = 60 * 60 * 2  # 2 hours before stopping the tuner

    def __init__(
        self,
        columns: List[str],
        target_column: str,
        n_iter: int = 10,
        n_jobs: int = 4,
        objective: str = "multi:softmax",
        k_folds: int = 3,
        callbacks: Optional[List[Union[Type[EarlyStopper], CheckpointSaver]]] = None,
        debug: bool = False,
    ) -> None:
        self.columns = list(columns)
        self.target_column = target_column
        self.debug = bool(debug)

        self.callbacks = self._get_callbacks(callbacks)

        self.search_spaces = {
            'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            'max_depth': Integer(0, 50),
            'max_delta_step': Integer(0, 20),
            'subsample': Real(0.01, 1.0, 'uniform'),
            'colsample_bytree': Real(0.01, 1.0, 'uniform'),
            # 'colsample_bylevel': Real(0.01, 1.0, 'uniform'),
            'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
            'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
            'gamma': Real(1e-9, 0.5, 'log-uniform'),
            'min_child_weight': Integer(0, 5),
            'n_estimators': Integer(50, 500),
            'scale_pos_weight': Real(1e-6, 500, 'log-uniform'),
        }
        self.tuner = BayesSearchCV(
            estimator=xgboost.XGBClassifier(n_jobs=n_jobs, objective=objective, silent=0),
            search_spaces=self.search_spaces,
            scoring=None,
            cv=StratifiedKFold(n_splits=k_folds, shuffle=True),
            n_jobs=n_jobs,
            n_iter=n_iter,
            verbose=4,
            refit=True,
        )
        self.result = None

    @property
    def __default_callbacks(self) -> List[Type[EarlyStopper]]:
        return [
            DeltaYStopper(delta=self._ES_DELTA_Y),
            DeadlineStopper(total_time=self._ES_DEADLINE),
        ]

    def _get_callbacks(self, callbacks: List[Type[EarlyStopper]]) -> List[Type[EarlyStopper]]:
        # in case we don't want callbacks
        if callbacks is None:
            return []

        # if provided callbacks is empty list, go with defaults
        callbacks = callbacks or self.__default_callbacks
        for callback in callbacks:
            if not isinstance(callback, (EarlyStopper, CheckpointSaver)):
                raise TypeError(
                    f"Invalid type for callback={callback}. Expected Type[EarlyStopper]. Got {type(callback)}"
                )
        return callbacks

    def search(self, data: pd.DataFrame):
        """
        Args:
            `data`: `pd.DataFrame`
                Input dataframe for gene data (contains both gene expresison and metadata columns)

        Returns:
            Tuned object
        """
        target_encoded = preprocessing.LabelEncoder().fit_transform(data[self.target_column])
        self.result = self.tuner.fit(data[self.columns], target_encoded, callback=self.callbacks)
        return self.result

    @property
    def best_params(self) -> dict:
        return self.tuner.best_params_

    @property
    def best_score(self) -> float:
        return self.tuner.best_score_


def main():
    pass


if __name__ == "__main__":
    main()
