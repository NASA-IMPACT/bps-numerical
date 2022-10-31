#!/usr/bin/env python3

from typing import List

import pandas as pd
import xgboost
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real


class BayesTuner:
    def __init__(
        self,
        columns: List[str],
        target_column: str,
        n_iter: int = 10,
        n_jobs: int = 4,
        objective: str = "multi:softmax",
        k_folds: int = 3,
        debug: bool = False,
    ) -> None:
        self.columns = list(columns)
        self.target_column = target_column
        self.debug = bool(debug)
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

    def search(self, data: pd.DataFrame):
        target_encoded = preprocessing.LabelEncoder().fit_transform(data[self.target_column])
        self.result = self.tuner.fit(data[self.columns], target_encoded)
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
