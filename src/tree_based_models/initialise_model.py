import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
from .ensemble import EnsembleClassifier, EnsembleRegressor
from sklearn.linear_model import LogisticRegression


def weighted_mse(y_true, y_pred):
    """
    Weighted MSE objective for XGBoost.
    Penalizes errors on small-magnitude targets more heavily.
    """
    weight = 1.0 / (np.abs(y_true) + 1e-6)
    grad = -2 * (y_true - y_pred) * weight
    hess = 2 * weight
    return grad, hess


model_params = {
    # Simple linear models
    "linear": {"fit_intercept": True, "positive": True},
    "ridge": {"alpha": 10.0, "fit_intercept": True, "random_state": 42},
    "ridge1": {"alpha": 1.0, "fit_intercept": True, "random_state": 42},
    "ridge2": {"alpha": 100.0, "fit_intercept": True, "random_state": 42},
    "ridge_benchmark": {"alpha": 1e-2, "fit_intercept": True, "random_state": 42},
    "lasso": {"alpha": 0.1, "fit_intercept": True, "random_state": 42},
    "logreg_params": {
        "C": 1 / 1e-2,
        "fit_intercept": True,
        "random_state": 42,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "logreg_params_2": {
        "C": 1 / 1000,
        "fit_intercept": True,
        "random_state": 42,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    # Tree-based regressors
    "rf": {"n_estimators": 20, "max_depth": 3, "random_state": 42},
    "xgb": {
        "n_estimators": 10,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "xgb_classif": {
        "n_estimators": 10,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "xgb_opt": {
        "learning_rate": 0.25,
        "max_depth": 4,
    },
    "xgb_objective": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": weighted_mse,
    },
    "lgbm": {
        "n_estimators": 50,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "lgbm_classif": {
        "n_estimators": 50,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "lgbm_chat": {
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 64,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    },
    "lgbm_opt": {
        "learning_rate": 0.23,
        "max_depth": 3,
        "verbose": -1,
        "metric": "mse",
    },
    "cat": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 0,
    },
}

model_registry = {
    "logreg_params": LogisticRegression,
    "logreg_params_2": LogisticRegression,
    "linear": LinearRegression,
    "ridge": Ridge,
    "ridge1": Ridge,
    "ridge2": Ridge,
    "ridge_benchmark": Ridge,
    "lasso": Lasso,
    "rf": RandomForestRegressor,
    "cat": CatBoostRegressor,
}


def get_model(model_type: str, custom_params: dict = None):
    """
    Returns a regression model with predefined hyperparameters.

    Parameters
    ----------
    model_type : str
        One of available model keys (see `list_available_models`).
    custom_params : dict, optional
        Custom parameters to override defaults.

    Returns
    -------
    model : Regressor
        Instantiated regression model.
    """
    if model_type == "voting":
        return EnsembleRegressor(
            models=[
                LGBMRegressor(**model_params["lgbm"]),
                XGBRegressor(**model_params["xgb"]),
            ]
        )

    if model_type == "voting_classif":
        return EnsembleClassifier(
            models=[
                LGBMClassifier(**model_params["lgbm_classif"]),
                XGBClassifier(**model_params["xgb_classif"]),
            ],
            weights=[0.2, 0.8],
        )

    if model_type == "voting_ridge":
        return EnsembleRegressor(
            models=[
                Ridge(**model_params["ridge"]),
                Ridge(**model_params["ridge1"]),
                Ridge(**model_params["ridge2"]),
            ],
        )

    if model_type not in model_registry or model_type not in model_params:
        list_models = list(model_registry.keys()) + ["voting"]
        raise ValueError(
            f"Invalid model_type '{model_type}'. " f"Choose from {list_models}"
        )

    params = {**model_params[model_type], **(custom_params or {})}
    return model_registry[model_type](**params)
