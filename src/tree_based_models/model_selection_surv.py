import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

from .initialise_model import get_model
from .evaluate import evaluate_model_surv


def model_selection_using_kfold_surv(
    data: pd.DataFrame,
    features: list[str],
    model_cls=None,
    params=None,
    target: str = "OS_YEARS",
    status: str = "OS_STATUS",
    feat_engineering=None,
    unique_id: str = "ROW_ID",
    plot_ft_importance: bool = False,
    n_splits: int = 8,
    log: bool = False,
    log_note: str = None,
):
    """
    Perform K-Fold cross-validation for survival model selection.
    Uses `target` (time) and `OS_STATUS` (event) as survival targets.

    Compatible with scikit-survival models such as RandomSurvivalForest.
    """

    unique_vals = data[unique_id].unique()
    metrics = {"cindex_train": [], "cindex_test": []}
    models = []

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]

        train_mask = data[unique_id].isin(train_vals)
        test_mask = data[unique_id].isin(test_vals)

        data_train = data.loc[train_mask].copy()
        data_test = data.loc[test_mask].copy()

        # Optional feature engineering
        if feat_engineering:
            data_train = feat_engineering(data_train)
            data_test = feat_engineering(data_test)

        # Separate features
        X_train = data_train[features]
        X_test = data_test[features]

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale numeric features
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        y_train = Surv.from_dataframe(status, target, data_train)
        y_test = Surv.from_dataframe(status, target, data_test)

        model = model_cls(**params)

        model.fit(X_train, y_train)

        model_eval = evaluate_model_surv(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        models.append(model)

        cindex_train = model_eval["cindex_train"]
        cindex_test = model_eval["cindex_test"]

        metrics["cindex_train"].append(cindex_train)
        metrics["cindex_test"].append(cindex_test)

        print(
            f"Fold {i+1} - Concordance Index (Train: {cindex_train:.4f} | Test: {cindex_test:.4f})"
        )

    # --- Aggregate results ---
    cindexes = np.array(metrics["cindex_test"])
    mean_c, std_c, min_c, max_c = (
        cindexes.mean(),
        cindexes.std(),
        cindexes.min(),
        cindexes.max(),
    )

    print(
        f"Concordance Index: {mean_c:.4f} (± {std_c:.4f}) "
        f"[Min: {min_c:.4f} ; Max: {max_c:.4f}]"
    )

    # --- Optional logging ---
    if log:
        logfile = "predictions/model_selection.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - Model Selection ({model_type}): "
                f"Mean C-index: {mean_c:.4f} | Std: {std_c:.4f} | "
                f"Min: {min_c:.4f} | Max: {max_c:.4f}{note_str}\n"
            )

    # --- Optional feature importance ---
    if plot_ft_importance:
        try:
            plot_feature_importance(models, features)
        except Exception:
            print("Feature importance not available for this model.")


def get_data(ids, unique_vals, col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series):
    """Extract subset of X and y given selected indices of unique values."""
    selected = unique_vals[ids]
    mask = col.isin(selected)
    return X_data.loc[mask], y_data.loc[mask], mask


def plot_feature_importance(models, features):
    """Plot mean feature importance across trained models."""
    feature_importances = pd.DataFrame(
        [
            model.feature_importances_
            for model in models
            if hasattr(model, "feature_importances_")
        ],
        columns=features,
    )

    if feature_importances.empty:
        print("No feature importance available for these models.")
        return

    mean_importance = feature_importances.mean().sort_values(ascending=False)
    top_features = mean_importance.head(10).index.tolist()

    print("\nTop 10 important features:")
    print(top_features)

    sns.barplot(
        data=feature_importances,
        orient="h",
        order=mean_importance.index,
    )
    return True
