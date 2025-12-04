import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .initialise_model import get_model
from .evaluate import evaluate_model


def model_selection_using_kfold(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    model_type: str,
    feat_engineering=None,
    unique_id: str = "ROW_ID",
    plot_ft_importance: bool = False,
    n_splits: int = 8,
    log: bool = False,
    log_note: str = None,
    scale: bool = False,
    n_importance: int = 10,
):
    """
    Perform K-Fold cross-validation for model selection,
    splitting folds on unique values (e.g., dates).
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

        data_local_train = data.loc[train_mask].copy()
        data_local_test = data.loc[test_mask].copy()

        if feat_engineering:
            data_local_train = feat_engineering(data_local_train)
            data_local_test = feat_engineering(data_local_test)

        X_local_train = data_local_train[features]
        X_local_test = data_local_test[features]

        if scale:
            scaler = StandardScaler()
            X_local_train = scaler.fit_transform(X_local_train)
            X_local_test = scaler.transform(X_local_test)

        imputer = SimpleImputer(strategy="median")
        X_local_train = imputer.fit_transform(X_local_train)
        X_local_test = imputer.transform(X_local_test)

        y_local_train = data_local_train[target]
        y_local_test = data_local_test[target]

        model = get_model(model_type)
        model.fit(X_local_train, y_local_train)

        model_eval = evaluate_model(
            model=model,
            X_train=X_local_train,
            X_test=X_local_test,
            y_train=data_local_train[[target, "OS_STATUS"]],
            y_test=data_local_test[[target, "OS_STATUS"]],
        )

        models.append(model)

        cindex_train = model_eval["cindex_train"]
        cindex_test = model_eval["cindex_test"]

        metrics["cindex_train"].append(cindex_train)
        metrics["cindex_test"].append(cindex_test)

        print(
            f"Fold {i+1} - IPCW C-index (Train: {cindex_train:.4f} | Test: {cindex_test:.4f})"
        )

    # Aggregate results (focus on test performance)
    cindexes = np.array(metrics["cindex_test"])
    mean_c = cindexes.mean()
    std_c = cindexes.std()
    min_c = cindexes.min()
    max_c = cindexes.max()

    print(
        f"IPCW C-index: {mean_c:.4f} (± {std_c:.4f}) "
        f"[Min: {min_c:.4f} ; Max: {max_c:.4f}]"
    )

    # Logging results
    if log:
        logfile = "predictions/model_selection.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - Model Selection ({model_type}): "
                f"Mean IPCW C-index: {mean_c:.4f} | Std: {std_c:.4f} | "
                f"Min: {min_c:.4f} | Max: {max_c:.4f}{note_str}\n"
            )

    # Feature importance
    if plot_ft_importance:
        try:
            plot_feature_importance(models, features, n_importance)
        except Exception:
            print("No possible to get feature importance for this model.")


def get_data(ids, unique_vals, col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series):
    """Extract subset of X and y given selected indices of unique values."""
    selected = unique_vals[ids]
    mask = col.isin(selected)
    return X_data.loc[mask], y_data.loc[mask], mask


def plot_feature_importance(models, features, n_importance=10):
    """
    Plot mean feature importance across trained models and log top features if required.
    """
    feature_importances = pd.DataFrame(
        [model.feature_importances_ for model in models], columns=features
    )
    mean_importance = feature_importances.mean().sort_values(ascending=False)

    top_features = mean_importance.head(n_importance).index.tolist()

    print(f"\nTop {n_importance} important features:")
    print(top_features)

    sns.barplot(
        data=feature_importances,
        orient="h",
        order=mean_importance.index,
    )
    return True
