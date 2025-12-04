import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sksurv.util import Surv

from .deep_surv import DeepSurv
from .train import train_deepsurv
from .evaluate import evaluate_model


def model_selection_using_kfold_deepsurv(
    data: pd.DataFrame,
    features: list[str],
    target: str = "OS_YEARS",
    status: str = "OS_STATUS",
    hidden_layers_sizes: list[int] = [64, 32],
    dropout: float = 0.3,
    n_splits: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    l1_reg: float = 0.0,
    feat_engineering=None,
    unique_id: str = "ROW_ID",
    device: str = "cpu",
    log: bool = False,
    log_note: str = None,
):
    """
    Perform K-Fold cross-validation with DeepSurv (PyTorch),
    evaluated using IPC-weighted C-index (via sksurv).
    """

    unique_vals = data[unique_id].unique()
    metrics = {"cindex_train": [], "cindex_test": []}
    models = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):

        train_ids = unique_vals[train_idx]
        test_ids = unique_vals[test_idx]

        df_train = data.loc[data[unique_id].isin(train_ids)].copy()
        df_test = data.loc[data[unique_id].isin(test_ids)].copy()

        # Optional feature engineering
        if feat_engineering:
            df_train = feat_engineering(df_train)
            df_test = feat_engineering(df_test)

        # Separate features and targets
        X_train = df_train[features]
        X_test = df_test[features]
        y_train = df_train[[status, target]].copy()
        y_test = df_test[[status, target]].copy()

        # Handle missing values
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize DeepSurv model
        n_in = X_train.shape[1]
        model = DeepSurv(
            n_in=n_in,
            hidden_layers_sizes=hidden_layers_sizes,
            dropout=dropout,
            activation="relu",
        )

        # Train model
        _ = train_deepsurv(
            model,
            x_train=X_train,
            e_train=y_train[status].values.astype(np.float32),
            t_train=y_train[target].values.astype(np.float32),
            x_valid=X_test,
            e_valid=y_test[status].values.astype(np.float32),
            t_valid=y_test[target].values.astype(np.float32),
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            l1_reg=l1_reg,
            device=device,
            verbose=False,
        )

        models.append(model)

        # --- Evaluate using IPC-weighted concordance index ---
        model_eval = evaluate_model(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            target=target,
            status=status,
            tau=7,
            verbose=False,
            log=False,
        )

        cindex_train = model_eval["cindex_train"]
        cindex_test = model_eval["cindex_test"]

        metrics["cindex_train"].append(cindex_train)
        metrics["cindex_test"].append(cindex_test)

        print(
            f"Fold {i+1} - IPCW C-index (Train: {cindex_train:.4f} | Test: {cindex_test:.4f})"
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
        f"\nConcordance Index (Test, IPCW): {mean_c:.4f} ± {std_c:.4f} "
        f"[Min: {min_c:.4f} ; Max: {max_c:.4f}]"
    )

    # --- Optional logging ---
    if log:
        logfile = "predictions/model_selection_deepsurv.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - DeepSurv (IPCW): Mean C-index: {mean_c:.4f} | Std: {std_c:.4f} | "
                f"Min: {min_c:.4f} | Max: {max_c:.4f}{note_str}\n"
            )

    return {
        "models": models,
        "metrics": metrics,
        "mean_cindex": mean_c,
        "std_cindex": std_c,
    }
