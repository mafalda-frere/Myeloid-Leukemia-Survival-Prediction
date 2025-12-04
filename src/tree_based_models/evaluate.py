from datetime import datetime
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    tau: int = 7,
    verbose: bool = False,
    log: bool = False,
    log_note: str = None,
) -> dict:
    """
    Evaluate a survival model using IPC-weighted concordance index (C-index)
    on both train and test sets.

    y_train / y_test must contain OS_STATUS and OS_YEARS columns.
    """

    # Convert labels to survival objects
    y_train_surv = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_train)
    y_test_surv = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_test)

    # Predict risk scores (negative sign for higher risk = shorter survival)
    pred_train = -model.predict(X_train)
    pred_test = -model.predict(X_test)

    # IPCW Concordance index
    train_ci_ipcw = concordance_index_ipcw(
        y_train_surv, y_train_surv, pred_train, tau=tau
    )[0]
    test_ci_ipcw = concordance_index_ipcw(
        y_train_surv, y_test_surv, pred_test, tau=tau
    )[0]

    results = {
        "cindex_train": train_ci_ipcw,
        "cindex_test": test_ci_ipcw,
    }

    if verbose:
        print(
            f"IPCW C-index (τ={tau}) - Train: {train_ci_ipcw:.4f} | Test: {test_ci_ipcw:.4f}"
        )

    if log:
        logfile = "predictions/evaluation.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - IPCW evaluation: "
                f"Train={train_ci_ipcw:.4f}, Test={test_ci_ipcw:.4f}{note_str}\n"
            )

    return results


def evaluate_model_surv(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    tau: int = 7,
    verbose: bool = False,
    log: bool = False,
    log_note: str = None,
) -> dict:
    """
    Evaluate a survival model using IPC-weighted concordance index (C-index)
    on both train and test sets.

    y_train / y_test must be structured survival arrays (from sksurv.util.Surv).
    """

    # Predict risk scores (negative sign for higher risk = shorter survival)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # IPCW Concordance index
    train_ci_ipcw = concordance_index_ipcw(y_train, y_train, pred_train, tau=tau)[0]
    test_ci_ipcw = concordance_index_ipcw(y_train, y_test, pred_test, tau=tau)[0]

    results = {
        "cindex_train": train_ci_ipcw,
        "cindex_test": test_ci_ipcw,
    }

    if verbose:
        print(
            f"IPCW C-index (τ={tau}) - Train: {train_ci_ipcw:.4f} | Test: {test_ci_ipcw:.4f}"
        )

    if log:
        logfile = "predictions/evaluation.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - IPCW evaluation: "
                f"Train={train_ci_ipcw:.4f}, Test={test_ci_ipcw:.4f}{note_str}\n"
            )

    return results
