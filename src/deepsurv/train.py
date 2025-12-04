import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lifelines.utils import concordance_index


def cox_ph_loss(log_hazard, events):
    """
    Negative log partial likelihood of Cox model.

    Args:
        log_hazard: tensor of shape (n, 1) — predicted log-risk F(x)
        events: tensor of shape (n,) — event indicators (1 = observed, 0 = censored)
    """
    # hazards must be sorted by descending survival time beforehand
    hazard_ratio = torch.exp(log_hazard)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = log_hazard - log_risk
    censored_likelihood = uncensored_likelihood * events.view(-1, 1)
    neg_likelihood = -torch.sum(censored_likelihood)
    return neg_likelihood


def train_deepsurv(
    model,
    x_train,
    e_train,
    t_train,
    x_valid=None,
    e_valid=None,
    t_valid=None,
    n_epochs=500,
    lr=1e-3,
    weight_decay=0.0,
    l1_reg=0.0,
    device="cpu",
    verbose=True,
):
    """
    Trains a DeepSurv model with Cox PH loss.

    Args:
        model: DeepSurv nn.Module
        x_train, e_train, t_train: training data
        x_valid, e_valid, t_valid: optional validation data
        n_epochs: int
        lr: learning rate
        weight_decay: L2 regularization (torch optimizer)
        l1_reg: L1 regularization coefficient
        device: 'cpu' or 'cuda'
    """
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=lr, weight_decay=weight_decay
    # )
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    e_train = torch.tensor(e_train, dtype=torch.float32).to(device)
    t_train = torch.tensor(t_train, dtype=torch.float32).to(device)

    # Sort descending by survival time
    sort_idx = torch.argsort(t_train, descending=True)
    x_train, e_train, t_train = x_train[sort_idx], e_train[sort_idx], t_train[sort_idx]

    if x_valid is not None:
        x_valid = torch.tensor(x_valid, dtype=torch.float32).to(device)
        e_valid = torch.tensor(e_valid, dtype=torch.float32).to(device)
        t_valid = torch.tensor(t_valid, dtype=torch.float32).to(device)
        sort_idx = torch.argsort(t_valid, descending=True)
        x_valid, e_valid, t_valid = (
            x_valid[sort_idx],
            e_valid[sort_idx],
            t_valid[sort_idx],
        )

    history = {"train_loss": [], "train_ci": [], "valid_loss": [], "valid_ci": []}
    start = time.time()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        log_hazard = model(x_train)
        loss = cox_ph_loss(log_hazard, e_train)

        # L1 regularization
        if l1_reg > 0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss += l1_reg * l1_penalty

        loss.backward()
        optimizer.step()

        # ---- Training metrics ----
        model.eval()
        with torch.no_grad():
            risks = -model(x_train).exp().cpu().numpy().flatten()
            ci_train = concordance_index(
                t_train.cpu().numpy(), risks, e_train.cpu().numpy()
            )

        history["train_loss"].append(loss.item())
        history["train_ci"].append(ci_train)

        # ---- Validation metrics ----
        if x_valid is not None:
            with torch.no_grad():
                log_hazard_val = model(x_valid)
                val_loss = cox_ph_loss(log_hazard_val, e_valid)
                if l1_reg > 0:
                    l1_penalty = sum(p.abs().sum() for p in model.parameters())
                    val_loss += l1_reg * l1_penalty
                risks_val = -log_hazard_val.exp().cpu().numpy().flatten()
                ci_val = concordance_index(
                    t_valid.cpu().numpy(), risks_val, e_valid.cpu().numpy()
                )

            history["valid_loss"].append(val_loss.item())
            history["valid_ci"].append(ci_val)

        if verbose and epoch % 10 == 0:
            if x_valid is not None:
                print(
                    f"Epoch {epoch:03d} | TrainLoss={loss.item():.4f} | TrainCI={ci_train:.4f} | "
                    f"ValLoss={val_loss.item():.4f} | ValCI={ci_val:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | TrainLoss={loss.item():.4f} | TrainCI={ci_train:.4f}"
                )

    if verbose:
        print(f"Training completed in {time.time() - start:.2f}s")

    return model
