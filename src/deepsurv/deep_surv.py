import numpy as np
import torch
import torch.nn as nn


class DeepSurv(nn.Module):
    def __init__(
        self,
        n_in,
        hidden_layers_sizes=None,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    ):
        """
        n_in: number of input features
        hidden_layers_sizes: list of hidden layer sizes
        activation: 'relu', 'tanh', 'elu', 'sigmoid'
        dropout: dropout probability (0–1)
        batch_norm: bool, add batch normalization
        """
        super().__init__()
        layers = []
        input_dim = n_in

        act_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "sigmoid": nn.Sigmoid(),
        }
        act_fn = act_map.get(activation, nn.ReLU())

        for h in hidden_layers_sizes or []:
            layers.append(nn.Linear(input_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h

        # output: linear layer (log-risk)
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass -> predicted log-risk F(x)"""
        return self.network(x)

    def predict_risk(self, x):
        """Return predicted log-risk as numpy array."""
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            return self.forward(x).cpu().numpy().flatten()

    def predict_partial_hazard(self, x):
        """Return exp(F(x))"""
        return np.exp(self.predict_risk(x))

    def predict(self, x):
        """
        Compatibility method for evaluation pipelines.
        Returns predicted log-risk (same as predict_risk).
        """
        return self.predict_risk(x)
