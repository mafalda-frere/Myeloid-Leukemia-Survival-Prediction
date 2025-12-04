import numpy as np
from scipy.stats import mode


class EnsembleRegressor:
    """
    Simple ensemble regressor (unweighted average).
    """

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.column_stack([model.predict(X) for model in self.models])
        return preds.mean(axis=1)


class EnsembleClassifier:

    def __init__(self, models, weights=None, voting="soft"):

        self.models = models
        self.weights = weights if weights is not None else [1] * len(models)
        self.voting = voting

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        if self.voting == "soft":
            # weighted probability averaging
            probas = np.asarray([model.predict_proba(X) for model in self.models])
            weights = np.array(self.weights) / np.sum(self.weights)
            avg_proba = np.tensordot(weights, probas, axes=(0, 0))
            return np.argmax(avg_proba, axis=1)
        else:
            # hard voting
            predictions = np.column_stack([model.predict(X) for model in self.models])
            # apply weights by repeating predictions
            weighted_preds = np.repeat(predictions, self.weights, axis=1)
            return mode(weighted_preds, axis=1, keepdims=False).mode

    def predict_proba(self, X):
        if self.voting == "soft":
            probas = np.asarray([model.predict_proba(X) for model in self.models])
            weights = np.array(self.weights) / np.sum(self.weights)
            return np.tensordot(weights, probas, axes=(0, 0))
        else:
            raise AttributeError("predict_proba is only available with soft voting.")
