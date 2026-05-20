"""Baseline estimators for the LogStruct paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from logstruct import LogStructClassifier, LogStructRegressor


def graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
    A = np.asarray(adjacency, dtype=np.float32)
    return np.diag(A.sum(axis=1)) - A


class FixedGraphLogStructClassifier(LogStructClassifier):
    """Like-for-like fixed-graph NSLR-style LogStruct baseline."""

    def __init__(self, prior_adjacency, **kwargs):
        kwargs.setdefault("freeze_adjacency", True)
        kwargs.setdefault("lambda_kl", 0.0)
        super().__init__(prior_adjacency=prior_adjacency, **kwargs)


class FixedGraphLogStructRegressor(LogStructRegressor):
    """Like-for-like fixed-graph LogStruct regressor baseline."""

    def __init__(self, prior_adjacency, **kwargs):
        kwargs.setdefault("freeze_adjacency", True)
        kwargs.setdefault("lambda_kl", 0.0)
        super().__init__(prior_adjacency=prior_adjacency, **kwargs)


class TorchLaplacianLogistic(BaseEstimator, ClassifierMixin):
    """Sparse logistic regression with a fixed Laplacian weight penalty.

    This ports the NSLR/glmgraph style objective into PyTorch:
    CE + lambda_l1 * |W|_1 + lambda_lap * trace(W L W^T).
    """

    def __init__(
        self,
        adjacency,
        lambda_l1: float = 1e-3,
        lambda_lap: float = 1.0,
        lr: float = 1e-2,
        max_iter: int = 200,
        random_state: int | None = 0,
        device: str = "cpu",
    ):
        self.adjacency = adjacency
        self.lambda_l1 = lambda_l1
        self.lambda_lap = lambda_lap
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_enc = np.searchsorted(self.classes_, y)
        self.scaler_ = StandardScaler().fit(X)
        X_t = torch.tensor(self.scaler_.transform(X), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_enc, dtype=torch.long, device=self.device)
        L = torch.tensor(graph_laplacian(self.adjacency), dtype=torch.float32, device=self.device)

        self.linear_ = nn.Linear(X.shape[1], len(self.classes_)).to(self.device)
        opt = torch.optim.AdamW(self.linear_.parameters(), lr=self.lr, weight_decay=0.0)
        ce = nn.CrossEntropyLoss()
        for _ in range(self.max_iter):
            opt.zero_grad()
            logits = self.linear_(X_t)
            W = self.linear_.weight
            loss = (
                ce(logits, y_t)
                + self.lambda_l1 * W.abs().sum()
                + self.lambda_lap * (W @ L * W).sum()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.linear_.parameters(), 1.0)
            opt.step()
        self.coef_ = self.linear_.weight.detach().cpu().numpy()
        self.intercept_ = self.linear_.bias.detach().cpu().numpy()
        return self

    def predict_proba(self, X):
        X_t = torch.tensor(self.scaler_.transform(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.softmax(self.linear_(X_t), dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class MLPClassifierTorch(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=128, lr=1e-3, max_iter=200, random_state=0, device="cpu"):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_enc = np.searchsorted(self.classes_, y)
        self.scaler_ = StandardScaler().fit(X)
        X_t = torch.tensor(self.scaler_.transform(X), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_enc, dtype=torch.long, device=self.device)
        self.net_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, len(self.classes_)),
        ).to(self.device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        for _ in range(self.max_iter):
            opt.zero_grad()
            loss = ce(self.net_(X_t), y_t)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        X_t = torch.tensor(self.scaler_.transform(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.softmax(self.net_(X_t), dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


@dataclass
class BaselineSpec:
    name: str
    estimator: Any
    grid: dict[str, list[Any]]


def classical_classification_specs(random_state: int = 0) -> list[BaselineSpec]:
    specs = [
        BaselineSpec(
            "logistic_l2",
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        BaselineSpec(
            "logistic_l1",
            LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            ),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        BaselineSpec(
            "elastic_net",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            ),
            {"C": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]},
        ),
        BaselineSpec(
            "random_forest",
            RandomForestClassifier(
                n_estimators=500, class_weight="balanced_subsample", random_state=random_state
            ),
            {"max_features": ["sqrt", 0.2]},
        ),
    ]
    return specs


def classical_regression_specs(random_state: int = 0) -> list[BaselineSpec]:
    return [
        BaselineSpec("ridge", Ridge(), {"alpha": [0.1, 1, 10, 100]}),
        BaselineSpec(
            "elastic_net",
            ElasticNet(max_iter=10000, random_state=random_state),
            {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]},
        ),
        BaselineSpec(
            "random_forest",
            RandomForestRegressor(n_estimators=500, random_state=random_state),
            {"max_features": ["sqrt", 0.2]},
        ),
    ]


def optional_xgboost_classifier(random_state: int = 0):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
    )


def optional_lightgbm_classifier(random_state: int = 0):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )
