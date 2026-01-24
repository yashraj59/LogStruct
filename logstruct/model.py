"""
LogStruct: Network-structured regression for biological data.

Sklearn-compatible API for logistic/linear regression with learned feature-graph smoothing.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse, spmatrix
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .utils import pick_device, normalize_rows


class _StructuredModel(nn.Module):
    """
    Internal PyTorch module for structured regression.
    
    Smoothing: x_smooth = x @ S
    where S = self_loop_weight * I + neighbor_weight * row_norm(sym(sigmoid(adj_logits / T)))
    
    Regularization:
        - Elastic net on weights
        - Laplacian smoothness on coefficients
        - KL divergence to prior adjacency
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prior_adjacency: torch.Tensor,
        *,
        temperature: float = 0.8,
        self_loop_weight: float = 1.0,
        neighbor_weight: float = 1.0,
        freeze_adjacency: bool = False,
        device: torch.device,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.self_loop_weight = self_loop_weight
        self.neighbor_weight = neighbor_weight
        self.device = device

        # Linear head
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

        # Prior logits (avoid ±inf)
        prior = torch.clamp(prior_adjacency, 1e-6, 1 - 1e-6)
        prior_logits = torch.log(prior / (1 - prior))

        # Store prior probs for KL computation
        self.register_buffer("prior_probs", torch.sigmoid(prior_logits / temperature))

        # Trainable adjacency
        self.adj_logits = nn.Parameter(prior_logits.clone(), requires_grad=not freeze_adjacency)
        
        # Zero out diagonal (no self-loops in learned graph)
        with torch.no_grad():
            idx = torch.arange(input_dim, device=device)
            self.adj_logits[idx, idx] = -10.0

        self.to(device)

    def current_adjacency(self) -> torch.Tensor:
        """Symmetric probability adjacency, no self-loops."""
        A = torch.sigmoid(self.adj_logits / self.temperature)
        idx = torch.arange(A.size(0), device=A.device)
        A = A.clone()
        A[idx, idx] = 0.0
        A = 0.5 * (A + A.t())
        return A

    def smoothing_operator(self) -> torch.Tensor:
        """S = self_loop * I + neighbor * row_norm(A)"""
        A = self.current_adjacency()
        A_norm = normalize_rows(A)
        I = torch.eye(A.size(0), device=A.device)
        return self.self_loop_weight * I + self.neighbor_weight * A_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.smoothing_operator()
        x_smooth = x @ S
        return self.linear(x_smooth)

    def elastic_net_penalty(self, alpha: float) -> torch.Tensor:
        """α||W||_1 + (1-α) * 0.5||W||_2^2"""
        W = self.linear.weight
        l1 = W.abs().sum()
        l2_sq = (W * W).sum() * 0.5
        return alpha * l1 + (1 - alpha) * l2_sq

    def smoothness_penalty(self) -> torch.Tensor:
        """trace(W @ L @ W.T) where L = D - A (graph Laplacian)"""
        A = self.current_adjacency()
        D = torch.diag(A.sum(dim=1))
        L = D - A
        W = self.linear.weight
        return (W @ L * W).sum()

    def kl_prior_penalty(self) -> torch.Tensor:
        """KL(q || p) for Bernoulli adjacency entries."""
        q = self.current_adjacency()
        p = self.prior_probs
        eps = 1e-6
        q = torch.clamp(q, eps, 1 - eps)
        p = torch.clamp(p, eps, 1 - eps)
        kl = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
        return kl.sum()


class LogStructClassifier(BaseEstimator, ClassifierMixin):
    """
    Network-structured logistic regression with learned feature-graph smoothing.
    
    Parameters
    ----------
    prior_adjacency : array-like of shape (n_features, n_features)
        Prior adjacency matrix (e.g., PPI network, pathway co-membership).
        Values should be in [0, 1] representing edge probabilities.
    
    alpha : float, default=0.5
        Elastic net mixing: 0 = pure L2, 1 = pure L1.
    
    lambda_en : float, default=1.0
        Elastic net regularization strength.
    
    lambda_smooth : float, default=1.0
        Coefficient smoothness penalty (Laplacian regularization).
    
    lambda_kl : float, default=1.0
        KL divergence penalty to prior adjacency.
    
    temperature : float, default=0.8
        Temperature for adjacency sigmoid (lower = sharper edges).
    
    self_loop_weight : float, default=1.0
        Weight for self-connections in smoothing operator.
    
    neighbor_weight : float, default=1.0
        Weight for neighbor aggregation in smoothing operator.
    
    learning_rate : float, default=0.01
        Learning rate for AdamW optimizer.
    
    max_iter : int, default=100
        Maximum training epochs.
    
    tol : float, default=1e-4
        Tolerance for early stopping (on validation loss).
    
    validation_fraction : float, default=0.1
        Fraction of training data for validation.
    
    early_stopping : bool, default=True
        Whether to use early stopping.
    
    n_iter_no_change : int, default=10
        Patience for early stopping.
    
    scale_features : bool, default=True
        Whether to standardize features before fitting.
    
    sparsity_threshold : float, default=0.05
        Threshold for sparsifying learned adjacency.
    
    freeze_adjacency : bool, default=False
        If True, don't learn adjacency (use prior directly).
    
    device : str, default="auto"
        Device for computation: "auto", "cpu", "cuda", or "mps".
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    verbose : bool, default=False
        Print training progress.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_classes, n_features) or (n_features,)
        Learned coefficients.
    
    intercept_ : ndarray of shape (n_classes,) or (1,)
        Learned intercepts.
    
    adjacency_ : ndarray of shape (n_features, n_features)
        Learned (and thresholded) adjacency matrix.
    
    classes_ : ndarray
        Unique class labels.
    
    n_iter_ : int
        Actual number of training iterations.
    
    history_ : dict
        Training history with loss and accuracy.
    
    Examples
    --------
    >>> from logstruct import LogStructClassifier
    >>> from logstruct.priors import from_random
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> prior = from_random(50, density=0.1)
    >>> clf = LogStructClassifier(prior_adjacency=prior, max_iter=50)
    >>> clf.fit(X, y)
    >>> clf.score(X, y)
    """

    def __init__(
        self,
        prior_adjacency: np.ndarray | spmatrix,
        *,
        alpha: float = 0.5,
        lambda_en: float = 1.0,
        lambda_smooth: float = 1.0,
        lambda_kl: float = 1.0,
        temperature: float = 0.8,
        self_loop_weight: float = 1.0,
        neighbor_weight: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-4,
        validation_fraction: float = 0.1,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        scale_features: bool = True,
        sparsity_threshold: float = 0.05,
        freeze_adjacency: bool = False,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.prior_adjacency = prior_adjacency
        self.alpha = alpha
        self.lambda_en = lambda_en
        self.lambda_smooth = lambda_smooth
        self.lambda_kl = lambda_kl
        self.temperature = temperature
        self.self_loop_weight = self_loop_weight
        self.neighbor_weight = neighbor_weight
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.scale_features = scale_features
        self.sparsity_threshold = sparsity_threshold
        self.freeze_adjacency = freeze_adjacency
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the structured classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        
        Returns
        -------
        self
        """
        # Setup
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = pick_device() if self.device == "auto" else torch.device(self.device)

        # Convert inputs
        X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Convert labels to 0-indexed
        y_encoded = np.searchsorted(self.classes_, y)

        # Validation split
        if self.early_stopping and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded,
                test_size=self.validation_fraction,
                stratify=y_encoded,
                random_state=self.random_state,
            )
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None

        # Scale
        if self.scale_features:
            self._scaler = StandardScaler().fit(X_train)
            X_train = self._scaler.transform(X_train)
            if X_val is not None:
                X_val = self._scaler.transform(X_val)
        else:
            self._scaler = None

        # To tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

        # Prior adjacency
        prior = np.asarray(
            self.prior_adjacency.toarray() if issparse(self.prior_adjacency) 
            else self.prior_adjacency,
            dtype=np.float32
        )
        if prior.shape != (n_features, n_features):
            raise ValueError(
                f"prior_adjacency shape {prior.shape} doesn't match n_features={n_features}"
            )
        prior_t = torch.tensor(prior, dtype=torch.float32, device=device)

        # Model
        self._model = _StructuredModel(
            input_dim=n_features,
            output_dim=n_classes,
            prior_adjacency=prior_t,
            temperature=self.temperature,
            self_loop_weight=self.self_loop_weight,
            neighbor_weight=self.neighbor_weight,
            freeze_adjacency=self.freeze_adjacency,
            device=device,
        )

        optimizer = optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.history_ = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.max_iter):
            self._model.train()
            optimizer.zero_grad()

            logits = self._model(X_train_t)
            ce_loss = criterion(logits, y_train_t)
            reg_loss = (
                self.lambda_en * self._model.elastic_net_penalty(self.alpha)
                + self.lambda_smooth * self._model.smoothness_penalty()
                + self.lambda_kl * self._model.kl_prior_penalty()
            )
            loss = ce_loss + reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                train_acc = (logits.argmax(1) == y_train_t).float().mean().item()

            self.history_["train_loss"].append(loss.item())
            self.history_["train_acc"].append(train_acc)

            # Validation
            if X_val is not None:
                self._model.eval()
                with torch.no_grad():
                    val_logits = self._model(X_val_t)
                    val_ce = criterion(val_logits, y_val_t)
                    val_reg = (
                        self.lambda_en * self._model.elastic_net_penalty(self.alpha)
                        + self.lambda_smooth * self._model.smoothness_penalty()
                        + self.lambda_kl * self._model.kl_prior_penalty()
                    )
                    val_loss = val_ce + val_reg
                    val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()

                self.history_["val_loss"].append(val_loss.item())
                self.history_["val_acc"].append(val_acc)

                # Early stopping
                if val_loss.item() + self.tol < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if self.early_stopping and no_improve >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if self.verbose and (epoch % 10 == 0 or epoch == self.max_iter - 1):
                msg = f"[{epoch+1}/{self.max_iter}] loss={loss.item():.4f} acc={train_acc:.3f}"
                if X_val is not None:
                    msg += f" val_loss={val_loss.item():.4f} val_acc={val_acc:.3f}"
                print(msg)

        self.n_iter_ = epoch + 1

        # Restore best
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Extract parameters
        self._model.eval()
        self.coef_ = self._model.linear.weight.detach().cpu().numpy()
        self.intercept_ = self._model.linear.bias.detach().cpu().numpy()
        
        if n_classes == 2:
            self.coef_ = self.coef_[1] - self.coef_[0]
            self.intercept_ = self.intercept_[1] - self.intercept_[0]

        # Learned adjacency
        adj = self._model.current_adjacency().detach().cpu().numpy()
        adj = 0.5 * (adj + adj.T)
        np.fill_diagonal(adj, 0.0)
        if self.sparsity_threshold > 0:
            adj[adj < self.sparsity_threshold] = 0.0
        self.adjacency_ = adj

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self)
        
        X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        device = next(self._model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        indices = probs.argmax(axis=1)
        return self.classes_[indices]

    def get_top_edges(self, n: int = 10) -> list[tuple[int, int, float]]:
        """
        Get top n edges by learned weight.
        
        Returns list of (i, j, weight) tuples.
        """
        check_is_fitted(self)
        adj = self.adjacency_
        iu, ju = np.triu_indices(adj.shape[0], k=1)
        weights = adj[iu, ju]
        top_idx = np.argsort(weights)[-n:][::-1]
        return [(int(iu[i]), int(ju[i]), float(weights[i])) for i in top_idx]


class LogStructRegressor(BaseEstimator, RegressorMixin):
    """
    Network-structured linear regression with learned feature-graph smoothing.
    
    Same parameters as LogStructClassifier, but for regression tasks.
    See LogStructClassifier for full parameter documentation.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_targets, n_features) or (n_features,)
        Learned coefficients.
    
    intercept_ : ndarray of shape (n_targets,) or float
        Learned intercepts.
    
    adjacency_ : ndarray of shape (n_features, n_features)
        Learned adjacency matrix.
    """

    def __init__(
        self,
        prior_adjacency: np.ndarray | spmatrix,
        *,
        alpha: float = 0.5,
        lambda_en: float = 1.0,
        lambda_smooth: float = 1.0,
        lambda_kl: float = 1.0,
        temperature: float = 0.8,
        self_loop_weight: float = 1.0,
        neighbor_weight: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-4,
        validation_fraction: float = 0.1,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        scale_features: bool = True,
        sparsity_threshold: float = 0.05,
        freeze_adjacency: bool = False,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.prior_adjacency = prior_adjacency
        self.alpha = alpha
        self.lambda_en = lambda_en
        self.lambda_smooth = lambda_smooth
        self.lambda_kl = lambda_kl
        self.temperature = temperature
        self.self_loop_weight = self_loop_weight
        self.neighbor_weight = neighbor_weight
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.scale_features = scale_features
        self.sparsity_threshold = sparsity_threshold
        self.freeze_adjacency = freeze_adjacency
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the structured regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        
        Returns
        -------
        self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = pick_device() if self.device == "auto" else torch.device(self.device)

        X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self._squeeze_output = True
        else:
            self._squeeze_output = False

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        # Validation split
        if self.early_stopping and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Scale
        if self.scale_features:
            self._scaler = StandardScaler().fit(X_train)
            X_train = self._scaler.transform(X_train)
            if X_val is not None:
                X_val = self._scaler.transform(X_val)
        else:
            self._scaler = None

        # To tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

        # Prior
        prior = np.asarray(
            self.prior_adjacency.toarray() if issparse(self.prior_adjacency)
            else self.prior_adjacency,
            dtype=np.float32
        )
        if prior.shape != (n_features, n_features):
            raise ValueError(
                f"prior_adjacency shape {prior.shape} doesn't match n_features={n_features}"
            )
        prior_t = torch.tensor(prior, dtype=torch.float32, device=device)

        # Model
        self._model = _StructuredModel(
            input_dim=n_features,
            output_dim=n_targets,
            prior_adjacency=prior_t,
            temperature=self.temperature,
            self_loop_weight=self.self_loop_weight,
            neighbor_weight=self.neighbor_weight,
            freeze_adjacency=self.freeze_adjacency,
            device=device,
        )

        optimizer = optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        criterion = nn.MSELoss()

        # Training
        self.history_ = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.max_iter):
            self._model.train()
            optimizer.zero_grad()

            preds = self._model(X_train_t)
            mse_loss = criterion(preds, y_train_t)
            reg_loss = (
                self.lambda_en * self._model.elastic_net_penalty(self.alpha)
                + self.lambda_smooth * self._model.smoothness_penalty()
                + self.lambda_kl * self._model.kl_prior_penalty()
            )
            loss = mse_loss + reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()

            self.history_["train_loss"].append(loss.item())

            # Validation
            if X_val is not None:
                self._model.eval()
                with torch.no_grad():
                    val_preds = self._model(X_val_t)
                    val_mse = criterion(val_preds, y_val_t)
                    val_reg = (
                        self.lambda_en * self._model.elastic_net_penalty(self.alpha)
                        + self.lambda_smooth * self._model.smoothness_penalty()
                        + self.lambda_kl * self._model.kl_prior_penalty()
                    )
                    val_loss = val_mse + val_reg

                self.history_["val_loss"].append(val_loss.item())

                if val_loss.item() + self.tol < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if self.early_stopping and no_improve >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if self.verbose and (epoch % 10 == 0 or epoch == self.max_iter - 1):
                msg = f"[{epoch+1}/{self.max_iter}] loss={loss.item():.4f}"
                if X_val is not None:
                    msg += f" val_loss={val_loss.item():.4f}"
                print(msg)

        self.n_iter_ = epoch + 1

        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Extract
        self._model.eval()
        self.coef_ = self._model.linear.weight.detach().cpu().numpy()
        self.intercept_ = self._model.linear.bias.detach().cpu().numpy()

        if self._squeeze_output:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = float(self.intercept_[0])

        # Adjacency
        adj = self._model.current_adjacency().detach().cpu().numpy()
        adj = 0.5 * (adj + adj.T)
        np.fill_diagonal(adj, 0.0)
        if self.sparsity_threshold > 0:
            adj[adj < self.sparsity_threshold] = 0.0
        self.adjacency_ = adj

        return self

    def predict(self, X):
        """Predict target values."""
        check_is_fitted(self)

        X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        device = next(self._model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_t).cpu().numpy()

        if self._squeeze_output:
            preds = preds.ravel()

        return preds

    def get_top_edges(self, n: int = 10) -> list[tuple[int, int, float]]:
        """Get top n edges by learned weight."""
        check_is_fitted(self)
        adj = self.adjacency_
        iu, ju = np.triu_indices(adj.shape[0], k=1)
        weights = adj[iu, ju]
        top_idx = np.argsort(weights)[-n:][::-1]
        return [(int(iu[i]), int(ju[i]), float(weights[i])) for i in top_idx]
