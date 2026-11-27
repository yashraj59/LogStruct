"""
LogStruct: Network-structured regression for biological data.

Sklearn-compatible API for logistic/linear regression with learned feature-graph smoothing.

The feature graph is stored as an edge list. Only edges that are in the prior
(plus optional data-driven candidate edges) carry a learnable weight, so the
number of graph parameters is the number of edges, not n_features ** 2.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse as sp
from scipy.sparse import issparse, spmatrix
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

from .utils import pick_device

# Above this many features, adjacency_ is returned as a scipy CSR matrix
# when sparse_adjacency="auto".
_DENSE_ADJACENCY_MAX_FEATURES = 5000


# --------------------------------------------------------------------------
# Edge extraction
# --------------------------------------------------------------------------

def _edges_from_prior(prior, edge_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn a prior adjacency into an undirected edge list.

    An entry is an edge if max(prior[i, j], prior[j, i]) > edge_threshold.

    Returns
    -------
    edges : int64 array of shape (n_edges, 2) with i < j
    probs : float32 array of shape (n_edges,)
    """
    if issparse(prior):
        A = sp.csr_matrix(prior, dtype=np.float32)
        A = A.maximum(A.T)
        A = sp.triu(A, k=1).tocoo()
        keep = A.data > edge_threshold
        rows, cols, vals = A.row[keep], A.col[keep], A.data[keep]
    else:
        A = np.asarray(prior, dtype=np.float32)
        A = np.maximum(A, A.T)
        rows, cols = np.nonzero(np.triu(A > edge_threshold, k=1))
        vals = A[rows, cols]

    edges = np.stack([rows, cols], axis=1).astype(np.int64)
    return edges, vals.astype(np.float32)


def _candidate_edges(
    X: np.ndarray,
    n_neighbors: int,
    existing: np.ndarray,
    block_size: int = 1024,
) -> np.ndarray:
    """
    Data-driven candidate edges: each feature is linked to its n_neighbors
    most correlated features (absolute Pearson correlation in X).

    Edges already in `existing` are left out.

    Returns
    -------
    edges : int64 array of shape (n_new_edges, 2) with i < j
    """
    n, p = X.shape
    k = min(n_neighbors, p - 1)
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    Xc = X - X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(Xc, axis=0, keepdims=True)
    norm[norm == 0] = 1.0
    Z = torch.from_numpy((Xc / norm).astype(np.float32))

    src, dst = [], []
    for start in range(0, p, block_size):
        stop = min(start + block_size, p)
        corr = (Z[:, start:stop].T @ Z).abs()
        corr[torch.arange(stop - start), torch.arange(start, stop)] = -1.0
        top = corr.topk(k, dim=1).indices.numpy()
        src.append(np.repeat(np.arange(start, stop), k))
        dst.append(top.ravel())
    src, dst = np.concatenate(src), np.concatenate(dst)

    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keys = np.unique(lo.astype(np.int64) * p + hi)
    if len(existing):
        existing_keys = existing[:, 0].astype(np.int64) * p + existing[:, 1]
        keys = np.setdiff1d(keys, existing_keys, assume_unique=True)
    return np.stack([keys // p, keys % p], axis=1).astype(np.int64)


# --------------------------------------------------------------------------
# Torch module
# --------------------------------------------------------------------------

class _StructuredModel(nn.Module):
    """
    Internal PyTorch module for structured regression.

    Smoothing: x_smooth = x @ S, prediction = x_smooth @ W.T + b
    where S = self_loop_weight * I + neighbor_weight * row_norm(A)
    and A is a sparse symmetric matrix with A_ij = sigmoid(edge_logit_ij / T)
    on the edge list and 0 everywhere else.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edges: torch.Tensor,
        edge_prior: torch.Tensor,
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

        # Start from zero coefficients. The problem is convex in the weights,
        # and a random start adds noise that the L1 step then has to remove.
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        prior = torch.clamp(edge_prior, 1e-4, 1 - 1e-4)
        self.register_buffer("edge_i", edges[:, 0].clone())
        self.register_buffer("edge_j", edges[:, 1].clone())
        self.register_buffer("edge_prior", prior)

        # Start exactly at the prior: sigmoid(logit / T) == prior.
        init_logits = temperature * torch.log(prior / (1 - prior))
        self.edge_logits = nn.Parameter(init_logits, requires_grad=not freeze_adjacency)

        self.to(device)

    @property
    def n_edges(self) -> int:
        return self.edge_i.numel()

    def edge_weights(self) -> torch.Tensor:
        """Edge probabilities, one per undirected edge."""
        return torch.sigmoid(self.edge_logits / self.temperature)

    def effective_weight(self) -> torch.Tensor:
        """
        Coefficients on the raw features: W @ S.T, shape (output_dim, input_dim).

        (x @ S) @ W.T is the same as x @ (W @ S.T).T, so the graph smoothing
        is applied to the small coefficient matrix instead of the data. The
        cost per step then depends on the number of edges, not on the number
        of cells.
        """
        W = self.linear.weight
        if self.n_edges == 0 or self.neighbor_weight == 0:
            return self.self_loop_weight * W

        w = self.edge_weights()
        deg = torch.zeros(self.input_dim, device=w.device, dtype=w.dtype)
        deg = deg.index_add(0, self.edge_i, w).index_add(0, self.edge_j, w)

        # Row-normalized A has entries A_ij / deg_i, so
        # (W @ A_norm.T)[:, i] = sum_j (A_ij / deg_i) * W[:, j]
        src = torch.cat([self.edge_i, self.edge_j])
        dst = torch.cat([self.edge_j, self.edge_i])
        vals = torch.cat([w, w]) / (deg[src] + 1e-8)
        neighbors = torch.zeros_like(W).index_add(1, src, W[:, dst] * vals)

        return self.self_loop_weight * W + self.neighbor_weight * neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.effective_weight().t() + self.linear.bias

    # ---- penalties (all are sums; the estimator divides them by n_samples) ----

    def l1_penalty(self) -> torch.Tensor:
        return self.linear.weight.abs().sum()

    def l2_penalty(self) -> torch.Tensor:
        W = self.linear.weight
        return 0.5 * (W * W).sum()

    def smoothness_penalty(self) -> torch.Tensor:
        """trace(W L W^T) = sum over edges of A_ij * ||W[:, i] - W[:, j]||^2"""
        if self.n_edges == 0:
            return self.linear.weight.new_zeros(())
        W = self.linear.weight
        diff = W[:, self.edge_i] - W[:, self.edge_j]
        return (self.edge_weights() * (diff * diff).sum(dim=0)).sum()

    def kl_prior_penalty(self) -> torch.Tensor:
        """Sum over edges of KL(q || p) for Bernoulli edge probabilities."""
        if self.n_edges == 0:
            return self.linear.weight.new_zeros(())
        eps = 1e-6
        q = torch.clamp(self.edge_weights(), eps, 1 - eps)
        p = self.edge_prior
        kl = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
        return kl.sum()


def _prox_l1_adam(param: torch.Tensor, optimizer: optim.Optimizer, strength: float) -> None:
    """
    Proximal (soft-threshold) step for an L1 penalty, in the metric Adam uses.

    Adam moves each weight by lr * m_hat / (sqrt(v_hat) + eps), so the matching
    soft threshold is lr * strength / (sqrt(v_hat) + eps). A weight stays at
    exactly zero whenever its average gradient is smaller than `strength`,
    which is the lasso optimality condition.
    """
    state = optimizer.state.get(param)
    if not state:
        return
    group = optimizer.param_groups[0]
    beta2 = group["betas"][1]
    step = float(state["step"])
    v_hat = state["exp_avg_sq"] / (1 - beta2 ** step)
    thresh = group["lr"] * strength / (v_hat.sqrt() + group["eps"])
    with torch.no_grad():
        param.copy_(torch.sign(param) * torch.clamp(param.abs() - thresh, min=0.0))


# --------------------------------------------------------------------------
# Shared estimator logic
# --------------------------------------------------------------------------

class _BaseLogStruct(BaseEstimator):
    """Shared fitting logic for the classifier and the regressor."""

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
        edge_threshold: float = 0.1,
        n_candidate_neighbors: int = 0,
        candidate_prior: float = 0.05,
        learning_rate: float = 0.03,
        max_iter: int = 500,
        tol: float = 1e-4,
        validation_fraction: float = 0.1,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        scale_features: bool = True,
        sparsity_threshold: float = 0.05,
        sparse_adjacency: bool | str = "auto",
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
        self.edge_threshold = edge_threshold
        self.n_candidate_neighbors = n_candidate_neighbors
        self.candidate_prior = candidate_prior
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.scale_features = scale_features
        self.sparsity_threshold = sparsity_threshold
        self.sparse_adjacency = sparse_adjacency
        self.freeze_adjacency = freeze_adjacency
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ---- helpers ----

    def _build_edges(self, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_features = X_train.shape[1]
        prior = self.prior_adjacency
        if not issparse(prior):
            prior = np.asarray(prior, dtype=np.float32)
        if prior.shape != (n_features, n_features):
            raise ValueError(
                f"prior_adjacency shape {prior.shape} doesn't match n_features={n_features}"
            )

        edges, probs = _edges_from_prior(prior, self.edge_threshold)
        self.n_prior_edges_ = len(edges)

        if self.n_candidate_neighbors > 0:
            extra = _candidate_edges(X_train, self.n_candidate_neighbors, edges)
            edges = np.concatenate([edges, extra], axis=0)
            probs = np.concatenate(
                [probs, np.full(len(extra), self.candidate_prior, dtype=np.float32)]
            )
        return edges, probs

    def _fit_loop(self, X_train, y_train, X_val, y_val, output_dim, criterion, y_dtype,
                  track_accuracy: bool) -> None:
        device = pick_device() if self.device == "auto" else torch.device(self.device)
        n_train, n_features = X_train.shape

        if self.scale_features:
            self._scaler = StandardScaler().fit(X_train)
            X_train = self._scaler.transform(X_train)
            if X_val is not None:
                X_val = self._scaler.transform(X_val)
        else:
            self._scaler = None

        edges, edge_prior = self._build_edges(X_train)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=y_dtype, device=device)
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val, dtype=y_dtype, device=device)

        self._model = _StructuredModel(
            input_dim=n_features,
            output_dim=output_dim,
            edges=torch.tensor(edges, dtype=torch.long),
            edge_prior=torch.tensor(edge_prior, dtype=torch.float32),
            temperature=self.temperature,
            self_loop_weight=self.self_loop_weight,
            neighbor_weight=self.neighbor_weight,
            freeze_adjacency=self.freeze_adjacency,
            device=device,
        )
        model = self._model
        criterion = criterion.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=0.0)

        # Penalties are sums divided by n_samples (the MAP convention, same as
        # sklearn's C = 1 / lambda). Their pull does not depend on how many
        # weights or edges there are, and it fades as more data comes in.
        l1_strength = self.lambda_en * self.alpha / n_train
        l2_strength = self.lambda_en * (1 - self.alpha) / n_train
        smooth_strength = self.lambda_smooth / n_train
        kl_strength = self.lambda_kl / n_train
        learn_edges = model.n_edges > 0 and not self.freeze_adjacency

        self.history_ = {"train_loss": [], "val_loss": []}
        if track_accuracy:
            self.history_["train_acc"] = []
            self.history_["val_acc"] = []

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        best_train_loss = float("inf")
        first_val_loss = None
        improved_once = False
        epoch = -1

        for epoch in range(self.max_iter):
            model.train()
            optimizer.zero_grad()

            out = model(X_train_t)
            task_loss = criterion(out, y_train_t)
            # The L1 term is handled by the proximal step below, not by autograd.
            smooth_loss = task_loss + l2_strength * model.l2_penalty()
            if smooth_strength > 0:
                smooth_loss = smooth_loss + smooth_strength * model.smoothness_penalty()
            if learn_edges and kl_strength > 0:
                smooth_loss = smooth_loss + kl_strength * model.kl_prior_penalty()

            # Full objective at the current parameters (before this step).
            with torch.no_grad():
                loss_value = smooth_loss.item() + l1_strength * model.l1_penalty().item()
            if X_val is None and loss_value < best_train_loss:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            smooth_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if l1_strength > 0:
                _prox_l1_adam(model.linear.weight, optimizer, l1_strength)

            self.history_["train_loss"].append(loss_value)
            if track_accuracy:
                train_acc = (out.argmax(1) == y_train_t).float().mean().item()
                self.history_["train_acc"].append(train_acc)

            # Validation: task loss only. The penalties do not depend on the
            # validation data, so they would only add noise to early stopping.
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_out = model(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
                self.history_["val_loss"].append(val_loss)
                if track_accuracy:
                    val_acc = (val_out.argmax(1) == y_val_t).float().mean().item()
                    self.history_["val_acc"].append(val_acc)

                if first_val_loss is None:
                    first_val_loss = val_loss
                elif val_loss + self.tol < first_val_loss:
                    improved_once = True

                if val_loss + self.tol < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                # Adam moves every coefficient at the same speed in its first
                # steps, so validation loss can go up before it goes down.
                # Patience only counts once the loss has beaten its start value.
                if not improved_once:
                    no_improve = 0

                if self.early_stopping and no_improve >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            else:
                # No validation set: follow the training objective.
                if loss_value < best_train_loss * (1 - self.tol) or best_train_loss == float("inf"):
                    no_improve = 0
                else:
                    no_improve += 1
                best_train_loss = min(best_train_loss, loss_value)

                if self.tol > 0 and no_improve >= self.n_iter_no_change:
                    # The objective stopped improving. With few samples and many
                    # features this usually means the step is too large, so cut
                    # the learning rate first and only stop once it is small.
                    lr = optimizer.param_groups[0]["lr"]
                    if lr > self.learning_rate * 0.01:
                        for group in optimizer.param_groups:
                            group["lr"] = lr * 0.3
                        no_improve = 0
                    else:
                        if self.verbose:
                            print(f"Converged at epoch {epoch + 1}")
                        break

            if self.verbose and (epoch % 10 == 0 or epoch == self.max_iter - 1):
                msg = f"[{epoch+1}/{self.max_iter}] loss={loss_value:.4f}"
                if track_accuracy:
                    msg += f" acc={train_acc:.3f}"
                if X_val is not None:
                    msg += f" val_loss={val_loss:.4f}"
                    if track_accuracy:
                        msg += f" val_acc={val_acc:.3f}"
                print(msg)

        self.n_iter_ = epoch + 1
        self.n_features_in_ = n_features

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Learned graph
        self.edges_ = edges
        self.edge_prior_ = edge_prior
        self.edge_weights_ = model.edge_weights().detach().cpu().numpy().astype(np.float32)
        self.adjacency_ = self._make_adjacency(n_features)

    def _make_adjacency(self, n_features: int):
        w = self.edge_weights_.copy()
        if self.sparsity_threshold > 0:
            w[w < self.sparsity_threshold] = 0.0
        keep = w > 0
        i, j, w = self.edges_[keep, 0], self.edges_[keep, 1], w[keep]

        as_sparse = self.sparse_adjacency
        if as_sparse == "auto":
            as_sparse = n_features > _DENSE_ADJACENCY_MAX_FEATURES

        adj = sp.coo_matrix(
            (np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
            shape=(n_features, n_features),
            dtype=np.float32,
        ).tocsr()
        return adj if as_sparse else adj.toarray()

    def _forward_numpy(self, X) -> torch.Tensor:
        check_is_fitted(self, "coef_")
        X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        device = next(self._model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        self._model.eval()
        with torch.no_grad():
            return self._model(X_t)

    def _seed(self) -> None:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def get_top_edges(self, n: int = 10) -> list[tuple[int, int, float]]:
        """
        Get top n edges by learned weight.

        Returns list of (i, j, weight) tuples.
        """
        check_is_fitted(self, "coef_")
        order = np.argsort(self.edge_weights_)[::-1][:n]
        return [
            (int(self.edges_[k, 0]), int(self.edges_[k, 1]), float(self.edge_weights_[k]))
            for k in order
        ]


# --------------------------------------------------------------------------
# Public estimators
# --------------------------------------------------------------------------

class LogStructClassifier(ClassifierMixin, _BaseLogStruct):
    """
    Network-structured logistic regression with learned feature-graph smoothing.

    Parameters
    ----------
    prior_adjacency : array-like or scipy sparse matrix of shape (n_features, n_features)
        Prior adjacency matrix (e.g., PPI network, pathway co-membership).
        Values should be in [0, 1] representing edge probabilities.

    alpha : float, default=0.5
        Elastic net mixing: 0 = pure L2, 1 = pure L1.

    lambda_en : float, default=1.0
        Elastic net regularization strength. All penalties are divided by the
        number of training samples, so lambda_en=1 matches C=1 in sklearn.
        The L1 part uses a proximal step, so coefficients become exactly zero.

    lambda_smooth : float, default=1.0
        Coefficient smoothness penalty (Laplacian regularization).

    lambda_kl : float, default=1.0
        KL divergence penalty to the prior edge probabilities.

    temperature : float, default=0.8
        Temperature for the edge sigmoid (lower = sharper edges).

    self_loop_weight : float, default=1.0
        Weight for self-connections in smoothing operator.

    neighbor_weight : float, default=1.0
        Weight for neighbor aggregation in smoothing operator.

    edge_threshold : float, default=0.1
        Prior entries above this value become learnable edges. Everything
        else is fixed at zero. Prior builders in logstruct.priors use a
        baseline of 0.05 or lower for non-edges, so the default keeps only
        real prior edges.

    n_candidate_neighbors : int, default=0
        If > 0, each feature also gets candidate edges to its k most
        correlated features in the training data. This lets the model find
        edges that are not in the prior.

    candidate_prior : float, default=0.05
        Prior probability for candidate edges.

    learning_rate : float, default=0.03
        Learning rate for AdamW optimizer.

    max_iter : int, default=500
        Maximum training epochs.

    tol : float, default=1e-4
        Stopping tolerance. Without early stopping, training stops when the
        training objective improves by less than this relative amount for
        n_iter_no_change epochs. With early stopping, it is the minimum
        improvement of the validation loss.

    validation_fraction : float, default=0.1
        Fraction of training data for validation (only with early_stopping).

    early_stopping : bool, default=False
        If True, hold out validation data and stop when its task loss stops
        improving. The default trains on all data until the regularized
        objective converges, like sklearn's LogisticRegression. On data that
        is close to separable the validation loss goes up while accuracy
        still improves, so early stopping can stop much too soon.

    n_iter_no_change : int, default=10
        Patience for both stopping rules.

    scale_features : bool, default=True
        Whether to standardize features before fitting.

    sparsity_threshold : float, default=0.05
        Learned edges below this weight are dropped from adjacency_.

    sparse_adjacency : bool or "auto", default="auto"
        Return adjacency_ as a scipy CSR matrix. "auto" uses a dense array
        up to 5000 features and CSR above that.

    freeze_adjacency : bool, default=False
        If True, don't learn edge weights (use prior directly).

    class_weight : None, "balanced" or dict, default=None
        Class weights for the cross-entropy loss. "balanced" weights each
        class by n_samples / (n_classes * class_count), which helps rare
        cell types.

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

    edges_ : ndarray of shape (n_edges, 2)
        Feature index pairs (i < j) of all learnable edges.

    edge_weights_ : ndarray of shape (n_edges,)
        Learned edge probabilities.

    edge_prior_ : ndarray of shape (n_edges,)
        Prior edge probabilities.

    n_prior_edges_ : int
        Number of edges taken from the prior. Edges after this index in
        edges_ are data-driven candidates.

    adjacency_ : ndarray or csr_matrix of shape (n_features, n_features)
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
        edge_threshold: float = 0.1,
        n_candidate_neighbors: int = 0,
        candidate_prior: float = 0.05,
        learning_rate: float = 0.03,
        max_iter: int = 500,
        tol: float = 1e-4,
        validation_fraction: float = 0.1,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        scale_features: bool = True,
        sparsity_threshold: float = 0.05,
        sparse_adjacency: bool | str = "auto",
        freeze_adjacency: bool = False,
        class_weight: str | dict | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            prior_adjacency,
            alpha=alpha,
            lambda_en=lambda_en,
            lambda_smooth=lambda_smooth,
            lambda_kl=lambda_kl,
            temperature=temperature,
            self_loop_weight=self_loop_weight,
            neighbor_weight=neighbor_weight,
            edge_threshold=edge_threshold,
            n_candidate_neighbors=n_candidate_neighbors,
            candidate_prior=candidate_prior,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            validation_fraction=validation_fraction,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            scale_features=scale_features,
            sparsity_threshold=sparsity_threshold,
            sparse_adjacency=sparse_adjacency,
            freeze_adjacency=freeze_adjacency,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.class_weight = class_weight

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
        self._seed()

        X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_encoded = np.searchsorted(self.classes_, y)

        if self.early_stopping and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            # Stratify when every class has at least 2 samples.
            stratify = y_encoded if np.bincount(y_encoded).min() >= 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded,
                test_size=self.validation_fraction,
                stratify=stratify,
                random_state=self.random_state,
            )
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None

        weight = None
        if self.class_weight is not None:
            present = np.unique(y_train)
            w_present = compute_class_weight(
                self.class_weight if self.class_weight == "balanced"
                else {np.searchsorted(self.classes_, k): v for k, v in self.class_weight.items()},
                classes=present,
                y=y_train,
            )
            w = np.ones(n_classes, dtype=np.float32)
            w[present] = w_present
            weight = torch.tensor(w, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight)

        self._fit_loop(
            X_train, y_train, X_val, y_val,
            output_dim=n_classes,
            criterion=criterion,
            y_dtype=torch.long,
            track_accuracy=True,
        )

        self.coef_ = self._model.linear.weight.detach().cpu().numpy()
        self.intercept_ = self._model.linear.bias.detach().cpu().numpy()
        if n_classes == 2:
            self.coef_ = self.coef_[1] - self.coef_[0]
            self.intercept_ = self.intercept_[1] - self.intercept_[0]

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        logits = self._forward_numpy(X)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]


class LogStructRegressor(RegressorMixin, _BaseLogStruct):
    """
    Network-structured linear regression with learned feature-graph smoothing.

    Same parameters as LogStructClassifier (except class_weight), but for
    regression tasks. See LogStructClassifier for full parameter documentation.

    Attributes
    ----------
    coef_ : ndarray of shape (n_targets, n_features) or (n_features,)
        Learned coefficients.

    intercept_ : ndarray of shape (n_targets,) or float
        Learned intercepts.

    edges_, edge_weights_, edge_prior_ : ndarray
        Learnable edges, their learned weights and their prior weights.

    adjacency_ : ndarray or csr_matrix of shape (n_features, n_features)
        Learned adjacency matrix.
    """

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
        self._seed()

        X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self._squeeze_output = True
        else:
            self._squeeze_output = False

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

        self._fit_loop(
            X_train, y_train, X_val, y_val,
            output_dim=y.shape[1],
            criterion=nn.MSELoss(),
            y_dtype=torch.float32,
            track_accuracy=False,
        )

        self.coef_ = self._model.linear.weight.detach().cpu().numpy()
        self.intercept_ = self._model.linear.bias.detach().cpu().numpy()
        if self._squeeze_output:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = float(self.intercept_[0])

        return self

    def predict(self, X):
        """Predict target values."""
        preds = self._forward_numpy(X).cpu().numpy()
        if self._squeeze_output:
            preds = preds.ravel()
        return preds
