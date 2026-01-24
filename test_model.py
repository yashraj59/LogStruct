"""
Tests for LogStruct.
"""

import numpy as np
import pytest

from logstruct import LogStructClassifier, LogStructRegressor
from logstruct import priors


class TestPriors:
    """Test prior builders."""

    def test_from_random(self):
        adj = priors.from_random(100, density=0.1, random_state=42)
        
        assert adj.shape == (100, 100)
        assert np.allclose(adj, adj.T)  # symmetric
        assert np.allclose(np.diag(adj), 0)  # no self-loops
        assert adj.min() >= 0 and adj.max() <= 1

    def test_from_identity(self):
        adj = priors.from_identity(50, baseline=0.01)
        
        assert adj.shape == (50, 50)
        assert np.allclose(np.diag(adj), 0)
        assert np.allclose(adj[0, 1], 0.01)

    def test_from_edge_list(self):
        edges = [(0, 1), (1, 2, 0.9), (0, 3)]
        adj = priors.from_edge_list(5, edges, default_weight=0.7)
        
        assert adj.shape == (5, 5)
        assert adj[0, 1] == 0.7
        assert adj[1, 2] == 0.9
        assert adj[0, 3] == 0.7
        assert np.allclose(adj, adj.T)

    def test_from_correlation(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 20))
        
        adj = priors.from_correlation(X, threshold=0.2)
        
        assert adj.shape == (20, 20)
        assert np.allclose(adj, adj.T)
        assert np.allclose(np.diag(adj), 0)

    def test_summarize(self):
        adj = priors.from_random(50, density=0.1, random_state=42)
        stats = priors.summarize(adj)
        
        assert stats["n_nodes"] == 50
        assert stats["n_edges"] > 0
        assert 0 <= stats["density"] <= 1
        assert stats["mean_weight"] > 0


class TestClassifier:
    """Test LogStructClassifier."""

    @pytest.fixture
    def simple_data(self):
        rng = np.random.default_rng(42)
        n, p = 200, 50
        X = rng.standard_normal((n, p))
        y = (X[:, 0] + X[:, 1] + 0.5 * rng.standard_normal(n) > 0).astype(int)
        prior = priors.from_random(p, density=0.1, random_state=42)
        return X, y, prior

    def test_fit_predict(self, simple_data):
        X, y, prior = simple_data
        
        clf = LogStructClassifier(
            prior_adjacency=prior,
            max_iter=20,
            random_state=42,
            verbose=False,
        )
        clf.fit(X, y)
        
        # Check fitted attributes
        assert hasattr(clf, "coef_")
        assert hasattr(clf, "intercept_")
        assert hasattr(clf, "adjacency_")
        assert hasattr(clf, "classes_")
        
        # Predictions
        y_pred = clf.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(y_pred).issubset(set(clf.classes_))
        
        # Probabilities
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1)

    def test_sklearn_score(self, simple_data):
        X, y, prior = simple_data
        
        clf = LogStructClassifier(
            prior_adjacency=prior,
            max_iter=50,
            random_state=42,
        )
        clf.fit(X, y)
        
        score = clf.score(X, y)
        assert 0.5 < score <= 1.0  # should do better than random

    def test_get_top_edges(self, simple_data):
        X, y, prior = simple_data
        
        clf = LogStructClassifier(prior_adjacency=prior, max_iter=20, random_state=42)
        clf.fit(X, y)
        
        edges = clf.get_top_edges(n=5)
        
        assert len(edges) == 5
        assert all(len(e) == 3 for e in edges)
        assert all(e[2] >= 0 for e in edges)
        # Should be sorted descending
        weights = [e[2] for e in edges]
        assert weights == sorted(weights, reverse=True)

    def test_multiclass(self):
        rng = np.random.default_rng(42)
        n, p = 300, 30
        X = rng.standard_normal((n, p))
        y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0).astype(int)  # 3 classes
        prior = priors.from_random(p, density=0.15, random_state=42)
        
        clf = LogStructClassifier(
            prior_adjacency=prior,
            max_iter=30,
            random_state=42,
        )
        clf.fit(X, y)
        
        assert len(clf.classes_) == 3
        proba = clf.predict_proba(X)
        assert proba.shape == (n, 3)


class TestRegressor:
    """Test LogStructRegressor."""

    @pytest.fixture
    def regression_data(self):
        rng = np.random.default_rng(42)
        n, p = 200, 40
        X = rng.standard_normal((n, p))
        y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + 0.3 * rng.standard_normal(n)
        prior = priors.from_random(p, density=0.1, random_state=42)
        return X, y, prior

    def test_fit_predict(self, regression_data):
        X, y, prior = regression_data
        
        reg = LogStructRegressor(
            prior_adjacency=prior,
            max_iter=30,
            random_state=42,
            verbose=False,
        )
        reg.fit(X, y)
        
        assert hasattr(reg, "coef_")
        assert hasattr(reg, "intercept_")
        assert hasattr(reg, "adjacency_")
        
        y_pred = reg.predict(X)
        assert y_pred.shape == (len(X),)

    def test_sklearn_score(self, regression_data):
        X, y, prior = regression_data
        
        reg = LogStructRegressor(
            prior_adjacency=prior,
            max_iter=50,
            random_state=42,
        )
        reg.fit(X, y)
        
        r2 = reg.score(X, y)
        assert r2 > 0.3  # should explain some variance

    def test_multitarget(self):
        rng = np.random.default_rng(42)
        n, p = 150, 25
        X = rng.standard_normal((n, p))
        y = np.column_stack([
            X[:, 0] + 0.1 * rng.standard_normal(n),
            -X[:, 1] + 0.1 * rng.standard_normal(n),
        ])
        prior = priors.from_random(p, density=0.1, random_state=42)
        
        reg = LogStructRegressor(
            prior_adjacency=prior,
            max_iter=30,
            random_state=42,
        )
        reg.fit(X, y)
        
        y_pred = reg.predict(X)
        assert y_pred.shape == (n, 2)


class TestRegularization:
    """Test regularization components."""

    def test_freeze_adjacency(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 20))
        y = (X[:, 0] > 0).astype(int)
        prior = priors.from_random(20, density=0.1, random_state=42)
        
        clf = LogStructClassifier(
            prior_adjacency=prior,
            freeze_adjacency=True,
            max_iter=20,
            random_state=42,
        )
        clf.fit(X, y)
        
        # Learned adjacency should be close to prior (only thresholding differs)
        diff = np.abs(clf.adjacency_ - prior)
        diff[prior < clf.sparsity_threshold] = 0  # ignore thresholded entries
        assert diff.max() < 0.1

    def test_different_lambdas(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 20))
        y = (X[:, 0] > 0).astype(int)
        prior = priors.from_random(20, density=0.1, random_state=42)
        
        # High L1 should give sparser weights
        clf_l1 = LogStructClassifier(
            prior_adjacency=prior,
            alpha=1.0,  # pure L1
            lambda_en=10.0,
            max_iter=50,
            random_state=42,
        )
        clf_l1.fit(X, y)
        
        clf_l2 = LogStructClassifier(
            prior_adjacency=prior,
            alpha=0.0,  # pure L2
            lambda_en=10.0,
            max_iter=50,
            random_state=42,
        )
        clf_l2.fit(X, y)
        
        # L1 should have more near-zero coefficients
        l1_near_zero = np.sum(np.abs(clf_l1.coef_) < 0.01)
        l2_near_zero = np.sum(np.abs(clf_l2.coef_) < 0.01)
        
        assert l1_near_zero >= l2_near_zero


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_prior_shape(self):
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        prior = priors.from_random(30, density=0.1)  # wrong size
        
        clf = LogStructClassifier(prior_adjacency=prior)
        
        with pytest.raises(ValueError, match="prior_adjacency shape"):
            clf.fit(X, y)

    def test_single_sample_predict(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 20))
        y = (X[:, 0] > 0).astype(int)
        prior = priors.from_random(20, density=0.1, random_state=42)
        
        clf = LogStructClassifier(prior_adjacency=prior, max_iter=10, random_state=42)
        clf.fit(X, y)
        
        # Single sample prediction
        y_pred = clf.predict(X[:1])
        assert y_pred.shape == (1,)
