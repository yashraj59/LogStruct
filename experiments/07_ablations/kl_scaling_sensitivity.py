"""Small synthetic check of KL reduction sensitivity."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from logstruct import LogStructClassifier  # noqa: E402
from logstruct.priors import from_random  # noqa: E402


def run_once(kl_reduction: str, n_features: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((500, n_features)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    prior = from_random(n_features, density=0.1, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    clf = LogStructClassifier(
        prior_adjacency=prior,
        lambda_en=0.1,
        lambda_smooth=0.1,
        lambda_kl=1.0,
        kl_reduction=kl_reduction,
        max_iter=50,
        random_state=seed,
        device="cpu",
    )
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    return {
        "kl_reduction": kl_reduction,
        "n_features": n_features,
        "accuracy": float(clf.score(X_test, y_test)),
        "final_train_loss": float(clf.history_["train_loss"][-1]),
        "final_val_loss": float(clf.history_["val_loss"][-1]),
        "wall_clock_sec": time.perf_counter() - start,
    }


def main():
    rows = []
    for n_features in [50, 100, 200]:
        for reduction in ["sum", "offdiag_mean"]:
            rows.append(run_once(reduction, n_features))
    out = ROOT / "results" / "audit" / "kl_reduction_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
