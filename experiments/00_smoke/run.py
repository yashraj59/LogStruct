"""Synthetic smoke test used before any real biological experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.shared.eval import (  # noqa: E402
    RunRecord,
    Timer,
    classification_metrics,
    environment_record,
    git_sha,
    write_run_record,
)
from logstruct import LogStructClassifier  # noqa: E402
from logstruct.priors import from_random  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((cfg["n_samples"], cfg["n_features"])).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    prior = from_random(cfg["n_features"], density=0.1, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=seed
    )

    model_cfg = dict(cfg["model"])
    model = LogStructClassifier(prior_adjacency=prior, random_state=seed, **model_cfg)
    with Timer() as timer:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = classification_metrics(y_test, y_pred, y_proba)
    metrics["n_iter"] = int(model.n_iter_)
    metrics["final_train_loss"] = float(model.history_["train_loss"][-1])
    if model.history_.get("val_loss"):
        metrics["final_val_loss"] = float(model.history_["val_loss"][-1])

    record = RunRecord(
        dataset_name=cfg["dataset_name"],
        method="LogStructClassifier",
        seed=seed,
        train_size=len(X_train),
        val_size=int(round(len(X_train) * model.validation_fraction)),
        test_size=len(X_test),
        hyperparameters=model_cfg,
        metrics=metrics,
        wall_clock_sec=timer.elapsed,
        device=model_cfg.get("device", "cpu"),
        git_sha=git_sha(),
        environment=environment_record(),
    )
    write_run_record(record, ROOT / "results" / "smoke" / "results.json")


if __name__ == "__main__":
    main()
