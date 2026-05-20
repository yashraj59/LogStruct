"""Strict entry point for configured real-data experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.shared.data_loader import load_dataset, require_raw  # noqa: E402


def main(config_path: str | Path) -> None:
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    required_raw = [ROOT / p for p in cfg.get("required_raw", [])]
    if required_raw:
        require_raw(required_raw)
    try:
        load_dataset(cfg["dataset_name"])
    except FileNotFoundError as exc:
        blocker = {
            "dataset": cfg["dataset_name"],
            "status": "blocked",
            "reason": str(exc),
            "config": str(config_path),
        }
        out = ROOT / "results" / cfg["dataset_name"] / "BLOCKED.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n")
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("config.yaml"))
