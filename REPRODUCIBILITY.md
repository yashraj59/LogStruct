# Reproducibility record

Date: 2026-05-19

Branch: `experiments/v1`

## Environment

Setup used in this workspace:

```
sudo apt-get install -y python3.10-venv python3-pip
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-research.txt
```

The installed PyTorch wheel is `2.12.0+cu130`, but the host NVIDIA driver is too
old for that CUDA runtime. Verified runs are CPU-only.

## Verified commands

```
.venv/bin/python -m pytest test_model.py -q
.venv/bin/python experiments/00_smoke/run.py
.venv/bin/python experiments/07_ablations/kl_scaling_sensitivity.py
make all
pandoc paper/main.md -o paper/main.pdf
```

Current test result: 21 passed.

## Seeds

- Unit tests: fixed seed 42 where stochastic.
- Smoke test: 42.
- KL scaling sensitivity: 42.
- Planned outer/inner biological splits: 0 unless overridden in dataset config.

## Data versions

Downloaded raw shakedown data:

- `data/raw/uci_gene_expression_cancer_rna_seq.zip`
  - SHA256: `06bbb28393ed85b4365f461f20ad75996f2579af4a047945c038ec27386e3bfb`

The remaining large raw biological datasets have not been downloaded into this
branch yet. Expected raw paths and source URLs are encoded in
`experiments/*/config.yaml` and `experiments/shared/data_loader.py`.

When raw files are downloaded, write checksums to:

```
data/raw/CHECKSUMS.txt
```

## Logged runs

- `results/smoke/results.json`
- `results/audit/kl_reduction_sensitivity.json`

These are synthetic pipeline checks only and must not be used as biological
evidence in the paper.
