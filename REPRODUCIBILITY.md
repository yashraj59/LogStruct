# Reproducibility record

Date: 2026-05-20

Branch: `experiments/v1`

## Environment

Setup used in this workspace:

```
sudo apt-get install -y python3.10-venv python3-pip
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-research.txt
```

The initial PyTorch wheel was `2.12.0+cu130`, which was incompatible with the
host NVIDIA driver. The venv was changed to `torch==2.7.1+cu126`, after which
`torch.cuda.is_available()` returned true on the NVIDIA L40S.

## Verified commands

```
.venv/bin/python -m pytest test_model.py -q
.venv/bin/python experiments/00_smoke/run.py
.venv/bin/python experiments/07_ablations/kl_scaling_sensitivity.py
.venv/bin/python experiments/01_brca_pam50/run.py --max-genes 1000 --methods dummy_most_frequent logistic_l2 nslr_string_700 logstruct_string_700 --device cuda
.venv/bin/python experiments/01_brca_pam50/make_outputs.py
.venv/bin/python experiments/06_norman_perturb_identity/run.py --max-outer 1 --max-genes 500 --methods dummy_most_frequent logistic_l2 logstruct_identity logstruct_string_700 --device cuda
.venv/bin/python experiments/06_norman_perturb_identity/make_outputs.py
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

Downloaded raw data:

- `data/raw/uci_gene_expression_cancer_rna_seq.zip`
  - SHA256: `06bbb28393ed85b4365f461f20ad75996f2579af4a047945c038ec27386e3bfb`
- `data/raw/brca_metabric.tar.gz`
  - Source: `https://datahub.assets.cbioportal.org/brca_metabric.tar.gz`
  - SHA256: `6d4683477d6b37a2d7edbedc0df610f67bc456f99e5e1bef6219f37b633a55f7`
- `data/raw/9606.protein.links.v12.0.txt.gz`
  - SHA256: `3e22f32572211aa341d5b4bd08d30c32e693e294603202120936872f87719d4f`
- `data/raw/9606.protein.aliases.v12.0.txt.gz`
  - SHA256: `b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680`
- `data/raw/norman_2019_adata.h5ad`
  - Source: Figshare article 24688110, file 43390776
  - SHA256: `0f4831cd80a3786a52f17f91d5d56ee2594645b33c083a824076c468fa5b3913`
  - MD5 verified against Figshare: `95cecb60395536aaa029573390d12d8e`
- `data/raw/tabula_sapiens_v2/tabula_sapiens___blood.h5ad`
  - SHA256: `57aa470d1aea87ee409199c1a424f037fd7af5eeac9e5049fd1dafc9a7251429`
- `data/raw/tabula_sapiens_v2/tabula_sapiens___spleen.h5ad`
  - SHA256: `2b510d552dc8b1e78f78a45433d81be4c38dd88ecc3db89cf9488c675c5f8714`
- `data/raw/tabula_sapiens_v2/tabula_sapiens___lymph_node.h5ad`
  - SHA256: `2461260870236be59f270963024cbd3589eb56a988cb13bb9e1868bac24af065`

The authoritative checksum file is `data/raw/CHECKSUMS.txt`.

## Logged runs

- `results/smoke/results.json`
- `results/audit/kl_reduction_sensitivity.json`
- `results/metabric_pam50/results.json`
- `results/norman_2019_perturb_identity/results.json`

The smoke and audit runs are synthetic pipeline checks only. The METABRIC run is
a real 5-fold 1,000-gene experiment. The Norman run is a one-fold 500-gene
pilot and should not be treated as a final estimate.
