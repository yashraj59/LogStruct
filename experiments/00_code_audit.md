# LogStruct code audit

Date: 2026-05-19

Branch: `experiments/v1`

Audited files: `logstruct/model.py`, `logstruct/priors.py`, `logstruct/analysis.py`,
`logstruct/viz.py`, `logstruct/utils.py`, and `test_model.py`.

## 1. Forward-pass math

The internal module `_StructuredModel.forward` computes:

```
A = zero_diag(0.5 * (sigmoid(L / T) + sigmoid(L / T)^T))
S = alpha_self * I + beta_neighbor * rownorm(A)
Y_hat = (X @ S) @ W^T + b
```

where `L` is `adj_logits`, `T` is `temperature`, `W` is
`self.linear.weight`, `alpha_self` is `self_loop_weight`, and
`beta_neighbor` is `neighbor_weight`.

The user-proposed expression was close, but the implementation is more simply
described as symmetrizing the probability matrix after zeroing the diagonal.
There is no explicit `- sigmoid(logits/T) * diag` term in the returned operator;
the diagonal is set to zero before symmetrization.

## 2. Paper framing correction

LogStruct is not only a graph Laplacian penalty on weights. The implementation
has three distinct graph-related mechanisms:

1. Explicit feature smoothing in the forward pass: `X` is replaced by `X @ S(A)`.
2. Laplacian smoothness on raw head weights: `trace(W L_A W^T)`.
3. Bernoulli KL anchoring from learned adjacency probabilities to a prior
   adjacency: `KL(q(A) || p(A_prior))`.

The method section must keep these separate. In particular, learned graph
adaptation changes both the prediction operator and the smoothness penalty.

## 3. Findings

### Fixed in this branch

1. Effective coefficients were missing.

   `coef_` exposed raw `W`, but the equivalent linear coefficients on the
   original feature space are:

   ```
   W_effective = W @ S^T
   ```

   because `X @ S @ W^T = X @ (S @ W^T)`. Added `effective_coef_` to classifier
   and regressor, added `smoothing_operator_`, and updated analysis/visualization
   helpers to prefer effective coefficients.

2. KL prior was temperature-distorted.

   Before this audit, `prior_probs = sigmoid(logit(prior) / temperature)`, so a
   prior entry of 0.7 was not anchored as 0.7 when `temperature != 1`. The fix
   stores the true clipped prior probability and initializes logits as
   `temperature * logit(prior)` so `current_adjacency()` starts at the provided
   prior for any temperature.

3. STRING gene-symbol priors did not resolve aliases.

   `from_string_db` compared STRING protein IDs such as `9606.ENSP...` directly
   to `gene_names`. For ordinary gene symbols this silently produced mostly
   baseline edges. Added optional `alias_path` support for
   `9606.protein.aliases.v12.0.txt.gz` and gzip-aware readers.

4. The trained prediction operator is unthresholded, while `adjacency_` is
   thresholded for interpretation.

   Added `adjacency_raw_` for the unthresholded adjacency used by the model.
   The paper should state that `sparsity_threshold` affects reported adjacency
   only, not predictions.

5. Unit tests added.

   Added tests for the Laplacian trace identity, effective coefficient identity,
   temperature-preserving prior initialization, KL off-diagonal mean reduction,
   and STRING alias parsing.

### Confirmed

1. `smoothness_penalty` equals `trace(W L W^T)`.

   The implementation `(W @ L * W).sum()` is elementwise multiplication followed
   by summation. This is algebraically equal to `trace(W L W^T)` and is now
   covered by a unit test.

2. Diagonal handling is safe in forward passes.

   `current_adjacency()` clones the sigmoid adjacency and sets the diagonal to
   zero before use. This means diagonal logits do not contribute to smoothing,
   smoothness, or KL after clipping. Because optimizer weight decay is zero,
   they should not move materially. The implementation is acceptable, but a
   future cleanup could register a diagonal mask to avoid re-creating indices.

3. Numerical KL clipping is present.

   `q` and `p` are clipped to `[1e-6, 1 - 1e-6]` before KL. No NaNs appeared in
   synthetic smoke tests or the existing unit tests.

### Design issue left configurable

KL scaling remains a core design choice. The legacy behavior sums Bernoulli KL
over all matrix entries, which grows as `O(p^2)`. This branch adds
`kl_reduction={"sum","mean","offdiag_mean"}` while preserving the default
`sum` for backward compatibility. The paper experiments should use and report
`offdiag_mean` unless there is a reason to preserve legacy scaling.

Small synthetic sensitivity run:

| n_features | KL reduction | Accuracy | Final val loss |
|---:|---|---:|---:|
| 50 | sum | 0.84 | 0.622 |
| 50 | offdiag_mean | 0.85 | 0.601 |
| 100 | sum | 0.73 | 0.647 |
| 100 | offdiag_mean | 0.80 | 0.630 |
| 200 | sum | 0.79 | 0.706 |
| 200 | offdiag_mean | 0.86 | 0.705 |

Logged at `results/audit/kl_reduction_sensitivity.json`. This is only a
synthetic scaling check, not a biological result.

## 4. Smoke test

The docstring-style synthetic classification smoke test now runs through
`experiments/00_smoke/run.py` and logs to `results/smoke/results.json`.

Logged result:

- Dataset: `synthetic_docstring_smoke`
- Seed: 42
- Train/test: 400/100
- Features: 100
- Hyperparameters: `lambda_en=0.1`, `lambda_smooth=0.1`, `lambda_kl=1.0`,
  `kl_reduction=offdiag_mean`, `max_iter=50`, `device=cpu`
- Balanced accuracy: 0.773
- Macro-F1: 0.777
- OvR AUROC: 0.965
- Wall clock: 0.865 s

This is the smoke test only. It should not appear as a paper result.

## 5. Verification

Commands run:

```
.venv/bin/python -m pytest test_model.py -q
.venv/bin/python experiments/00_smoke/run.py
.venv/bin/python experiments/07_ablations/kl_scaling_sensitivity.py
make all
```

Current result: 21 tests passed. PyTorch warns that the installed CUDA runtime
cannot use the host NVIDIA driver, so all verified runs are CPU-only.

## 6. Issue tracking

GitHub CLI is not installed, but issues were opened through the GitHub REST API:

- https://github.com/yashraj59/LogStruct/issues/1 - effective coefficients
- https://github.com/yashraj59/LogStruct/issues/2 - temperature-distorted prior
- https://github.com/yashraj59/LogStruct/issues/3 - STRING alias resolution
- https://github.com/yashraj59/LogStruct/issues/4 - KL scaling
- https://github.com/yashraj59/LogStruct/issues/5 - METABRIC download blocker

The unambiguous code defects are fixed in this branch and the remaining design
choices are documented in `BLOCKERS.md`.
