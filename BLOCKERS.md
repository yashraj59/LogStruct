# Blockers and issue log

Date: 2026-05-19

## Open blockers

1. Full empirical campaign not yet run.

   The requested bulk, single-cell, perturb-seq, drug-response, biological
   validation, and paper-figure campaign requires downloading large third-party
   datasets and running many cross-validated baselines. This branch now has a
   strict scaffold and smoke-tested harness, but no real-data result should be
   claimed yet.

2. Raw datasets are absent.

   Missing raw files include METABRIC, UCSC Xena/TCGA or UCI pan-cancer,
   GDSC expression/response, GTEx expression/annotations, Tabula Sapiens H5ADs,
   Norman 2019 H5AD, STRING links/aliases, MSigDB GMTs, and DoRothEA regulons.
   The runner intentionally fails rather than silently substituting datasets.

   Additional check: the specified METABRIC S3 tarball
   `https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz` returned
   HTTP 403 from this environment on 2026-05-20. The current cBioPortal Datahub
   documentation recommends downloading study folders from the GitHub datahub
   repository with git-lfs, or using the cBioPortal dataset download interface.
   I did not silently switch the METABRIC source in the experiment config.

3. GPU is unavailable to PyTorch in this environment.

   PyTorch reports that the host NVIDIA driver is too old for the installed CUDA
   runtime. All verified runs are CPU-only. A full campaign should use a
   compatible CUDA driver or a CPU-only torch wheel.

4. GitHub issue creation through `gh` is blocked by tooling.

   `gh` is not installed. Issues were opened through the GitHub REST API
   instead: #1 through #5 on `yashraj59/LogStruct`.

## Local issues corresponding to audit findings

1. KL scaling grows as `O(p^2)`.

   Status: partially addressed. Added `kl_reduction` with `offdiag_mean`, but
   preserved the legacy default `sum`. Real experiments must compare reductions
   before recommending a default.

2. Dense adjacency scaling ceiling.

   Status: unresolved design limitation. `S(A)` and KL both materialize dense
   `p x p` matrices. Scaling experiments must quantify memory and runtime.

3. Dataset-specific parsers are incomplete.

   Status: unresolved implementation work. The current code provides strict
   registry/download/split/preprocessing utilities, but full parsers for every
   requested source still need to be implemented and tested against downloaded
   raw files.
