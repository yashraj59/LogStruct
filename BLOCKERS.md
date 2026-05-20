# Blockers and issue log

Date: 2026-05-19

## Open blockers

1. Full empirical campaign not yet run.

   METABRIC has one real 5-fold 1,000-gene run and Norman has one real
   single-fold 500-gene pilot. The requested full 5k-10k bulk feature campaign,
   full Norman 5-fold run, Tabula Sapiens preprocessing/training, drug-response,
   GTEx, TCGA/Xena, ablations, and biological validation remain incomplete.

2. Some raw datasets are still absent.

   Downloaded and checksummed: METABRIC, UCI pan-cancer shakedown, STRING v12
   links/aliases, Norman 2019 labeled H5AD, and Tabula Sapiens v2 Blood,
   Spleen, and Lymph Node H5ADs.

   Still missing: UCSC Xena full Pan-Cancer Atlas, GDSC expression/response,
   GTEx expression/annotations, MSigDB licensed GMT downloads, and DoRothEA
   regulons. The runner should continue to fail rather than silently substituting
   datasets.

   Additional check: the specified METABRIC S3 tarball
   `https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz` returned
   HTTP 403 from this environment on 2026-05-20. The current cBioPortal Datahub
   documentation recommends downloading study folders from the GitHub datahub
   repository with git-lfs, or using the cBioPortal dataset download interface.
   The registry now uses the working official cBioPortal asset host
   `https://datahub.assets.cbioportal.org/brca_metabric.tar.gz`; source metadata
   is recorded under `results/source_metadata/`.

3. GPU environment was repaired but should remain pinned.

   The initial `torch==2.12.0+cu130` wheel could not use the L40S with the
   installed driver. Reinstalling `torch==2.7.1+cu126` fixed CUDA availability.
   Keep the CUDA 12.6 wheel pinned for this host.

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

   Status: partially addressed. METABRIC and Norman parsers are implemented and
   tested against downloaded raw files. Tabula Sapiens H5ADs open in backed mode
   and expose `donor_id` and `cell_type`, but donor-held-out preprocessing is not
   yet implemented. TCGA/Xena, GDSC, and GTEx parsers remain absent.
