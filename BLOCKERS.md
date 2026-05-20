# Blockers and issue log

Date: 2026-05-20

## Open blockers

1. Full empirical campaign not yet run.

   METABRIC has one real 5-fold 1,000-gene run, GTEx has one real 5-fold
   500-gene run, Norman has one real 5-fold 500-gene run, and Tabula Sapiens has
   one real 5-fold donor-held-out 500-gene LogStruct run with a class cap. The
   requested full 5k-10k bulk feature campaign, tuned Tabula logistic/celltypist
   baselines, drug-response, TCGA/Xena, ablations, and biological validation
   remain incomplete.

2. Some raw datasets are still absent.

   Downloaded and checksummed: METABRIC, UCI pan-cancer shakedown, STRING v12
   links/aliases, Norman 2019 labeled H5AD, Tabula Sapiens v2 Blood/Spleen/Lymph
   Node H5ADs, GTEx v8 expression/annotations, and GDSC2 dose-response.

   Still missing: UCSC Xena full Pan-Cancer Atlas expression, GDSC expression,
   MSigDB licensed GMT downloads, and DoRothEA regulons. The runner should
   continue to fail rather than silently substituting datasets.

   Additional check: the specified METABRIC S3 tarball
   `https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz` returned
   HTTP 403 from this environment on 2026-05-20. The current cBioPortal Datahub
   documentation recommends downloading study folders from the GitHub datahub
   repository with git-lfs, or using the cBioPortal dataset download interface.
   The registry now uses the working official cBioPortal asset host
   `https://datahub.assets.cbioportal.org/brca_metabric.tar.gz`; source metadata
   is recorded under `results/source_metadata/`.

   Additional source checks on 2026-05-20:

   - UCSC Xena Pan-Cancer expression redirected to S3 and returned HTTP 403 from
     this environment, while the survival supplemental phenotype file returned
     HTTP 200.
   - The old GDSC expression URL
     `https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip`
     returned HTTP 410. The GDSC2 fitted dose-response XLSX did download from
     the release 8.5 Sanger URL.

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

   Status: partially addressed. METABRIC, GTEx, and Norman parsers are
   implemented and tested against downloaded raw files. Tabula Sapiens now has a
   custom sparse donor-held-out runner with train-fold gene selection, but still
   lacks the full baseline suite. TCGA/Xena and GDSC parsers remain absent.
