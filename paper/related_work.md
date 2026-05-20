# Related work

LogStruct belongs to a long line of methods that use biological networks to
regularize high-dimensional genomic prediction. The closest prior work is not
graph neural networks, but sparse linear modeling with fixed graph penalties.
The novelty to test empirically is narrower: LogStruct uses a learnable
Bernoulli-parameterized adjacency, anchored to a biological prior by KL
regularization, as an explicit feature-smoothing operator inside a linear model.

## Sparse linear models with fixed graphs

Li and Li introduced network-constrained regularization for genomic regression,
using a graph Laplacian penalty to encourage connected genes to have similar
coefficients in high-dimensional genomic prediction. The Bioinformatics article
frames the problem as variable selection and prediction from gene-expression
data with graphical structure, and explicitly uses the graph Laplacian as a
network-constraint penalty [Li and Li 2008](https://academic.oup.com/bioinformatics/article/24/9/1175/206444).

The closest logistic-regression comparator is the BMC Genomics Logit-Lapnet
work by Zhang et al. The paper proposes graph Laplacian-regularized logistic
regression for disease classification and pathway association, compares against
lasso and elastic net in simulations, and validates the approach on TCGA breast
cancer subtype data using BioGRID protein-protein interactions
[Zhang et al. 2013](https://pubmed.ncbi.nlm.nih.gov/24564637/). This is the
most important like-for-like baseline because it keeps the graph fixed and only
penalizes coefficients.

`glmgraph` generalizes this family into an R package for sparse linear and
logistic regression with graph-constrained regularization. It combines L1 or
MCP sparsity with a Laplacian coefficient-smoothing penalty and solves the
objective by coordinate descent [Chen et al. 2015](https://pubmed.ncbi.nlm.nih.gov/26315909/).
For this project, the appropriate comparison is a PyTorch reimplementation of
the objective, not a direct dependency on the R package.

GELnet provides the broadest fixed-graph framing. Sokolov et al. define a
generalized elastic net that incorporates pathway information through a
feature-feature penalty matrix `P`, including graph Laplacian and related
network penalties. Their PLOS Computational Biology paper argues that pathway
regularization can steer models toward mechanistically linked genes and applies
the method to drug-response prediction in breast cancer cell lines
[Sokolov et al. 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004790).
LogStruct differs by learning the adjacency used in the smoothing operator,
whereas GELnet treats the feature-feature penalty matrix as fixed.

## Learned or adaptive graph methods in linear models

Recent adaptive graph and hypergraph methods move closer to LogStruct's goal
of letting structure respond to data. Jin et al. propose an adaptive hypergraph
regularized logistic regression model for cancer classification and gene
selection, using established biological knowledge, statistical information in
the gene data, adaptive penalties, and constraints on correlated gene pairs
[Jin et al. 2024](https://link.springer.com/article/10.1007/s10489-024-05304-5).
This is an important contemporary comparator, but it is not the same modeling
choice: LogStruct uses a pairwise Bernoulli adjacency with a KL anchor and an
explicit feature-smoothing operator, not a hypergraph objective solved by block
coordinate descent.

The empirical paper should avoid presenting LogStruct as a new paradigm. It is
better described as an incremental, testable combination: graph-constrained
linear modeling plus learned graph probabilities plus a prior-preserving KL
term. The core empirical question is whether this extra graph flexibility helps
relative to fixed-graph Laplacian baselines without destroying interpretability.

## Graph neural networks

GCNs and GATs are natural nonlinear baselines when a biological graph is
available. Kipf and Welling introduced a scalable graph convolutional network
based on a localized first-order spectral approximation
[Kipf and Welling 2017](https://arxiv.org/abs/1609.02907). Velickovic et al.
introduced graph attention networks, which learn attention weights over
neighbors rather than using fixed graph convolution weights
[Velickovic et al. 2018](https://arxiv.org/abs/1710.10903).

These models are conceptually different from LogStruct. In omics settings,
samples are patients or cells and features are genes; a GNN baseline must decide
whether graph nodes are genes, samples, or a bipartite construction. A gene-graph
GCN can use the PPI graph to transform gene features, but it generally loses the
single effective coefficient vector that makes LogStruct easy to inspect.

DIAL-GNN is the most conceptually similar neural method because it jointly
learns graph structure and graph embeddings end to end
[Chen et al. 2019](https://arxiv.org/abs/1912.07832). The distinction is that
DIAL-GNN is a learned-graph GNN, while LogStruct is a learned-graph linear model
with a direct coefficient interpretation.

## Classical baselines

The evaluation must include strong non-graph baselines: unregularized logistic
or linear regression, lasso, ridge, elastic net, random forest, gradient boosted
trees, and a parameter-matched MLP. Elastic net is particularly important
because graph methods often claim biological structure advantages in settings
where ordinary correlated-feature shrinkage is already strong. XGBoost and
LightGBM are not interpretable in the same way, but if they outperform
LogStruct, that result should be reported plainly.

## Positioning statement

LogStruct sits between Li and Li/Logit-Lapnet/glmgraph/GELnet fixed-graph
regularization and DIAL-GNN-style learned graph neural networks. Its specific
contribution is a learnable Bernoulli-parameterized adjacency with a KL anchor
to a biological prior, used as an explicit feature-smoothing operator inside a
linear sklearn-compatible estimator. This preserves a single effective
coefficient vector while allowing the graph to deviate from noisy or incomplete
biological priors. That is a modest but biologically motivated contribution,
and the experiments should test exactly that claim.
