"""
LogStruct: Network-structured regression for biological data.

A scikit-learn compatible library for logistic and linear regression
with learned feature-graph smoothing, designed for omics data with
biological network priors.

Examples
--------
>>> from logstruct import LogStructClassifier
>>> from logstruct.priors import from_pathway_gmt
>>>
>>> # Build prior from pathway database
>>> prior = from_pathway_gmt("c2.cp.kegg.v7.gmt", gene_names)
>>>
>>> # Fit classifier
>>> clf = LogStructClassifier(prior_adjacency=prior, lambda_smooth=1.0)
>>> clf.fit(X_train, y_train)
>>>
>>> # Predictions
>>> y_pred = clf.predict(X_test)
>>>
>>> # Examine learned network
>>> top_edges = clf.get_top_edges(n=20)
>>> learned_adj = clf.adjacency_
"""

__version__ = "0.1.0"

from .model import LogStructClassifier, LogStructRegressor
from . import priors
from . import viz
from . import analysis
from .utils import pick_device

__all__ = [
    "LogStructClassifier",
    "LogStructRegressor",
    "priors",
    "viz",
    "analysis",
    "pick_device",
    "__version__",
]
