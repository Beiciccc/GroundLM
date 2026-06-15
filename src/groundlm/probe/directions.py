"""Stage 2 — two-axis probing primitives.

A "direction" is a unit vector in residual-stream space along which a binary
notion (context-faithfulness or parametric-factuality) is most linearly
decodable. We support two estimators:

* mass-mean (difference-of-means, a la Marks & Tegmark 2024): robust, label-
  polarity-signed, the primary estimator for the geometry/angle analyses.
* L2-regularized logistic probe: a discriminative alternative for AUROC and
  for the gate's decision scores.

All directions are returned L2-normalized. Probe *scores* are signed scalar
projections onto a direction.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ArrayF = np.ndarray


def _unit(v: ArrayF) -> ArrayF:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def mass_mean_direction(X: ArrayF, y: ArrayF) -> ArrayF:
    """Difference of class means, L2-normalized. y in {0,1}."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    mu1 = X[y == 1].mean(axis=0)
    mu0 = X[y == 0].mean(axis=0)
    return _unit(mu1 - mu0)


def logistic_direction(X: ArrayF, y: ArrayF, C: float = 1.0) -> ArrayF:
    """Coefficient vector of an L2-logistic probe, L2-normalized."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X, y)
    return _unit(clf.coef_.ravel())


def fit_direction(X: ArrayF, y: ArrayF, method: str = "mass_mean", **kw) -> ArrayF:
    if method == "mass_mean":
        return mass_mean_direction(X, y)
    if method == "logistic":
        return logistic_direction(X, y, **kw)
    raise ValueError(f"unknown method {method!r}")


def project(X: ArrayF, d: ArrayF) -> ArrayF:
    """Signed scalar projection of each row of X onto unit direction d."""
    return np.asarray(X, dtype=np.float64) @ _unit(np.asarray(d, dtype=np.float64))


def probe_auroc(X: ArrayF, y: ArrayF, d: ArrayF) -> float:
    """AUROC of the signed projection onto d as a classifier for y.

    Direction polarity is resolved by taking max(auc, 1-auc) so a sign flip in
    the estimated axis does not register as a failure.
    """
    s = project(X, d)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    auc = roc_auc_score(y, s)
    return float(max(auc, 1.0 - auc))


def cv_auroc(X: ArrayF, y: ArrayF, method: str = "mass_mean", folds: int = 5,
             seed: int = 0, **kw) -> float:
    """Cross-validated probe AUROC (fit direction on train, score on test)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    splits = np.array_split(idx, folds)
    aucs = []
    for k in range(folds):
        te = splits[k]
        tr = np.concatenate([splits[j] for j in range(folds) if j != k])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        d = fit_direction(X[tr], y[tr], method=method, **kw)
        aucs.append(probe_auroc(X[te], y[te], d))
    return float(np.mean(aucs)) if aucs else float("nan")
