"""Stage 4 — confidence disentanglement and the axis-specific asymmetry (C2).

Hypothesis: softmax confidence and activation norm absorb the PARAMETRIC-
FACTUALITY axis far more than the CONTEXT-FAITHFULNESS axis, so faithfulness is
the confidence-independent residual.

We "purge" confidence by residualizing every hidden dimension on a set of
confound regressors (max-softmax, answer log-prob, activation L2 norm, answer
length, context-answer lexical overlap), then refit each probe on the residual
features. "Absorption" for an axis is the share of its above-chance probe AUROC
that disappears under purging:

    absorbed = (auroc_raw - auroc_purged) / (auroc_raw - 0.5)

The headline test is the ASYMMETRY: absorbed(factuality) - absorbed(faithfulness),
with a bootstrap CI and a one-sided p-value for "> 0".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .directions import cv_auroc, fit_direction, project

ArrayF = np.ndarray


def purge(X: ArrayF, confounds: ArrayF) -> ArrayF:
    """Residualize each column of X on the confounds via OLS (with intercept).

    Returns X - C @ beta, i.e. the part of the representation not linearly
    explained by [1, confounds]. Both train and apply on the same split here;
    for the gate (Stage 6) fit beta on calibration data and apply out-of-sample.
    """
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(confounds, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    # standardize confounds for numerical stability, prepend intercept
    Cz = (C - C.mean(0)) / (C.std(0) + 1e-8)
    A = np.concatenate([np.ones((len(Cz), 1)), Cz], axis=1)
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta


def _proj_r2(s: ArrayF, confounds: ArrayF) -> float:
    """Adjusted R^2 of the axis projection ``s`` explained by [1, confounds].

    Adjusted (rather than raw) R^2 removes the upward bias from fitting p
    regressors on finite n, so the metric is comparable across sample sizes
    (e.g. between the full sample and bootstrap subsamples) and not inflatable
    by adding confounds.
    """
    C = np.asarray(confounds, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    n, p = len(s), C.shape[1]
    Cz = (C - C.mean(0)) / (C.std(0) + 1e-8)
    A = np.concatenate([np.ones((n, 1)), Cz], axis=1)
    beta, *_ = np.linalg.lstsq(A, s, rcond=None)
    pred = A @ beta
    ss_res = float(((s - pred) ** 2).sum())
    ss_tot = float(((s - s.mean()) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    if n - p - 1 > 0:
        r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    return float(max(0.0, r2))


def absorbed_fraction(X: ArrayF, y: ArrayF, confounds: ArrayF, *,
                      method: str = "mass_mean", folds: int = 5, seed: int = 0) -> dict:
    """How much of an axis is "confidence".

    Primary metric ``absorbed`` = R^2 of the axis projection explained by the
    confounds (max-softmax, log-prob, activation norm, length, lexical overlap).
    This directly measures the share of the axis coordinate that confidence can
    reproduce and — unlike a purge-then-refit AUROC delta — is not confounded by
    the *denoising* side effect of residualizing on activation norm. We also
    report raw/purged AUROC as secondary diagnostics.
    """
    d = fit_direction(X, y, method=method)
    s = project(X, d)
    absorbed = _proj_r2(s, confounds)
    auc_raw = cv_auroc(X, y, method=method, folds=folds, seed=seed)
    auc_purged = cv_auroc(purge(X, confounds), y, method=method, folds=folds, seed=seed)
    return {
        "absorbed": absorbed,
        "conf_explained_r2": absorbed,
        "auroc_raw": auc_raw,
        "auroc_purged": auc_purged,
    }


@dataclass
class AsymmetryResult:
    faith: dict
    fact: dict
    asymmetry: float          # absorbed(fact) - absorbed(faith)
    asymmetry_ci: tuple
    p_value: float            # one-sided P(asymmetry <= 0) under bootstrap
    n: int

    def supports_hypothesis(self, alpha: float = 0.05) -> bool:
        return self.asymmetry > 0 and self.p_value < alpha


def asymmetry_test(
    X: ArrayF,
    faith_y: ArrayF,
    fact_y: ArrayF,
    confounds: ArrayF,
    *,
    method: str = "mass_mean",
    n_boot: int = 200,
    seed: int = 0,
) -> AsymmetryResult:
    X = np.asarray(X, dtype=np.float64)
    faith_y = np.asarray(faith_y).astype(int)
    fact_y = np.asarray(fact_y).astype(int)
    confounds = np.asarray(confounds, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(faith_y)

    faith = absorbed_fraction(X, faith_y, confounds, method=method, seed=seed)
    fact = absorbed_fraction(X, fact_y, confounds, method=method, seed=seed)
    asym = fact["absorbed"] - faith["absorbed"]

    # SUBSAMPLE without replacement: avoids duplicate rows leaking across the
    # cross-validation folds inside absorbed_fraction (which would inflate AUROC
    # and decouple the bootstrap distribution from the point estimate).
    def _absorbed_only(Xs, ys, cs):
        return _proj_r2(project(Xs, fit_direction(Xs, ys, method=method)), cs)

    m = max(20, int(0.8 * n))
    boot = []
    for b in range(n_boot):
        idx = rng.permutation(n)[:m]
        if len(np.unique(faith_y[idx])) < 2 or len(np.unique(fact_y[idx])) < 2:
            continue
        boot.append(_absorbed_only(X[idx], fact_y[idx], confounds[idx])
                    - _absorbed_only(X[idx], faith_y[idx], confounds[idx]))
    boot = np.asarray(boot)
    if boot.size:
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        p = float((boot <= 0).mean())
    else:
        ci, p = (float("nan"), float("nan")), float("nan")

    return AsymmetryResult(faith=faith, fact=fact, asymmetry=float(asym),
                           asymmetry_ci=ci, p_value=p, n=n)
