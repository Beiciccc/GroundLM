"""Stage 5 — cross-family transfer of a probe direction (contribution C3).

Claim: only the confidence-PURGED faithfulness direction transfers across model
families; the factuality / confidence-laden component is family-idiosyncratic.

Setup: all models are run on the SAME CtrlPairs statements, so feature rows are
paired across models. On a held-out calibration split we fit a map between two
models' residual spaces, transport a SOURCE-trained probe into the TARGET space,
and measure target AUROC on an eval split the map never saw.

Two maps:
  * orthogonal Procrustes (semi-orthogonal generalization for unequal hidden
    dims, the d_from != d_to case across families) — our method, after Atlas-Alignment;
  * an unconstrained ridge map — ablation.
And the ACS anchor-projection baseline (after Cross-Family Universality): represent
every activation by its cosine similarity to a fixed set of anchor statements, a
model-agnostic coordinate space in which a source probe applies directly.

Convention: ``fit_map(X_from, X_to)`` returns A with ``X_from @ A ~= X_to``. To
evaluate a probe ``d`` trained in the TO space on FROM activations, transport it to
FROM space as ``A @ d`` (since (X_from @ A)·d = X_from·(A@d)).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ..probe.directions import fit_direction, _unit
from ..probe.confidence import purge

ArrayF = np.ndarray


def fit_map(X_from: ArrayF, X_to: ArrayF, orthogonal: bool = True, ridge: float = 1.0) -> ArrayF:
    """Linear map A (d_from x d_to) with X_from @ A ~= X_to (paired rows)."""
    Xf = np.asarray(X_from, dtype=np.float64)
    Xt = np.asarray(X_to, dtype=np.float64)
    Xf = Xf - Xf.mean(0)
    Xt = Xt - Xt.mean(0)
    if orthogonal:
        # semi-orthogonal Procrustes: min ||Xf A - Xt|| via SVD of Xf^T Xt
        M = Xf.T @ Xt                       # (d_from x d_to)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)  # U:(d_from,r) Vt:(r,d_to)
        return U @ Vt                       # (d_from x d_to)
    # ridge-regularized least squares
    d_from = Xf.shape[1]
    G = Xf.T @ Xf + ridge * np.eye(d_from)
    return np.linalg.solve(G, Xf.T @ Xt)


def transport_direction(A: ArrayF, d_to: ArrayF) -> ArrayF:
    """Bring a TO-space direction into FROM space: A @ d_to (unit-normalized)."""
    return _unit(np.asarray(A) @ np.asarray(d_to, dtype=np.float64))


def acs_features(X: ArrayF, anchors: ArrayF) -> ArrayF:
    """Anchor Coordinate Space: cosine similarity of each row to each anchor.

    Anchors are activations of a fixed set of statements; because every model is
    run on the SAME statements, the K anchor coordinates correspond across models,
    giving a shared space without any learned map.
    """
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(anchors, dtype=np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    return Xn @ An.T


def _auroc(scores: ArrayF, y: ArrayF) -> float:
    """Signed AUROC: polarity is fixed by the calibration fold, never by the test fold.

    This used to return max(a, 1-a), which floors a non-predictive direction above 0.5
    and so inflates exactly the near-chance quantities (the map-aware nulls) that the
    transfer analysis relies on. Use _auroc_maxflip only to demonstrate that inflation.
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _auroc_maxflip(scores: ArrayF, y: ArrayF) -> float:
    """The max(AUC, 1-AUC) convention, kept only for the appendix's inflation demo."""
    a = _auroc(scores, y)
    return a if a != a else float(max(a, 1.0 - a))


def purged_direction(X: ArrayF, y: ArrayF, confounds: ArrayF, method="mass_mean") -> ArrayF:
    return fit_direction(purge(X, confounds), y, method=method)


def transfer_auroc(
    X_src: ArrayF, y_src: ArrayF, conf_src: ArrayF,
    X_tgt: ArrayF, y_tgt: ArrayF, conf_tgt: ArrayF,
    *, mode: str = "procrustes_purged", cal_frac: float = 0.5,
    method: str = "mass_mean", seed: int = 0,
    groups: ArrayF | None = None, purged: bool | None = None,
) -> dict:
    """Train a probe on SOURCE, transport to TARGET, report target AUROC.

    mode: 'procrustes_purged' | 'procrustes_raw' | 'ridge_purged' | 'acs_purged'
          | 'within_target' (upper bound, no transfer) | 'random' (lower-bound null)

    groups: split calibration/evaluation by these group ids instead of by row, so
        near-duplicate statements of one source item cannot straddle the boundary.
    purged: override the preprocessing, which is otherwise inferred from the mode
        string. Needed because 'within_target' and 'random' contain no 'purged'
        token and would silently be scored raw while the mode they are compared
        against is purged.
    """
    rng = np.random.default_rng(seed)
    n = len(y_src)
    if groups is None:
        idx = rng.permutation(n)
        ncal = int(cal_frac * n)
        cal, ev = idx[:ncal], idx[ncal:]
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        perm = rng.permutation(len(uniq))
        cal_groups = set(uniq[perm[: int(cal_frac * len(uniq))]].tolist())
        in_cal = np.array([x in cal_groups for x in g])
        cal, ev = np.where(in_cal)[0], np.where(~in_cal)[0]

    Xs, Xt = np.asarray(X_src, float), np.asarray(X_tgt, float)
    if purged is None:
        purged = "purged" in mode
    Xs_use = purge(Xs, conf_src) if purged else Xs

    if mode == "within_target":
        d = fit_direction((purge(Xt, conf_tgt) if purged else Xt)[cal], y_tgt[cal], method=method)
        Xt_eval = (purge(Xt, conf_tgt) if purged else Xt)[ev]
        return {"mode": mode, "auroc": _auroc(Xt_eval @ d, y_tgt[ev]), "n_eval": len(ev)}

    if mode == "random":
        d = _unit(rng.standard_normal(Xt.shape[1]))
        return {"mode": mode, "auroc": _auroc(Xt[ev] @ d, y_tgt[ev]), "n_eval": len(ev)}

    # train source probe on calibration rows
    d_src = fit_direction(Xs_use[cal], y_src[cal], method=method)

    if mode.startswith("procrustes") or mode.startswith("ridge"):
        Xt_use = purge(Xt, conf_tgt) if purged else Xt
        A = fit_map(Xt_use[cal], Xs_use[cal], orthogonal=mode.startswith("procrustes"))
        d_tgt = transport_direction(A, d_src)          # in target space
        return {"mode": mode, "auroc": _auroc(Xt_use[ev] @ d_tgt, y_tgt[ev]), "n_eval": len(ev)}

    if mode == "acs_purged" or mode == "acs":
        anchors_idx = cal[: min(256, len(cal))]
        Zs = acs_features(Xs_use, Xs_use[anchors_idx])
        Xt_use = purge(Xt, conf_tgt) if purged else Xt
        Zt = acs_features(Xt_use, Xt_use[anchors_idx])
        d = fit_direction(Zs[cal], y_src[cal], method=method)
        return {"mode": mode, "auroc": _auroc(Zt[ev] @ d, y_tgt[ev]), "n_eval": len(ev)}

    raise ValueError(f"unknown mode {mode!r}")


def transfer_matrix(features: dict, layer: int, pooling: str, confounds_of, *,
                    axis: str = "grounding", modes=("procrustes_purged", "procrustes_raw", "acs_purged"),
                    method="mass_mean", seed=0) -> dict:
    """Full source->target AUROC matrix for one axis.

    features: {model_short: feature_dict}; all share the SAME statement rows.
    confounds_of: callable(feature_dict, X)->confounds (e.g. analyze.build_confounds).
    """
    models = list(features.keys())
    out = {m: {} for m in models}
    for s in models:
        Xs = features[s][f"{pooling}_{layer}"].astype(np.float64)
        ys = features[s][axis].astype(int)
        cs = confounds_of(features[s], Xs)
        for t in models:
            Xt = features[t][f"{pooling}_{layer}"].astype(np.float64)
            yt = features[t][axis].astype(int)
            ct = confounds_of(features[t], Xt)
            row = {}
            for mode in modes:
                m = "within_target" if s == t else mode
                row[mode] = transfer_auroc(Xs, ys, cs, Xt, yt, ct, mode=m, method=method, seed=seed)["auroc"]
            out[s][t] = row
    return out
