"""Stage 3 — null-calibrated separability test (contribution C1).

Question: do context-faithfulness and parametric-factuality occupy
geometrically *distinct* residual-stream directions?

We treat each notion as defining an *axis* (an undirected line through the
origin), so the relevant quantity is the acute angle between two axes,
``theta = arccos(|cos|) in [0, 90] deg``. Two facts make a raw cross-notion
angle uninterpretable on its own, which is exactly what prior work missed:

1. Estimated axes are noisy, so even two estimates of the *same* notion are
   not perfectly collinear. We therefore calibrate against a WITHIN-NOTION
   NULL: split one notion's data into disjoint halves and measure the
   half-vs-half axis angle. The cross-notion angle is "real" only if it
   significantly exceeds this null.
2. A shared "truth" component can inflate apparent distinctness or, per Bao et
   al. (2025), a single direction may span both. We therefore also report the
   cross-notion angle AFTER projecting out a shared axis.

This module is estimator-agnostic (mass-mean by default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .directions import fit_direction, _unit

ArrayF = np.ndarray


def axis_angle_deg(d1: ArrayF, d2: ArrayF) -> float:
    """Acute angle (degrees) between two axes (sign/polarity-invariant)."""
    d1 = _unit(np.asarray(d1, dtype=np.float64))
    d2 = _unit(np.asarray(d2, dtype=np.float64))
    cos = float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def project_out(X: ArrayF, d: ArrayF) -> ArrayF:
    """Remove the component of every row of X along unit axis d."""
    d = _unit(np.asarray(d, dtype=np.float64))
    X = np.asarray(X, dtype=np.float64)
    return X - np.outer(X @ d, d)


def _within_notion_null(X: ArrayF, y: ArrayF, method: str, n_iter: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Distribution of half-vs-half axis angles for a SINGLE notion.

    Captures the sampling variability of an estimated axis given this sample
    size and class balance. Cross-notion angle is compared against this.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n = len(y)
    angles = []
    for _ in range(n_iter):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        da = fit_direction(X[a], y[a], method=method)
        db = fit_direction(X[b], y[b], method=method)
        angles.append(axis_angle_deg(da, db))
    return np.asarray(angles)


@dataclass
class SeparabilityResult:
    cross_angle: float
    within_faith_null_mean: float
    within_fact_null_mean: float
    within_faith_null_q95: float
    within_fact_null_q95: float
    p_value: float            # P(within-null angle >= observed cross); conservative (max over notions)
    cross_angle_boot_ci: tuple
    cross_angle_shared_removed: float
    n: int
    method: str
    extras: dict = field(default_factory=dict)

    def reject_collinear(self, alpha: float = 0.05) -> bool:
        """True if the cross-notion angle significantly exceeds the within-null."""
        return self.p_value < alpha


def separability_test(
    X: ArrayF,
    faith_y: ArrayF,
    fact_y: ArrayF,
    *,
    method: str = "mass_mean",
    n_null: int = 300,
    n_boot: int = 300,
    shared_axis: ArrayF | None = None,
    seed: int = 0,
) -> SeparabilityResult:
    """Core C1 test.

    Parameters
    ----------
    X : (n, d) residual-stream features for a single (model, layer, pooling).
    faith_y, fact_y : (n,) binary labels for the two notions on the SAME stimuli.
    shared_axis : optional axis to project out before the "shared-removed"
        cross-angle (e.g. a jointly-fit truth direction or the factuality axis).
    """
    X = np.asarray(X, dtype=np.float64)
    faith_y = np.asarray(faith_y).astype(int)
    fact_y = np.asarray(fact_y).astype(int)
    rng = np.random.default_rng(seed)
    n = len(faith_y)

    d_faith = fit_direction(X, faith_y, method=method)
    d_fact = fit_direction(X, fact_y, method=method)
    cross = axis_angle_deg(d_faith, d_fact)

    null_faith = _within_notion_null(X, faith_y, method, n_null, rng)
    null_fact = _within_notion_null(X, fact_y, method, n_null, rng)

    # p = P(within-null >= cross). Conservative: take the larger p across notions.
    p_faith = float((null_faith >= cross).mean()) if null_faith.size else 1.0
    p_fact = float((null_fact >= cross).mean()) if null_fact.size else 1.0
    p_value = max(p_faith, p_fact)

    # CI of the cross-notion angle via SUBSAMPLING without replacement.
    # (With-replacement bootstrap duplicates rows, which both inflates the
    # angle-between-noisy-directions statistic and — when fed to cross-validated
    # probes downstream — leaks identical rows across train/test folds.)
    m = max(10, int(0.8 * n))
    boot = []
    for _ in range(n_boot):
        idx = rng.permutation(n)[:m]
        if len(np.unique(faith_y[idx])) < 2 or len(np.unique(fact_y[idx])) < 2:
            continue
        boot.append(axis_angle_deg(
            fit_direction(X[idx], faith_y[idx], method=method),
            fit_direction(X[idx], fact_y[idx], method=method),
        ))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (float("nan"),) * 2

    # shared-axis-removed cross angle (answers Bao et al.)
    if shared_axis is None:
        # default shared axis: the factuality (truth) direction itself
        shared_axis = d_fact
    Xs = project_out(X, shared_axis)
    cross_shared = axis_angle_deg(
        fit_direction(Xs, faith_y, method=method),
        fit_direction(Xs, fact_y, method=method),
    )

    return SeparabilityResult(
        cross_angle=cross,
        within_faith_null_mean=float(null_faith.mean()) if null_faith.size else float("nan"),
        within_fact_null_mean=float(null_fact.mean()) if null_fact.size else float("nan"),
        within_faith_null_q95=float(np.percentile(null_faith, 95)) if null_faith.size else float("nan"),
        within_fact_null_q95=float(np.percentile(null_fact, 95)) if null_fact.size else float("nan"),
        p_value=p_value,
        cross_angle_boot_ci=ci,
        cross_angle_shared_removed=cross_shared,
        n=n,
        method=method,
        extras={"p_faith": p_faith, "p_fact": p_fact},
    )
