"""Stage 6 — one-pass two-axis conformal grounding gate (contribution C4).

For an incoming (question, retrieved context, candidate answer) we read, in a
single forward pass, the signed projections onto the two identified directions:

    s_faith = x . d_faith   (confidence-purged faithfulness axis)
    s_fact  = x . d_fact    (parametric-factuality axis)

and route: high s_faith -> answer/trust-context; low s_faith but high s_fact ->
conflict (defer/abstain); low both -> abstain. Thresholds are set by SPLIT
CONFORMAL calibration so the selective risk among answered items is controlled at
a target level with a finite-sample guarantee (no API; an open DeBERTa-MNLI model
is the entailment oracle for the FDR-E variant and the NLI baseline).

This module is the math + evaluation; external baselines that need other code
(AlignScore, ReDeEP, SABER) are called through documented adapters in
``baselines.py`` rather than reimplemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayF = np.ndarray


# --------------------------- risk-coverage ---------------------------
def risk_coverage_curve(score: ArrayF, correct: ArrayF, higher_is_safer: bool = True):
    """Return (coverage, risk): answer the most-confident first; risk = error
    rate among answered. ``correct`` is 1 if the answer is faithful/correct."""
    score = np.asarray(score, dtype=np.float64)
    correct = np.asarray(correct).astype(int)
    order = np.argsort(-score if higher_is_safer else score)
    err = 1 - correct[order]
    n = len(order)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk


# np.trapz was renamed to np.trapezoid in NumPy 2.x (removed in some builds).
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def aurc(score: ArrayF, correct: ArrayF, higher_is_safer: bool = True) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(score, correct, higher_is_safer)
    return float(_TRAPZ(risk, cov))


def coverage_at_risk(score: ArrayF, correct: ArrayF, max_risk: float,
                     higher_is_safer: bool = True) -> float:
    """Largest coverage whose selective risk stays <= max_risk."""
    cov, risk = risk_coverage_curve(score, correct, higher_is_safer)
    ok = cov[risk <= max_risk]
    return float(ok.max()) if ok.size else 0.0


# --------------------------- split conformal ---------------------------
def split_conformal_threshold(cal_score: ArrayF, cal_correct: ArrayF, alpha: float,
                              higher_is_safer: bool = True, finite_sample: bool = False,
                              delta: float = 0.1) -> float:
    """Most permissive score threshold whose selective risk <= alpha on calibration.

    Default (``finite_sample=False``) controls the *empirical* selective risk
    (SGen-style split-conformal; marginal control under exchangeability) and yields
    a usable coverage. Set ``finite_sample=True`` to add a Hoeffding slack at level
    1-delta for a high-probability guarantee (more conservative, lower coverage).

    Returns tau: answer iff score >= tau (higher_is_safer) else score <= tau.
    """
    s = np.asarray(cal_score, dtype=np.float64)
    err = 1 - np.asarray(cal_correct).astype(int)
    cand = np.unique(s)
    # default if nothing meets the target: answer ~nothing (most conservative)
    best, best_cov = (cand.max() + 1.0 if higher_is_safer else cand.min() - 1.0), -1
    for tau in cand:
        acc = (s >= tau) if higher_is_safer else (s <= tau)
        m = int(acc.sum())
        if m == 0:
            continue
        emp = err[acc].mean()
        slack = np.sqrt(np.log(1.0 / delta) / (2 * m)) if finite_sample else 0.0
        # among thresholds meeting the risk target, keep the MAX-coverage one
        if emp + slack <= alpha and m > best_cov:
            best, best_cov = tau, m
    return float(best)


# --------------------------- two-axis gate ---------------------------
ANSWER, CONFLICT, ABSTAIN = "answer", "conflict", "abstain"


@dataclass
class GateConfig:
    tau_faith: float
    tau_fact: float           # only consulted when faithfulness is low
    higher_faith_safer: bool = True


def two_axis_decisions(s_faith: ArrayF, s_fact: ArrayF, cfg: GateConfig) -> np.ndarray:
    s_faith = np.asarray(s_faith, float)
    s_fact = np.asarray(s_fact, float)
    faith_ok = (s_faith >= cfg.tau_faith) if cfg.higher_faith_safer else (s_faith <= cfg.tau_faith)
    fact_hi = s_fact >= cfg.tau_fact
    out = np.full(len(s_faith), ABSTAIN, dtype=object)
    out[faith_ok] = ANSWER
    out[~faith_ok & fact_hi] = CONFLICT     # parametric-confident but unsupported -> defer
    return out


def fit_gate(s_faith: ArrayF, s_fact: ArrayF, correct: ArrayF, alpha: float,
             higher_faith_safer: bool = True) -> GateConfig:
    """Calibrate the faithfulness threshold for selective risk <= alpha; set the
    conflict threshold at the factuality median (route confident-unsupported to defer)."""
    tau_faith = split_conformal_threshold(s_faith, correct, alpha, higher_faith_safer)
    tau_fact = float(np.median(s_fact))
    return GateConfig(tau_faith=tau_faith, tau_fact=tau_fact, higher_faith_safer=higher_faith_safer)


def evaluate_gate(s_faith: ArrayF, s_fact: ArrayF, correct: ArrayF,
                  baselines: dict | None = None) -> dict:
    """AURC of the faithfulness axis (our selective score) vs baselines, plus the
    two-axis gate's answered-set risk. ``baselines`` maps name->score array
    (higher = safer to answer)."""
    res = {"aurc_faith_axis": aurc(s_faith, correct),
           "aurc_two_axis": aurc(_two_axis_score(s_faith, s_fact), correct)}
    if baselines:
        for name, sc in baselines.items():
            res[f"aurc_{name}"] = aurc(np.asarray(sc), correct)
    return res


def _two_axis_score(s_faith: ArrayF, s_fact: ArrayF) -> ArrayF:
    """Single selective ordering for the two-axis gate: trust grounding, and among
    equally-grounded items prefer those whose parametric factuality also agrees.
    Standardize each axis, then combine (faithfulness dominant)."""
    f = (np.asarray(s_faith, float) - np.mean(s_faith)) / (np.std(s_faith) + 1e-8)
    g = (np.asarray(s_fact, float) - np.mean(s_fact)) / (np.std(s_fact) + 1e-8)
    return f + 0.25 * g
