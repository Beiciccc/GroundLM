"""Stage 3/4 driver — turn cached features into the C1/C2 results per layer.

Faithfulness label = ``grounding`` (entailed by given context).
Factuality   label = ``factuality`` (asserted answer == world-gold a_true).

Confounds for the C2 purge: mean/first max-softmax, mean answer log-prob,
(whitespace) answer length, lexical overlap, and the per-row activation L2 norm
of the layer being analysed.
"""
from __future__ import annotations

import numpy as np

from .probe import separability_test, asymmetry_test


# IMPORTANT: keep CONFIDENCE confounds separate from SURFACE confounds.
# lexical_overlap correlates with the grounding label ~0.99 by construction, so
# folding it into the "confidence" purge would absorb the faithfulness axis for the
# wrong reason. The C2 confidence-asymmetry test uses CONFIDENCE only; overlap/length
# are a separate surface control.
CONFOUND_GROUPS = {
    # The three teacher-forced confidence scalars. Activation L2 norm was previously
    # included here, which disagreed with the manuscript and with the scripts that
    # produce Table 1 (they stack exactly these three); it is excluded so every
    # confidence purge in the paper uses one definition.
    "confidence": ["mean_maxsoftmax", "first_maxsoftmax", "mean_logprob"],
    "surface": ["lexical_overlap", "_len"],
    "all": ["mean_maxsoftmax", "first_maxsoftmax", "mean_logprob", "_norm",
            "lexical_overlap", "_len"],
}


def _len_key(data: dict) -> str:
    for k in ("answer_tok_len", "answer_tok_len_ws", "answer_char_len"):
        if k in data:
            return k
    raise KeyError("no answer-length field in features")


def faith_key(data: dict) -> str:
    """v2 uses 'support'; v1 uses 'grounding'."""
    return "support" if "support" in data else "grounding"


def build_confounds(data: dict, X: np.ndarray, kind: str = "confidence",
                    idx: np.ndarray | None = None) -> np.ndarray:
    cols = []
    for name in CONFOUND_GROUPS[kind]:
        if name == "_norm":
            cols.append(np.linalg.norm(X, axis=1))   # X is already masked to idx
            continue
        key = _len_key(data) if name == "_len" else name
        v = data[key].astype(np.float64)
        cols.append(v[idx] if idx is not None else v)
    return np.stack(cols, axis=1)


def run_layer(data: dict, layer: int, pooling: str = "last", *,
              method: str = "mass_mean", knows_gold_mask: np.ndarray | None = None,
              confound_kind: str = "confidence", seed: int = 0) -> dict:
    X = data[f"{pooling}_{layer}"].astype(np.float64)
    faith_y = data[faith_key(data)].astype(int)
    fact_y = data["factuality"].astype(int)
    idx = None
    if knows_gold_mask is not None:
        idx = np.asarray(knows_gold_mask, dtype=bool)
        X, faith_y, fact_y = X[idx], faith_y[idx], fact_y[idx]
    conf = build_confounds(data, X, kind=confound_kind, idx=idx)

    sep = separability_test(X, faith_y, fact_y, method=method, seed=seed)
    asym = asymmetry_test(X, faith_y, fact_y, conf, method=method, seed=seed)
    return {
        "layer": layer, "pooling": pooling, "n": int(len(faith_y)),
        "cross_angle": sep.cross_angle,
        "within_faith_null_mean": sep.within_faith_null_mean,
        "within_fact_null_mean": sep.within_fact_null_mean,
        "separable_p": sep.p_value, "separable": sep.reject_collinear(),
        "cross_angle_shared_removed": sep.cross_angle_shared_removed,
        "absorbed_faith": asym.faith["absorbed"],
        "absorbed_fact": asym.fact["absorbed"],
        "confidence_asymmetry": asym.asymmetry,
        "asymmetry_ci": asym.asymmetry_ci,
        "asymmetry_p": asym.p_value,
        "asym_supported": asym.supports_hypothesis(),
    }


def run_layer_sweep(data: dict, pooling: str = "last", **kw) -> list[dict]:
    layers = [int(L) for L in data["layers"]] if "layers" in data else \
        sorted({int(k.split("_")[1]) for k in data if k.startswith(f"{pooling}_")})
    return [run_layer(data, L, pooling=pooling, **kw) for L in layers]


def best_layer(results: list[dict]) -> dict:
    """Pick the layer with the largest separable, significant cross-angle."""
    sig = [r for r in results if r["separable"]] or results
    return max(sig, key=lambda r: r["cross_angle"])
