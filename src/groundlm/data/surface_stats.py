"""Stage 0 audit — verify the matched-pair surface guarantees actually hold.

The 2x2 gives two clean guarantees and two *expected* (controlled) confounds:

* Flipping GROUNDING (same item, same answer, org<->sub context) keeps the
  asserted answer identical, so answer length is matched across grounding.
  GUARANTEE: corr(answer_length, grounding) ~ 0.
* Flipping FACTUALITY (same item, same context, true<->swap answer) keeps the
  context identical, so lexical structure is matched across factuality.
  GUARANTEE: corr(lexical_overlap, factuality) ~ 0.
* EXPECTED confounds (NOT failures; included in the C2 confound set and purged):
  length tracks factuality (a_true vs a_swap are different entities), and
  lexical_overlap tracks grounding (a grounded answer appears in its context).

PASS requires balanced cell counts, orthogonal labels, and both guarantees.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

_CELLS = [(1, 1), (0, 0), (0, 1), (1, 0)]


def _as_dicts(stmts: Sequence) -> list[dict]:
    return [s if isinstance(s, dict) else asdict(s) for s in stmts]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def audit(stmts: Sequence, guarantee_tol: float = 0.10) -> dict:
    rows = _as_dicts(stmts)
    g = np.array([r["grounding"] for r in rows], dtype=float)
    f = np.array([r["factuality"] for r in rows], dtype=float)
    length = np.array([r["answer_tok_len"] for r in rows], dtype=float)
    overlap = np.array([r["lexical_overlap"] for r in rows], dtype=float)

    counts = {f"G{gg}F{ff}": int(((g == gg) & (f == ff)).sum()) for gg, ff in _CELLS}
    balanced = len(set(counts.values())) == 1

    corr_GF = _corr(g, f)
    guarantees = {
        "corr_length_grounding": _corr(length, g),   # ~0 required
        "corr_overlap_factuality": _corr(overlap, f),  # ~0 required
    }
    expected_confounds = {
        "corr_length_factuality": _corr(length, f),   # nonzero expected (entity differs)
        "corr_overlap_grounding": _corr(overlap, g),   # nonzero expected (answer in context)
    }
    guarantees_ok = all(abs(v) < guarantee_tol for v in guarantees.values())

    return {
        "n_statements": len(rows),
        "n_items": len(rows) // 4,
        "cell_counts": counts,
        "cell_counts_balanced": balanced,
        "label_corr_G_F": corr_GF,
        "matched_guarantees": guarantees,
        "expected_confounds": expected_confounds,
        "PASS": bool(balanced and abs(corr_GF) < 1e-6 and guarantees_ok),
    }


def format_audit(rep: dict) -> str:
    g = rep["matched_guarantees"]; c = rep["expected_confounds"]
    return "\n".join([
        f"CtrlPairs audit: {rep['n_statements']} statements / {rep['n_items']} items",
        f"  cell counts: {rep['cell_counts']}  balanced={rep['cell_counts_balanced']}",
        f"  label corr(G,F): {rep['label_corr_G_F']:+.4f} (target ~0)",
        "  matched guarantees (target ~0):",
        f"    corr(length, grounding)   = {g['corr_length_grounding']:+.4f}",
        f"    corr(overlap, factuality) = {g['corr_overlap_factuality']:+.4f}",
        "  expected/controlled confounds (purged in C2):",
        f"    corr(length, factuality)  = {c['corr_length_factuality']:+.4f}",
        f"    corr(overlap, grounding)  = {c['corr_overlap_grounding']:+.4f}",
        f"  PASS={rep['PASS']}",
    ])
