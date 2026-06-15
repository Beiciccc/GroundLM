"""Local validation of Stage-0 CtrlPairs construction (no model needed).

Fabricates NQ-Swap-shaped records (same-type entity swaps) and checks:
  * each item yields a full, correctly-labelled 2x2,
  * labels G and F are orthogonal and balanced,
  * the matched guarantees hold: corr(length, grounding)~0 and
    corr(overlap, factuality)~0,
  * the expected confounds appear (overlap tracks grounding; length tracks
    factuality) so we know what to purge.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.data import build_ctrlpairs, audit, format_audit  # noqa: E402

CITIES_TRUE = ["Paris", "Rome", "Berlin", "Madrid", "Vienna", "Lisbon", "Prague", "Oslo"]
CITIES_SWAP = ["Toronto", "Nairobi", "Jakarta", "Bogota", "Helsinki", "Bangkok", "Cairo", "Lima"]


def fake_records(n: int = 60):
    recs = []
    for i in range(n):
        a_true = CITIES_TRUE[i % len(CITIES_TRUE)]
        a_swap = CITIES_SWAP[i % len(CITIES_SWAP)]
        q = f"What is the capital referenced in document {i}?"
        c_org = f"Report {i}: the administrative seat discussed here is {a_true}, founded long ago."
        c_sub = f"Report {i}: the administrative seat discussed here is {a_swap}, founded long ago."
        recs.append({"question": q, "org_answer": a_true, "sub_answer": a_swap,
                     "org_context": c_org, "sub_context": c_sub})
    return recs


def main():
    stmts = build_ctrlpairs(fake_records(60))
    assert len(stmts) == 60 * 4, len(stmts)

    # check the four cells for item 0
    cells = {(s.context_type, s.answer_type): s for s in stmts if s.item_id == 0}
    assert cells[("org", "true")].grounding == 1 and cells[("org", "true")].factuality == 1
    assert cells[("sub", "swap")].grounding == 1 and cells[("sub", "swap")].factuality == 0
    assert cells[("sub", "true")].grounding == 0 and cells[("sub", "true")].factuality == 1
    assert cells[("org", "swap")].grounding == 0 and cells[("org", "swap")].factuality == 0
    # grounded answers are the ones present in their given context
    assert cells[("org", "true")].lexical_overlap > 0
    assert cells[("sub", "true")].lexical_overlap == 0  # a_true absent from sub context

    rep = audit(stmts)
    print(format_audit(rep))
    assert rep["cell_counts_balanced"], "cells not balanced"
    assert abs(rep["label_corr_G_F"]) < 1e-6, "labels not orthogonal"
    g = rep["matched_guarantees"]
    assert abs(g["corr_length_grounding"]) < 0.10, g
    assert abs(g["corr_overlap_factuality"]) < 0.10, g
    c = rep["expected_confounds"]
    assert abs(c["corr_overlap_grounding"]) > 0.3, "overlap should track grounding"

    print("\nALL CTRLPAIRS CHECKS PASS")


if __name__ == "__main__":
    main()
