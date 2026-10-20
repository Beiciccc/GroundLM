"""Stage 0 v2 — SUPPORT x OVERLAP decoupled construction.

The v1 NQ-Swap 2x2 made grounding == lexical overlap (corr 0.99), so the
"faithfulness axis" was just a string-copy detector (pilot). v2
crosses *semantic support* with *lexical overlap* so the grounding probe cannot
be solved by overlap:

  cell A (S1,O1): (q,  c_org, a_true)              supported + present   (model-free)
  cell B (S1,O0): (q,  paraphrase(c_org, drop a_true), a_true)  supported + ABSENT  (needs paraphraser)
  cell C (S0,O1): (q', c_org, a_true)              UNsupported + present (wrong-question trap, model-free)
  cell D (S0,O0): (q,  c_sub, a_true)              unsupported + absent  (model-free)

corr(Support,Overlap): {A,C,D} -> 0.5 ; {A,B,C,D} -> 0.0.
``support`` is the faithfulness/grounding label; ``overlap_hi`` the lexical
covariate; ``factuality`` = asserted answer is the world-gold for the asked q.
Support labels for C (and B) should be NLI-verified in the full pipeline
(verify_support); the model-free cells assume the by-construction label.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

from .build_ctrlpairs import normalize_record, load_source  # reuse v1 loaders


@dataclass
class StatementV2:
    item_id: int
    cell: str                 # A | B | C | D
    question: str
    context: str
    asserted_answer: str
    support: int              # S: context semantically supports answer as the answer to question
    overlap_hi: int           # O: intended lexical presence of answer in context
    overlap_measured: int     # O actually measured (answer string in context)
    factuality: int           # F: asserted answer == world-gold for the asked question
    answer_tok_len: int
    lexical_overlap: float     # continuous: frac of answer tokens in context
    nli_entail_prob: float = -1.0  # open-NLI entailment of (context -> claim); the C4 NLI baseline


def _overlap_frac(answer: str, context: str) -> float:
    at = [t for t in answer.lower().split() if t]
    if not at:
        return 0.0
    cl = context.lower()
    return sum(1 for t in at if t in cl) / len(at)


def _present(answer: str, context: str) -> int:
    return int(answer.strip().lower() in context.lower())


def build_decoupled(records, *, paraphraser=None, max_items=None,
                    require_overlap_match: bool = True) -> list[StatementV2]:
    """paraphraser: optional callable(context, answer)->context' that keeps the
    answer entailed but removes the literal answer token (cell B). If None, B is
    skipped. ``require_overlap_match`` drops cells whose measured overlap differs
    from the intended overlap (keeps the factorial clean)."""
    recs = [r for r in (normalize_record(x) for x in records) if r is not None]
    if max_items:
        recs = recs[:max_items]
    out: list[StatementV2] = []
    n = len(recs)
    for i, rec in enumerate(recs):
        partner = recs[(i + 1) % n]            # wrong-question donor for cell C
        cells = [
            ("A", rec["question"], rec["c_org"], rec["a_true"], 1, 1, 1),
            ("C", partner["question"], rec["c_org"], rec["a_true"], 0, 1, 0),
            ("D", rec["question"], rec["c_sub"], rec["a_true"], 0, 0, 1),
        ]
        if paraphraser is not None:
            ctx_b = paraphraser(rec["c_org"], rec["a_true"])
            if ctx_b:
                cells.append(("B", rec["question"], ctx_b, rec["a_true"], 1, 0, 1))
        for cell, q, ctx, ans, S, O, F in cells:
            meas = _present(ans, ctx)
            if require_overlap_match and meas != O:
                continue                       # construction failed for this cell; drop
            out.append(StatementV2(
                item_id=i, cell=cell, question=q, context=ctx, asserted_answer=ans,
                support=S, overlap_hi=O, overlap_measured=meas, factuality=F,
                answer_tok_len=len(ans.split()), lexical_overlap=_overlap_frac(ans, ctx)))
    return out


def decoupling_audit(stmts) -> dict:
    import numpy as np
    rows = [s if isinstance(s, dict) else asdict(s) for s in rows_iter(stmts)]
    S = np.array([r["support"] for r in rows], float)
    O = np.array([r["overlap_measured"] for r in rows], float)
    ov = np.array([r["lexical_overlap"] for r in rows], float)
    cells = {}
    for r in rows:
        cells[r["cell"]] = cells.get(r["cell"], 0) + 1

    def corr(a, b):
        return 0.0 if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1])
    return {
        "n": len(rows), "cell_counts": cells,
        "corr_support_overlapbin": corr(S, O),       # target: ~0 (full) / ~0.5 (no paraphrase)
        "corr_support_overlapfrac": corr(S, ov),
        "support_rate": float(S.mean()), "overlap_rate": float(O.mean()),
    }


def rows_iter(stmts):
    return stmts


def save(stmts, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for s in stmts:
            fh.write(json.dumps(asdict(s)) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pminervini/NQ-Swap")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out", default="data/ctrlpairs_v2.jsonl")
    args = ap.parse_args()
    raw = load_source(args.source, args.split)
    stmts = build_decoupled(raw, max_items=args.max_items)  # model-free cells A,C,D
    save(stmts, args.out)
    print(f"built {len(stmts)} statements (cells A/C/D, no paraphrase) -> {args.out}")
    print(json.dumps(decoupling_audit(stmts), indent=2))


if __name__ == "__main__":
    main()
