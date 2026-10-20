"""Stage 0 — matched minimal-pair construction (CtrlPairs).

We turn NQ-Swap (Longpre et al., EMNLP 2021) into a fully-crossed 2x2 in which
the *grounding* bit and the *factuality* bit are flipped independently while the
surface form is held maximally fixed.

NQ-Swap gives, per question q: a gold/world-true answer ``a_true`` entailed by an
original context ``c_org``, and a substituted same-type entity ``a_swap`` entailed
by an entity-swapped context ``c_sub``. We cross the two contexts with the two
answers:

    context \\ answer |     a_true (F=1)      |     a_swap (F=0)
    ------------------+-----------------------+-----------------------
    c_org             | G=1 (org supports     | G=0 (org supports
                      |      true)            |      true, not swap)
    c_sub             | G=0 (sub supports     | G=1 (sub supports
                      |      swap, not true)  |      swap)

  * grounding   G = 1  iff the asserted answer is the one the GIVEN context entails
  * factuality  F = 1  iff the asserted answer equals the world-gold a_true

Each context appears with both answers and each answer with both contexts, so the
two labels are *orthogonal and balanced by construction* and answer-entity / length
/ register are matched across cells. The only surface difference between c_org and
c_sub is the swapped entity token. This is the control prior work lacked.

The module is dataset-loader-agnostic: pass a JSONL path or an HF dataset id whose
records expose the NQ-Swap fields (aliases handled below).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable

# field-name aliases seen across NQ-Swap mirrors
_ALIASES = {
    "question": ["question", "query", "q"],
    "a_true": ["org_answer", "gold_answer", "answer", "original_answer", "a_true"],
    "a_swap": ["sub_answer", "swapped_answer", "substituted_answer", "a_swap"],
    "c_org": ["org_context", "original_context", "gold_context", "context", "c_org"],
    "c_sub": ["sub_context", "swapped_context", "substituted_context", "c_sub"],
}

# (context_type, answer_type, grounding, factuality)
_CELLS = [
    ("org", "true", 1, 1),
    ("org", "swap", 0, 0),
    ("sub", "true", 0, 1),
    ("sub", "swap", 1, 0),
]


@dataclass
class Statement:
    item_id: int
    question: str
    context: str
    asserted_answer: str
    context_type: str        # org | sub
    answer_type: str         # true | swap
    grounding: int           # G: asserted answer entailed by GIVEN context
    factuality: int          # F: asserted answer == world-gold a_true
    a_true: str              # carried for the per-model parametric-belief check
    a_swap: str
    answer_char_len: int
    answer_tok_len: int       # whitespace tokens (proxy; true tok len set at extraction)
    lexical_overlap: float    # frac of answer tokens appearing in the context


def _first(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def normalize_record(raw: dict) -> dict | None:
    out = {}
    for canon, aliases in _ALIASES.items():
        v = _first(raw, aliases)
        if v is None:
            return None
        out[canon] = v.strip() if isinstance(v, str) else v
    # NQ-Swap answers are sometimes lists; take the first surface form
    for k in ("a_true", "a_swap"):
        if isinstance(out[k], (list, tuple)):
            out[k] = out[k][0]
    if out["a_true"] == out["a_swap"]:
        return None  # swap failed / degenerate; skip
    return out


def _overlap(answer: str, context: str) -> float:
    atoks = [t for t in answer.lower().split() if t]
    if not atoks:
        return 0.0
    cl = context.lower()
    return sum(1 for t in atoks if t in cl) / len(atoks)


def build_2x2(rec: dict, item_id: int) -> list[Statement]:
    ctx = {"org": rec["c_org"], "sub": rec["c_sub"]}
    ans = {"true": rec["a_true"], "swap": rec["a_swap"]}
    stmts = []
    for ctype, atype, g, f in _CELLS:
        context, answer = ctx[ctype], ans[atype]
        stmts.append(Statement(
            item_id=item_id, question=rec["question"], context=context,
            asserted_answer=answer, context_type=ctype, answer_type=atype,
            grounding=g, factuality=f, a_true=rec["a_true"], a_swap=rec["a_swap"],
            answer_char_len=len(answer), answer_tok_len=len(answer.split()),
            lexical_overlap=_overlap(answer, context),
        ))
    return stmts


def build_ctrlpairs(records: Iterable[dict], max_items: int | None = None) -> list[Statement]:
    out: list[Statement] = []
    kept = 0
    for raw in records:
        rec = normalize_record(raw)
        if rec is None:
            continue
        out.extend(build_2x2(rec, kept))
        kept += 1
        if max_items and kept >= max_items:
            break
    return out


# --------------------------- loaders ---------------------------
def _load_jsonl(path: str) -> list[dict]:
    recs = []
    with open(path) as fh:
        if path.endswith(".json"):
            data = json.load(fh)
            return data if isinstance(data, list) else data.get("data", [])
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


_HF_CANDIDATES = ["pminervini/NQ-Swap", "NQ-Swap", "kdpark/NQ-Swap"]


def load_source(source: str, split: str = "validation") -> list[dict]:
    """Load raw NQ-Swap records from a local file or an HF dataset id."""
    if os.path.exists(source):
        return _load_jsonl(source)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("install `datasets`, or pass a local NQ-Swap .jsonl path") from e
    candidates = [source] if source not in ("auto", "") else _HF_CANDIDATES
    last_err = None
    for cid in candidates:
        try:
            ds = load_dataset(cid, split=split)
            return [dict(r) for r in ds]
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        f"could not load NQ-Swap from {candidates}. Get it from "
        f"github.com/apple/ml-knowledge-conflicts and pass the .jsonl path. "
        f"Last error: {last_err}")


def save_statements(stmts: list[Statement], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for s in stmts:
            fh.write(json.dumps(asdict(s)) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build matched 2x2 CtrlPairs from NQ-Swap")
    ap.add_argument("--source", default="auto", help="local .jsonl path or HF id ('auto' tries known mirrors)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out", default="data/ctrlpairs.jsonl")
    args = ap.parse_args()

    raw = load_source(args.source, args.split)
    stmts = build_ctrlpairs(raw, max_items=args.max_items)
    save_statements(stmts, args.out)
    print(f"built {len(stmts)} statements from {len(stmts)//4} items -> {args.out}")

    from .surface_stats import audit
    print(audit(stmts))


if __name__ == "__main__":
    main()
