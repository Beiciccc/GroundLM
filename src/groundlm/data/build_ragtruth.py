"""RAGTruth real-hallucination eval set (for the C4 gate test).

We read each RAGTruth (context, response) with OUR model, pool the response-span
activations, and project onto the SUPPORT axis learned from CtrlPairs. The label is
faithful = the response has NO annotated hallucination span. This tests whether the
synthetic-CtrlPairs grounding axis detects REAL hallucinations — the proper C4
setting (CtrlPairs' balanced 33%-grounded mix is not a deployment distribution).

Source: `wandb/RAGTruth-processed` (splits train/test); fields query/context/output/
task_type/hallucination_labels.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


@dataclass
class RTStatement:
    item_id: int
    question: str            # the RAGTruth query/instruction
    context: str             # the source passage(s)
    asserted_answer: str     # the model-generated response (output)
    faithful: int            # 1 if no annotated hallucination span, else 0
    task_type: str           # QA | Summary | Data2txt
    answer_tok_len: int
    lexical_overlap: float
    nli_entail_prob: float = -1.0


def _overlap(answer: str, context: str) -> float:
    at = [t for t in answer.lower().split() if t]
    if not at:
        return 0.0
    cl = context.lower()
    return sum(1 for t in at if t in cl) / len(at)


def _short_task(t: str) -> str:
    t = (t or "").lower()
    if "summ" in t:
        return "Summary"
    if "data" in t or "2txt" in t or "yelp" in t:
        return "Data2txt"
    return "QA"


def build_ragtruth(source: str = "wandb/RAGTruth-processed", split: str = "test",
                   max_items: int | None = None, task_types=None,
                   min_resp_tokens: int = 5) -> list[RTStatement]:
    from datasets import load_dataset
    ds = load_dataset(source, split=split)
    alls: list[RTStatement] = []
    for r in ds:
        resp = (r.get("output") or "").strip()
        ctx = (r.get("context") or "").strip()
        if len(resp.split()) < min_resp_tokens or not ctx:
            continue
        tt = _short_task(r.get("task_type", ""))
        if task_types and tt not in task_types:
            continue
        # NOTE: hallucination_labels is a JSON *string* ("[]" is truthy!); the
        # reliable signal is the processed dict (faithful = all counts zero).
        proc = r.get("hallucination_labels_processed")
        if isinstance(proc, dict):
            faithful = int(sum(int(v) for v in proc.values()) == 0)
        else:
            try:
                faithful = int(len(json.loads(r.get("hallucination_labels") or "[]")) == 0)
            except Exception:
                faithful = 1
        alls.append(RTStatement(
            item_id=0, question=(r.get("query") or "").strip(), context=ctx,
            asserted_answer=resp, faithful=faithful, task_type=tt,
            answer_tok_len=len(resp.split()), lexical_overlap=_overlap(resp, ctx)))
    # the split is grouped by task; take a STRIDED subset so the sample spans
    # tasks and labels rather than the first-N (which is one task, one label).
    if max_items and len(alls) > max_items:
        step = len(alls) / max_items
        alls = [alls[int(i * step)] for i in range(max_items)]
    for i, s in enumerate(alls):
        s.item_id = i
    return alls


def save(stmts, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for s in stmts:
            fh.write(json.dumps(asdict(s)) + "\n")


def summary(stmts) -> dict:
    rows = [s if isinstance(s, dict) else asdict(s) for s in stmts]
    n = len(rows)
    faith = sum(r["faithful"] for r in rows)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_type"], [0, 0])
        by_task[r["task_type"]][0] += 1
        by_task[r["task_type"]][1] += r["faithful"]
    return {"n": n, "faithful_rate": faith / max(1, n),
            "by_task": {k: {"n": v[0], "faithful_rate": v[1] / max(1, v[0])} for k, v in by_task.items()}}
