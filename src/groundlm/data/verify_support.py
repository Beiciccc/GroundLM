"""Open-model NLI verification of the SUPPORT (grounding) label — no API, no humans.

For each statement we ask an MNLI model whether the context entails the claim
"the answer to <question> is <answer>". This (a) validates the by-construction
support labels (esp. that the cell-C wrong-question trap is genuinely unsupported
even though the answer string is present), and (b) gives a model-checked support
label decoupled from lexical overlap for the C1 analysis.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

DEFAULT_NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def _hypothesis(question: str, answer: str) -> str:
    q = question.strip().rstrip("?")
    return f'The answer to the question "{q}?" is {answer}.'


def nli_entailment(statements, model_id: str = DEFAULT_NLI, *, device: str = "cpu",
                   batch_size: int = 16, max_length: int = 512) -> dict:
    """Return per-statement entailment probability and 3-way argmax label.
    premise = context, hypothesis = declarativized (question, answer)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = [s if isinstance(s, dict) else asdict(s) for s in statements]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    ent_idx = next((i for i, l in id2label.items() if "entail" in l), 0)

    ent_prob, pred_lbl = [], []
    for i in range(0, len(rows), batch_size):
        b = rows[i:i + batch_size]
        prem = [r["context"] for r in b]
        hyp = [_hypothesis(r["question"], r["asserted_answer"]) for r in b]
        enc = tok(prem, hyp, truncation=True, padding=True, max_length=max_length,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)
        ent_prob.extend(probs[:, ent_idx].float().cpu().tolist())
        pred_lbl.extend([id2label[int(j)] for j in probs.argmax(-1).cpu().tolist()])
    return {"entail_prob": np.asarray(ent_prob), "pred_label": np.asarray(pred_lbl, dtype=object)}


def verify_and_audit(statements, model_id: str = DEFAULT_NLI, *, device: str = "cpu",
                     entail_thresh: float = 0.5) -> dict:
    rows = [s if isinstance(s, dict) else asdict(s) for s in statements]
    res = nli_entailment(rows, model_id, device=device)
    ep = res["entail_prob"]
    nli_support = (ep >= entail_thresh).astype(int)
    by_constr = np.array([r["support"] for r in rows])
    overlap = np.array([r["overlap_measured"] for r in rows], float)

    # agreement between NLI support and by-construction support, per cell
    cells = sorted({r["cell"] for r in rows})
    per_cell = {}
    for c in cells:
        m = np.array([r["cell"] == c for r in rows])
        per_cell[c] = {
            "n": int(m.sum()),
            "byconstr_support_rate": float(by_constr[m].mean()),
            "nli_support_rate": float(nli_support[m].mean()),
            "mean_entail_prob": float(ep[m].mean()),
        }
    agree = float((nli_support == by_constr).mean())

    def corr(a, b):
        return 0.0 if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1])
    return {
        "n": len(rows), "nli_model": model_id,
        "agreement_nli_vs_byconstruction": agree,
        "per_cell": per_cell,
        "corr_nli_support_overlap": corr(nli_support.astype(float), overlap),
        "corr_byconstr_support_overlap": corr(by_constr.astype(float), overlap),
        "entail_prob": ep, "nli_support": nli_support,
    }
