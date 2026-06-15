"""Baselines for the Stage-6 gate comparison.

Implemented here (API-free, runnable): max-softmax confidence, and NLI/entailment
scoring with an open DeBERTa-MNLI model. AlignScore, ReDeEP, and SABER need their
authors' released code/checkpoints; we expose typed ADAPTERS that you wire to those
repos rather than reimplementing (and risking misrepresenting) them — a reviewer
expects the real baselines, not approximations.
"""
from __future__ import annotations

import numpy as np


def max_softmax_baseline(features: dict) -> np.ndarray:
    """Raw confidence baseline: mean answer-span max-softmax (higher = safer)."""
    return np.asarray(features["mean_maxsoftmax"], dtype=np.float64)


def nli_entailment_scores(contexts, answers, model_id: str,
                          device: str = "cuda", batch_size: int = 16,
                          max_length: int = 512) -> np.ndarray:
    """P(entailment) that the answer is supported by the context, via an MNLI model.

    Returns an array of entailment probabilities (higher = safer to answer). Uses
    the model's label map to locate the 'entailment' class robustly.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    ent_idx = next((i for i, l in id2label.items() if "entail" in l), 0)

    out = []
    for b, i in enumerate(range(0, len(contexts), batch_size)):
        prem = list(contexts[i:i + batch_size])
        hyp = list(answers[i:i + batch_size])
        enc = tok(prem, hyp, truncation=True, padding=True, max_length=max_length,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)[:, ent_idx]
        out.extend(probs.float().cpu().tolist())
        if device == "mps" and b % 16 == 15:   # release MPS fragmentation periodically
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    return np.asarray(out, dtype=np.float64)


# ---------------- adapters to external code (do NOT fake these) ----------------
def alignscore_scores(contexts, answers, ckpt_path: str):
    raise NotImplementedError(
        "Run AlignScore from github.com/yuh-zha/AlignScore with its released "
        "checkpoint; return its score array aligned to `answers`.")


def redeep_scores(features_or_model, **kw):
    raise NotImplementedError(
        "Compute the ReDeEP score from github.com/Jeryi-Sun/ReDeEP-public "
        "(Knowledge-FFN vs Copying-Head decoupling); return per-item scores.")


def saber_scores(model_id, statements, **kw):
    raise NotImplementedError(
        "Run SABER (arXiv:2605.18792) from the authors' release: multi-trace "
        "inference -> per-item reliability beliefs -> selective score. This is the "
        "primary gate baseline; benchmark it on the same risk-coverage frontier.")
