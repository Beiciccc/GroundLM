"""Stage 1 — residual-stream + confidence extraction (optimized).

Optimizations over the pilot version (which idled the GPU):
  * load the model ONCE and reuse it for extraction AND parametric belief
    (pilot loaded two copies -> ~30 GB);
  * vectorized batch pooling: one GPU->CPU transfer per (layer, batch) instead of
    one per (example, layer) -> keeps the GPU busy;
  * BATCHED parametric-belief scoring (pilot used batch size 1).

Plain prompt: "{context}\n\nQuestion: {question}\nAnswer: {asserted_answer}".
LEFT padding puts the answer at the final positions, so last-token / answer-span
indexing is correct under batching. Metadata fields are auto-collected, so the
same extractor handles the v1 and v2 statement schemas.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

PROMPT_TMPL = "{context}\n\nQuestion: {question}\nAnswer:"
_TEXT_FIELDS = {"question", "context", "asserted_answer", "a_true", "a_swap"}


def load_model(model_id: str, dtype: str = "bfloat16", device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if device == "cpu" and dtype in ("bfloat16", "float16"):
        dtype = "float32"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), output_hidden_states=True)
    model.to(device)
    model.eval()
    return model, tok


def _resolve_layers(spec, n_layers: int) -> list[int]:
    if spec in (None, "all"):
        return list(range(n_layers + 1))
    if spec == "mid":
        lo, hi = int(0.3 * n_layers), int(0.8 * n_layers)
        return list(range(lo, hi + 1))
    return [int(x) for x in str(spec).split(",") if x != ""]


def _rows(statements):
    return [s if isinstance(s, dict) else asdict(s) for s in statements]


def _encode_prompt_answer(tok, prompt: str, answer: str, max_len: int):
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    a_ids = tok(" " + answer.strip(), add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
    ids = (p_ids + a_ids)[-max_len:]
    alen = min(len(a_ids), len(ids) - 1)
    return ids, max(1, alen)


def _left_pad(batch_ids, pad_id):
    import torch
    T = max(len(x) for x in batch_ids)
    input_ids = torch.full((len(batch_ids), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch_ids), T), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        input_ids[i, T - len(ids):] = torch.tensor(ids)
        attn[i, T - len(ids):] = 1
    return input_ids, attn, T


def extract(model_id: str, statements, *, layers="mid", pooling=("last", "mean"),
            batch_size: int = 16, dtype: str = "bfloat16", device: str = "cuda",
            max_len: int = 1024, model=None, tok=None, prompt_mode: str = "normal") -> dict:
    # prompt_mode: "normal" (context+question+answer) | "no_context" (question+answer only,
    # ablation for Point-3 question-answer-compatibility) | "shuffled_context"
    # (question+answer with an UNRELATED donor context, ablation that context is actually used).
    import torch
    import torch.nn.functional as F

    if model is None:
        model, tok = load_model(model_id, dtype=dtype, device=device)
    device = str(next(model.parameters()).device)   # always match the model's device
    rows = _rows(statements)

    with torch.no_grad():
        n_layers = len(model(**tok("hi", return_tensors="pt").to(device)).hidden_states) - 1
    keep = _resolve_layers(layers, n_layers)

    feats = {f"{pool}_{L}": [] for pool in pooling for L in keep}
    conf = {"mean_logprob": [], "mean_maxsoftmax": [], "first_maxsoftmax": []}
    meta_keys = [k for k, v in rows[0].items()
                 if k not in _TEXT_FIELDS and (isinstance(v, (int, float, bool))
                 or (isinstance(v, str) and len(v) <= 16))]   # keep 'cell' etc. for subsetting
    meta = {k: [] for k in meta_keys}

    donor_ctx = None
    if prompt_mode == "shuffled_context":                 # unrelated donor context per row (seeded)
        perm = np.random.default_rng(12345).permutation(len(rows))
        perm = [(int(p) + 1) % len(rows) if int(p) == i else int(p) for i, p in enumerate(perm)]
        donor_ctx = [rows[p]["context"] for p in perm]

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        enc, alens = [], []
        for j, r in enumerate(batch):
            if prompt_mode == "no_context":
                prompt = f"Question: {r['question']}\nAnswer:"
            elif prompt_mode == "shuffled_context":
                prompt = PROMPT_TMPL.format(context=donor_ctx[start + j], question=r["question"])
            else:
                prompt = PROMPT_TMPL.format(context=r["context"], question=r["question"])
            ids, alen = _encode_prompt_answer(tok, prompt, r["asserted_answer"], max_len)
            enc.append(ids); alens.append(alen)
        input_ids, attn, T = _left_pad(enc, tok.pad_token_id)
        input_ids, attn = input_ids.to(device), attn.to(device)
        La = torch.tensor(alens, device=device)
        ans_mask = (torch.arange(T, device=device)[None, :] >= (T - La)[:, None]).float()  # (B,T)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
        hs, logits = out.hidden_states, out.logits

        for L in keep:
            h = hs[L]                                    # (B,T,H)
            if "last" in pooling:
                feats[f"last_{L}"].append(h[:, -1, :].float().cpu().numpy())
            if "mean" in pooling:
                summed = (h * ans_mask[:, :, None]).sum(1)
                mean = summed / ans_mask.sum(1, keepdim=True).clamp(min=1)
                feats[f"mean_{L}"].append(mean.float().cpu().numpy())

        for i in range(len(batch)):                      # confidence: small per-example slices
            a0 = T - alens[i]
            pred = logits[i, a0 - 1:T - 1, :].float()     # (La,V)
            tgt = input_ids[i, a0:T]
            lp = F.log_softmax(pred, dim=-1)
            conf["mean_logprob"].append(float(lp.gather(1, tgt[:, None]).mean()))
            sm = F.softmax(pred, dim=-1).max(dim=-1).values
            conf["mean_maxsoftmax"].append(float(sm.mean()))
            conf["first_maxsoftmax"].append(float(sm[0]))
        for r in batch:
            for k in meta_keys:
                meta[k].append(r[k])

    result = {"model_id": model_id, "layers": keep, "pooling": list(pooling), "n_layers": n_layers,
              "prompt_mode": prompt_mode}
    for k, v in feats.items():
        result[k] = np.concatenate(v).astype(np.float16)
    for k, v in conf.items():
        result[k] = np.asarray(v, dtype=np.float32)
    for k, v in meta.items():
        result[k] = np.asarray(v)
    return result


def _score_answers(model, tok, pairs, device, batch_size, max_len):
    """Mean answer-token log-prob for each (prompt, answer) pair, batched."""
    import torch
    import torch.nn.functional as F
    out = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        enc, alens = [], []
        for prompt, ans in chunk:
            ids, alen = _encode_prompt_answer(tok, prompt, ans, max_len)
            enc.append(ids); alens.append(alen)
        input_ids, attn, T = _left_pad(enc, tok.pad_token_id)
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        for i in range(len(chunk)):
            a0 = T - alens[i]
            pred = logits[i, a0 - 1:T - 1, :].float()
            tgt = input_ids[i, a0:T]
            out.append(float(F.log_softmax(pred, -1).gather(1, tgt[:, None]).mean()))
    return out


def score_parametric_belief(model_id: str, statements, *, dtype="bfloat16", device="cuda",
                            batch_size: int = 32, max_len: int = 512,
                            model=None, tok=None) -> dict:
    """Closed-book: per item, is P(a_true|q) > P(a_swap|q)? Batched over all items."""
    if model is None:
        model, tok = load_model(model_id, dtype=dtype, device=device)
    device = str(next(model.parameters()).device)   # always match the model's device
    rows = _rows(statements)
    items = {}
    for r in rows:
        if "a_true" in r and "a_swap" in r:
            items.setdefault(r["item_id"], r)
    items = list(items.values())

    pairs, idx = [], []
    for it in items:
        q = it["question"] if "question" in it else it.get("q", "")
        prompt = f"Question: {q}\nAnswer:"
        idx.append(len(pairs))
        pairs.append((prompt, it["a_true"])); pairs.append((prompt, it["a_swap"]))
    scores = _score_answers(model, tok, pairs, device, batch_size, max_len)

    out = {"item_id": [], "knows_gold": [], "margin": []}
    for it, j in zip(items, idx):
        lt, ls = scores[j], scores[j + 1]
        out["item_id"].append(it["item_id"])
        out["knows_gold"].append(bool(lt > ls))
        out["margin"].append(lt - ls)
    return {k: np.asarray(v) for k, v in out.items()}


def save_features(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)
    meta = {k: result[k] for k in ("model_id", "layers", "pooling", "n_layers")}
    with open(path + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def load_features(path: str) -> dict:
    if not path.endswith(".npz"):
        path += ".npz"
    data = dict(np.load(path, allow_pickle=True))
    metap = path + ".meta.json"
    if os.path.exists(metap):
        data.update(json.load(open(metap)))
    return data
