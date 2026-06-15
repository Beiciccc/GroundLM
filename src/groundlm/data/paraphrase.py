"""Cell-B builder — "supported but NOT lexically copied" via an open model.

Uses an open instruct model (local; no API) to rewrite a supporting context so the
answer remains entailed but the literal answer string is removed. Each candidate is
verified with an open NLI model: keep it only if (a) the answer token is absent and
(b) the rewritten context still entails the (question, answer) claim. This yields
the S1/O0 cell that drives corr(support, overlap) to ~0.

Generation is the only GPU-heavy piece; everything else is model-free or NLI.
"""
from __future__ import annotations

from dataclasses import asdict

from .build_ctrlpairs import normalize_record
from .build_ctrlpairs_v2 import StatementV2, _present, _overlap_frac
from .verify_support import nli_entailment, DEFAULT_NLI

_PROMPT = (
    "Rewrite the passage so a reader can still infer that the answer to the question "
    "is \"{answer}\", but do NOT write \"{answer}\" anywhere — refer to it only "
    "indirectly (by description, role, or pronoun). Keep it concise and factual.\n\n"
    "Passage: {context}\nQuestion: {question}\nRewritten passage:"
)


class Paraphraser:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 max_new_tokens: int = 220):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == "cpu" and dtype in ("bfloat16", "float16"):
            dtype = "float32"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def rewrite(self, context: str, question: str, answer: str) -> str:
        import torch
        content = _PROMPT.format(answer=answer, context=context[:2000], question=question)
        if self.tok.chat_template:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True)
        else:
            text = content
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def build_cell_b(records, paraphraser: Paraphraser, *, nli_model: str = DEFAULT_NLI,
                 nli_device: str = "cpu", entail_thresh: float = 0.5,
                 overlap_max: float = 0.34, max_items: int | None = None) -> tuple[list, dict]:
    """Generate + verify cell-B (S1,O0) statements. Returns (statements, yield_stats).

    A candidate is kept only if the answer's TOKEN-level overlap with the rewrite is
    <= overlap_max (catches partial-name leaks like "Timothy B. Schmit"->"Timmy B.
    Schmit", which the exact-string test would miss) and NLI still entails it."""
    recs = [r for r in (normalize_record(x) for x in records) if r is not None]
    if max_items:
        recs = recs[:max_items]
    cands = []
    n_low_overlap = 0
    for i, rec in enumerate(recs):
        para = paraphraser.rewrite(rec["c_org"], rec["question"], rec["a_true"])
        if not para or _overlap_frac(rec["a_true"], para) > overlap_max:
            continue                                        # answer (or most of it) still present
        n_low_overlap += 1
        cands.append((i, rec, para))

    # NLI-verify the candidates still entail the (question, answer) claim
    stmts = []
    if cands:
        probe = [StatementV2(item_id=i, cell="B", question=r["question"], context=p,
                             asserted_answer=r["a_true"], support=1, overlap_hi=0,
                             overlap_measured=0, factuality=1,
                             answer_tok_len=len(r["a_true"].split()),
                             lexical_overlap=_overlap_frac(r["a_true"], p))
                 for (i, r, p) in cands]
        ent = nli_entailment(probe, nli_model, device=nli_device)["entail_prob"]
        for s, e in zip(probe, ent):
            if e >= entail_thresh:
                stmts.append(s)
    stats = {"n_items": len(recs), "n_low_overlap": n_low_overlap,
             "n_verified": len(stmts),
             "yield": len(stmts) / max(1, len(recs))}
    return stmts, stats
