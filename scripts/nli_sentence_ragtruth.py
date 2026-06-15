"""Sentence-level (SummaC-ZS style) NLI baseline on RAGTruth.

This is a *stronger* NLI baseline than the whole-response single-hypothesis NLI we
already report (`nli_entail_prob`, premise=full context, hypothesis=full response).
A reviewer would expect the standard decomposed entailment check, so we add it:

  1. split the response into sentences (claims) and the context into sentences;
  2. for each response sentence, test entailment against EVERY context sentence with
     the SAME open DeBERTa-MNLI model, and take the max (is this claim supported
     anywhere in the context?);
  3. aggregate over response sentences: mean = SummaC-ZS; min = worst-supported claim.

Higher score = more supported = faithful, so AUROC(score, faithful) is directly
comparable to the whole-response NLI and lexical-overlap baselines. Fully API-free;
runs on the local NLI model (no 8B LLM, no GPU box). Writes runs/nli_sentence_ragtruth.json.

Env: NLI_DEVICE (mps|cpu|cuda), NLI_BATCH, CTX_CAP, LIMIT (>0 = small subset probe).
"""
import os, sys, json, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import nltk
from sklearn.metrics import roc_auc_score
from groundlm.gate.baselines import nli_entailment_scores

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEVICE = os.environ.get("NLI_DEVICE", "mps")
BATCH = int(os.environ.get("NLI_BATCH", "64"))
MAXLEN = int(os.environ.get("MAXLEN", "256"))     # 98% of (ctx_sent, ans_sent) pairs are <=256 tok
CTX_CAP = int(os.environ.get("CTX_CAP", "60"))   # cap context sentences (>2% of pairs trimmed)
CHUNK = int(os.environ.get("CHUNK", "20000"))     # checkpoint granularity (resume if interrupted)
CACHE = os.environ.get("CACHE", "runs/_nli_sent_cache")
LIMIT = int(os.environ.get("LIMIT", "0"))         # >0 -> small subset for a speed/correctness probe


def ensure_punkt():
    for pkg in ["punkt_tab", "punkt"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}"); return
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def sents(t):
    t = (t or "").strip()
    if not t:
        return []
    try:
        ss = nltk.sent_tokenize(t)
    except LookupError:
        import re
        ss = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in ss if s.strip()]


def auc(s, yy):
    return float(roc_auc_score(yy, s)) if len(np.unique(yy)) == 2 else float("nan")


def main():
    ensure_punkt()
    rows = [json.loads(l) for l in open("data/ragtruth.jsonl")]
    if LIMIT:
        rows = rows[:LIMIT]

    # Flatten every (context_sentence, response_sentence) pair; remember the slice that
    # belongs to each response sentence so we can max/aggregate after one batched pass.
    premises, hyps, spans = [], [], []
    for r in rows:
        cs = sents(r["context"])[:CTX_CAP] or [(r["context"] or "").strip()[:2000] or "."]
        asx = sents(r["asserted_answer"]) or [(r["asserted_answer"] or "").strip() or "."]
        sp = []
        for h in asx:
            a = len(premises)
            premises.extend(cs)
            hyps.extend([h] * len(cs))
            sp.append((a, len(premises)))
        spans.append(sp)

    print(f"items={len(rows)} pairs={len(premises)} device={DEVICE} batch={BATCH} maxlen={MAXLEN} ctx_cap={CTX_CAP}", flush=True)
    # Sort by approx length so each batch pads to a similar (usually short) length: minimizes
    # padding compute and avoids MPS OOM spikes from a long pair landing in a big batch. Restore after.
    order = sorted(range(len(premises)), key=lambda k: len(premises[k]) + len(hyps[k]))
    inv = np.empty(len(order), dtype=np.int64)
    for rank, idx in enumerate(order):
        inv[idx] = rank
    prem_s = [premises[k] for k in order]
    hyp_s = [hyps[k] for k in order]
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    n = len(prem_s)
    nch = (n + CHUNK - 1) // CHUNK
    parts = []
    for ci in range(nch):
        a, b = ci * CHUNK, min((ci + 1) * CHUNK, n)
        cp = os.path.join(CACHE, f"chunk_{ci:04d}_n{n}.npy")   # n in name -> LIMIT/full don't collide
        if os.path.exists(cp):
            s = np.load(cp)
            if len(s) == b - a:
                parts.append(s)
                print(f"chunk {ci+1}/{nch} cached [{a}:{b}]", flush=True)
                continue
        s = nli_entailment_scores(prem_s[a:b], hyp_s[a:b], NLI, device=DEVICE, batch_size=BATCH, max_length=MAXLEN)
        np.save(cp, s)
        parts.append(s)
        print(f"chunk {ci+1}/{nch} [{a}:{b}] done, {time.time()-t0:.0f}s elapsed", flush=True)
    ent_s = np.concatenate(parts)
    ent = np.asarray(ent_s)[inv]   # undo the length sort
    print(f"NLI forward done in {time.time() - t0:.0f}s ({len(premises)/max(time.time()-t0,1):.0f} pairs/s)", flush=True)

    sm_mean, sm_min = [], []
    for sp in spans:
        per = [float(ent[a:b].max()) for (a, b) in sp]   # each claim's best support in context
        sm_mean.append(float(np.mean(per)))
        sm_min.append(float(np.min(per)))
    sm_mean, sm_min = np.array(sm_mean), np.array(sm_min)

    y = np.array([r["faithful"] for r in rows]).astype(int)
    tt = np.array([r["task_type"] for r in rows])
    nli_whole = np.array([r["nli_entail_prob"] for r in rows]).astype(float)
    ov = np.array([r["lexical_overlap"] for r in rows]).astype(float)

    methods = {"summac_mean": sm_mean, "summac_min": sm_min, "nli_whole": nli_whole, "overlap": ov}
    pooled = {k: auc(v, y) for k, v in methods.items()}
    tasks = sorted(set(tt.tolist()))
    per_task = {}
    for tk in tasks:
        m = tt == tk
        per_task[tk] = {"n": int(m.sum()), "faithful_rate": float(y[m].mean()),
                        **{k: auc(v[m], y[m]) for k, v in methods.items()}}
    macro = {k: float(np.mean([per_task[tk][k] for tk in tasks])) for k in methods}

    out = {"n": int(len(rows)), "pairs": int(len(premises)), "device": DEVICE, "ctx_cap": CTX_CAP,
           "nli_model": NLI, "aggregation": "max over context sentences; response = mean (SummaC-ZS) | min",
           "pooled": pooled, "macro": macro, "per_task": per_task}
    if not LIMIT:
        os.makedirs("runs", exist_ok=True)
        json.dump(out, open("runs/nli_sentence_ragtruth.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
