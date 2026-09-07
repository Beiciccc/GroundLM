"""Point-3 no-context ablation analysis (CPU, from cached features).

For each model, compare supp_O1 (support decodability within the answer-present O=1
stratum, cells A vs C, donor-collision-free) under three prompt regimes:
  normal           : "{context}\\n\\nQuestion: {q}\\nAnswer:"  (runs/{m}_v2)
  no_context       : "Question: {q}\\nAnswer:"                  (runs/{m}_v2_nc)
  shuffled_context : unrelated donor context + q                (runs/{m}_v2_shuf)

If supp_O1 stays high WITHOUT the context (no_context >= normal), the within-O=1 signal
is question-answer compatibility, NOT context-grounding. If it drops toward chance, the
signal genuinely uses the context. Shuffled is an intermediate control (context present
but irrelevant). Writes runs/nocontext_ablation.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys_path_marker = None  # noqa
import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cr_common import seeded_group_folds, source_groups   # noqa: E402
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project          # noqa: E402
from groundlm.probe.confidence import purge                           # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
CP = {"qwen25_7b": 14, "mistral7b_v03": 16, "llama31_8b": 16, "gemma2_9b": 21}  # a-priori depth-0.5


def load(d):
    return dict(np.load(f"runs/{d}/features.npz", allow_pickle=True))


def auroc(s, y):
    """Signed AUROC. Polarity is fixed by the training fold, never by the test fold:
    max(AUC, 1-AUC) floors a non-predictive direction above 0.5 and so inflates
    exactly the near-chance quantities this paper relies on."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def purge_fit(X_tr, C_tr):
    C = np.asarray(C_tr, float)
    if C.ndim == 1:
        C = C[:, None]
    mu, sd = C.mean(0), C.std(0) + 1e-8
    A = np.concatenate([np.ones((len(C), 1)), (C - mu) / sd], 1)
    beta, *_ = np.linalg.lstsq(A, np.asarray(X_tr, float), rcond=None)
    return beta, mu, sd


def purge_apply(X, C, fit):
    beta, mu, sd = fit
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    A = np.concatenate([np.ones((len(C), 1)), (C - mu) / sd], 1)
    return np.asarray(X, float) - A @ beta


def gcv_massmean(X, y, groups, folds=5, C=None, nseed=10):
    """Seeded source-grouped CV, mean over nseed splits; purge fitted inside each fold."""
    y = np.asarray(y).astype(int)
    vals = []
    for seed in range(nseed):
        oof = np.zeros(len(y))
        for tr, te in seeded_group_folds(groups, folds, seed):
            if C is None:
                Xtr, Xte = X[tr], X[te]
            else:
                f = purge_fit(X[tr], C[tr])
                Xtr, Xte = purge_apply(X[tr], C[tr], f), purge_apply(X[te], C[te], f)
            oof[te] = project(Xte, fit_direction(Xtr, y[tr]))
        vals.append(auroc(oof, y))
    return float(np.mean(vals))


# donor-collision map (same items across all prompt regimes)
rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
byitem = {}
for r in rows:
    byitem.setdefault(r["item_id"], {})[r["cell"]] = r
norm = lambda x: " ".join(x.lower().split())
collision = {i: (("A" in d and "C" in d) and norm(d["A"]["question"]) == norm(d["C"]["question"]))
             for i, d in byitem.items()}


def supp_o1(data, L):
    """Collision-free supp_O1 (raw and confidence-purged) at layer L."""
    O = data["overlap_measured"].astype(int); cell = data["cell"]; item = data["item_id"]
    src = source_groups(item)   # group by source QA, as everywhere else
    coll = np.array([bool(collision.get(int(item[k]), False)) for k in range(len(O))])
    m1 = (O == 1) & ~((cell == "C") & coll)
    X = data[f"last_{L}"].astype(np.float64)[m1]
    S = data["support"].astype(int)[m1]; g = src[m1]
    conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"], data["first_maxsoftmax"]], 1)[m1]
    raw = gcv_massmean(X, S, g)
    purged = gcv_massmean(X, S, g, C=conf)
    return raw, purged, int((cell[m1] == "A").sum()), int((cell[m1] == "C").sum())


out = {}
print("model        | normal raw/purged | no_context raw/purged | shuffled raw/purged")
for m in MODELS4:
    L = CP[m]
    res = {}
    for regime, d in [("normal", f"{m}_v2"), ("no_context", f"{m}_v2_nc"), ("shuffled", f"{m}_v2_shuf")]:
        if not os.path.exists(f"runs/{d}/features.npz"):
            res[regime] = None; continue
        raw, pur, nA, nC = supp_o1(load(d), L)
        res[regime] = {"raw": raw, "purged": pur, "n_A": nA, "n_C": nC}
    out[m] = {"layer": L, **res}
    g = lambda r, k: (res[r][k] if res.get(r) else float("nan"))
    print(f"{m:<12} | {g('normal','raw'):.3f}/{g('normal','purged'):.3f}       "
          f"| {g('no_context','raw'):.3f}/{g('no_context','purged'):.3f}         "
          f"| {g('shuffled','raw'):.3f}/{g('shuffled','purged'):.3f}")

json.dump(out, open("runs/nocontext_ablation.json", "w"), indent=2, default=float)
print("\nwrote runs/nocontext_ablation.json")
print("Interpretation: no_context >= normal -> signal is question-answer compatibility, "
      "NOT context-grounding. Big drop -> signal uses the context.")
