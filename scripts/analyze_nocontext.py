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
from sklearn.model_selection import GroupKFold
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


def gcv_massmean(X, y, groups, folds=5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        d = fit_direction(X[tr], y[tr]); oof[te] = project(X[te], d)
    return auroc(oof, y)


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
    coll = np.array([bool(collision.get(int(item[k]), False)) for k in range(len(O))])
    m1 = (O == 1) & ~((cell == "C") & coll)
    X = data[f"last_{L}"].astype(np.float64)[m1]
    S = data["support"].astype(int)[m1]; g = item[m1]
    conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"], data["first_maxsoftmax"]], 1)[m1]
    raw = gcv_massmean(X, S, g)
    purged = gcv_massmean(purge(X, conf), S, g)
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
