"""Sensitivity checks on the duplicate-leakage finding.

(1) LOGISTIC probe (high capacity, the estimator that CAN memorise a duplicate row)
    on the same Table-1 subset, under the three grouping schemes. If duplicate leakage
    is harmless even here, the near-null effect on Table 1 is not a mass-mean artifact.
(2) The Table-1 `cross` column (train A vs D, evaluate A vs C). As shipped it fits the
    direction on ALL A/D rows and evaluates on A/C rows -- so every A row is in both
    train and eval, a train-on-test overlap independent of duplicates. Recompute it
    with the A rows held out by group.
"""
from __future__ import annotations
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from _cr_common import load, apriori_layer, auroc, seeded_group_folds, collision_map, stat
from groundlm.probe.directions import fit_direction, project

MODELS = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
NSEED = 10
rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
norm = lambda x: " ".join(str(x).lower().split())
byitem = {}
for r in rows:
    byitem.setdefault(r["item_id"], {})[r["cell"]] = r
qca_of = {i: (norm(d["A"]["question"]), norm(d["A"]["context"]), norm(d["A"]["asserted_answer"]))
          for i, d in byitem.items()}
q_of = {i: norm(d["A"]["question"]) for i, d in byitem.items()}
qid = {k: n for n, k in enumerate(sorted(set(qca_of.values())))}
qqid = {k: n for n, k in enumerate(sorted(set(q_of.values())))}

coll = collision_map()
cell = np.array([r["cell"] for r in rows]); item = np.array([r["item_id"] for r in rows])
O = np.array([r["overlap_measured"] for r in rows])
S_all = np.array([r["support"] for r in rows]).astype(int)
keepC = ~((cell == "C") & np.array([bool(coll.get(int(i), False)) for i in item]))
m1 = (O == 1) & keepC
mAD = np.isin(cell, ["A", "D"]); mAC = np.isin(cell, ["A", "C"]) & keepC

G = {"item_id": lambda idx: item[idx],
     "source_qca": lambda idx: np.array([qid[qca_of[int(i)]] for i in item[idx]]),
     "source_question": lambda idx: np.array([qqid[q_of[int(i)]] for i in item[idx]])}


def gcv_logistic(X, y, g, seed, C=0.5):
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(g, 5, seed):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return auroc(oof, y)


out = {}
t0 = time.time()
for m in MODELS:
    z, meta = load(f"{m}_v2"); L = apriori_layer(meta)
    XL = z.layer(L)
    idx1 = np.where(m1)[0]
    X, y = XL[idx1], S_all[idx1]
    e = {"layer": int(L)}
    for name, gf in G.items():
        g = gf(idx1)
        v = [gcv_logistic(X, y, g, s) for s in range(NSEED)]
        e["logistic_" + name] = stat(v)
        print(f"  {m} logistic {name:<16} {np.mean(v):.4f}+-{np.std(v, ddof=1):.4f} [{time.time()-t0:.0f}s]", flush=True)

    # --- cross column ---
    iAD, iAC = np.where(mAD)[0], np.where(mAC)[0]
    d = fit_direction(XL[iAD], S_all[iAD])
    e["cross_as_shipped"] = float(auroc(project(XL[iAC], d), S_all[iAC]))
    for name, gf in G.items():
        vals = []
        for s in range(NSEED):
            gAC = gf(iAC)
            oof = np.zeros(len(iAC))
            for tr, te in seeded_group_folds(gAC, 5, s):
                heldout_items = set(item[iAC[te]].tolist())
                trAD = iAD[~np.isin(item[iAD], list(heldout_items))]
                dd = fit_direction(XL[trAD], S_all[trAD])
                oof[te] = project(XL[iAC[te]], dd)
            vals.append(auroc(oof, S_all[iAC]))
        e["cross_grouped_" + name] = stat(vals)
        print(f"  {m} cross    {name:<16} {np.mean(vals):.4f}+-{np.std(vals, ddof=1):.4f} "
              f"(as-shipped {e['cross_as_shipped']:.4f})", flush=True)
    out[m] = e
    z.drop(f"last_{L}"); del XL, X
json.dump(out, open("runs/cr_dupleak_sensitivity.json", "w"), indent=2)
print("wrote runs/cr_dupleak_sensitivity.json")
