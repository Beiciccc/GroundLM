"""What Table 1 becomes if the duplicate A rows are actually REMOVED (not just regrouped).

Keeps one A row per unique normalised (question, context, answer) source triple
(1200 -> 555 A rows), keeps the collision-free C rows unchanged, and re-runs the
paper's supp_O1 raw / conf-purged mass-mean grouped CV.
"""
from __future__ import annotations
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from _cr_common import load, apriori_layer, auroc_signed as auroc, seeded_group_folds, collision_map, stat
from groundlm.probe.directions import fit_direction, project
from groundlm.probe.confidence import purge

rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
norm = lambda x: " ".join(str(x).lower().split())
cell = np.array([r["cell"] for r in rows]); item = np.array([r["item_id"] for r in rows])
O = np.array([r["overlap_measured"] for r in rows])
S_all = np.array([r["support"] for r in rows]).astype(int)
coll = collision_map()
keepC = ~((cell == "C") & np.array([bool(coll.get(int(i), False)) for i in item]))
m1 = (O == 1) & keepC

byitem = {}
for r in rows:
    byitem.setdefault(r["item_id"], {})[r["cell"]] = r
qca_of = {i: (norm(d["A"]["question"]), norm(d["A"]["context"]), norm(d["A"]["asserted_answer"]))
          for i, d in byitem.items()}
qid = {k: n for n, k in enumerate(sorted(set(qca_of.values())))}

# keep the FIRST A row of each unique source triple
seen, dropA = set(), np.zeros(len(rows), bool)
for k in range(len(rows)):
    if cell[k] == "A":
        key = qca_of[int(item[k])]
        if key in seen:
            dropA[k] = True
        else:
            seen.add(key)
mded = m1 & ~dropA
idx = np.where(mded)[0]
g_item = item[idx]
g_qca = np.array([qid[qca_of[int(i)]] for i in item[idx]])
print(f"dedup subset: {len(idx)} rows  (A={int((cell[idx]=='A').sum())}, C={int((cell[idx]=='C').sum())}); "
      f"item groups={len(np.unique(g_item))}, qca groups={len(np.unique(g_qca))}")


def gcv(X, y, g, seed):
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(g, 5, seed):
        oof[te] = project(X[te], fit_direction(X[tr], y[tr]))
    return auroc(oof, y)


out = {"n": len(idx), "n_A": int((cell[idx] == 'A').sum()), "n_C": int((cell[idx] == 'C').sum()), "models": {}}
for m in ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]:
    z, meta = load(f"{m}_v2"); L = apriori_layer(meta)
    X = z.layer(L)[idx]; z.drop(f"last_{L}")
    y = S_all[idx]
    conf = np.stack([z["mean_logprob"], z["mean_maxsoftmax"], z["first_maxsoftmax"]], 1)[idx]
    Xp = purge(X, conf)
    e = {"layer": int(L)}
    for name, g in [("item_id", g_item), ("source_qca", g_qca)]:
        r = [gcv(X, y, g, s) for s in range(10)]
        p = [gcv(Xp, y, g, s) for s in range(10)]
        e[name] = {"raw": stat(r), "purged": stat(p)}
        print(f"  {m} dedup {name:<12} raw {np.mean(r):.4f}+-{np.std(r,ddof=1):.4f}  "
              f"purged {np.mean(p):.4f}+-{np.std(p,ddof=1):.4f}", flush=True)
    out["models"][m] = e
    del X, Xp
json.dump(out, open("runs/cr_dupleak_dedup.json", "w"), indent=2)
print("wrote runs/cr_dupleak_dedup.json")
