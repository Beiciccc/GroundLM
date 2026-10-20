"""Duplicate-leakage audit of Table 1 (supp_O1 raw / conf-purged, collision-free).

Re-runs the paper's Table-1 headline under three grouping schemes for the grouped CV:
  (i)   item_id                     -- what the paper does (1200 groups)
  (ii)  source-QA triple            -- normalised (question, context, answer) of the
                                       item's cell-A row (555 groups)
  (iii) source question             -- normalised cell-A question (418 groups)
plus two diagnostics:
  (ii-ctl) size-matched random regrouping into groups with the SAME size profile as
           (ii) but assigned at random -- isolates "coarser grouping" from "duplicate
           removal";
  (row)    ungrouped row-level KFold  -- the no-control reference.

Everything else is byte-for-byte the paper's pipeline (scripts/reviewer_analyses.py P4):
a-priori layer at relative depth 0.5, mask (overlap_measured==1) & collision-free cell-C,
mass-mean direction fit on train / projected on test, max(AUC,1-AUC), and purge() applied
to the whole masked sample before CV (deliberately transductive here, to isolate grouping as the only variable; the paper uses a fold-local purge).
"""
from __future__ import annotations
import json, sys, os, collections, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from _cr_common import load, apriori_layer, auroc_signed as auroc, seeded_group_folds, collision_map, stat
from groundlm.probe.directions import fit_direction, project
from groundlm.probe.confidence import purge

MODELS = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
NSEED = 10
DATA = "data/ctrlpairs_v2.jsonl"
OUT = "runs/cr_dupleak_table1.json"

rows = [json.loads(l) for l in open(DATA)]
norm = lambda x: " ".join(str(x).lower().split())
byitem = {}
for r in rows:
    byitem.setdefault(r["item_id"], {})[r["cell"]] = r

# --- group keys, defined per item_id, applied to every row of that item ---
qca_of_item, q_of_item = {}, {}
for i, d in byitem.items():
    a = d["A"]
    qca_of_item[i] = (norm(a["question"]), norm(a["context"]), norm(a["asserted_answer"]))
    q_of_item[i] = norm(a["question"])
qca_id = {k: n for n, k in enumerate(sorted(set(qca_of_item.values())))}
q_id = {k: n for n, k in enumerate(sorted(set(q_of_item.values())))}

coll = collision_map(DATA)
cell = np.array([r["cell"] for r in rows])
item = np.array([r["item_id"] for r in rows])
O = np.array([r["overlap_measured"] for r in rows])
keepC = ~((cell == "C") & np.array([bool(coll.get(int(i), False)) for i in item]))
m1 = (O == 1) & keepC

g_item = item[m1]
g_qca = np.array([qca_id[qca_of_item[int(i)]] for i in item[m1]])
g_q = np.array([q_id[q_of_item[int(i)]] for i in item[m1]])


def size_matched_regroup(seed):
    """Random partition of item_ids into groups whose size profile equals scheme (ii)."""
    sizes = sorted(collections.Counter(qca_of_item[i] for i in byitem).values(), reverse=True)
    rng = np.random.default_rng(1000 + seed)
    ids = rng.permutation(sorted(byitem))
    out, p = {}, 0
    for gi, s in enumerate(sizes):
        for i in ids[p:p + s]:
            out[int(i)] = gi
        p += s
    return np.array([out[int(i)] for i in item[m1]])


def row_folds(n, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.permutation(n) % n_splits
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]


def gcv(X, y, folds):
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = project(X[te], fit_direction(X[tr], y[tr]))
    return auroc(oof, y)


res = {"n_rows": int(m1.sum()),
       "n_groups": {"item_id": int(len(np.unique(g_item))),
                    "source_qca": int(len(np.unique(g_qca))),
                    "source_question": int(len(np.unique(g_q)))},
       "n_seeds": NSEED, "models": {}}
t0 = time.time()
for m in MODELS:
    z, meta = load(f"{m}_v2")
    L = apriori_layer(meta)
    X = z.layer(L)[m1]
    z.drop(f"last_{L}")
    S = np.array([r["support"] for r in rows])[m1].astype(int)
    conf = np.stack([z["mean_logprob"], z["mean_maxsoftmax"], z["first_maxsoftmax"]], 1)[m1]
    Xp = purge(X, conf)
    entry = {"layer": int(L), "n": int(len(S)), "n_pos": int(S.sum())}
    schemes = {"item_id": g_item, "source_qca": g_qca, "source_question": g_q}
    for name, g in schemes.items():
        raw = [gcv(X, S, seeded_group_folds(g, 5, s)) for s in range(NSEED)]
        pur = [gcv(Xp, S, seeded_group_folds(g, 5, s)) for s in range(NSEED)]
        entry[name] = {"raw": stat(raw), "purged": stat(pur)}
        print(f"  {m} {name:<16} raw {np.mean(raw):.4f}+-{np.std(raw, ddof=1):.4f}  "
              f"purged {np.mean(pur):.4f}+-{np.std(pur, ddof=1):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    raw = [gcv(X, S, seeded_group_folds(size_matched_regroup(s), 5, s)) for s in range(NSEED)]
    pur = [gcv(Xp, S, seeded_group_folds(size_matched_regroup(s), 5, s)) for s in range(NSEED)]
    entry["size_matched_random_555"] = {"raw": stat(raw), "purged": stat(pur)}
    print(f"  {m} {'sizematched-ctl':<16} raw {np.mean(raw):.4f}+-{np.std(raw, ddof=1):.4f}  "
          f"purged {np.mean(pur):.4f}+-{np.std(pur, ddof=1):.4f}", flush=True)
    raw = [gcv(X, S, row_folds(len(S), 5, s)) for s in range(NSEED)]
    pur = [gcv(Xp, S, row_folds(len(S), 5, s)) for s in range(NSEED)]
    entry["row_level_ungrouped"] = {"raw": stat(raw), "purged": stat(pur)}
    print(f"  {m} {'row-ungrouped':<16} raw {np.mean(raw):.4f}+-{np.std(raw, ddof=1):.4f}  "
          f"purged {np.mean(pur):.4f}+-{np.std(pur, ddof=1):.4f}", flush=True)
    res["models"][m] = entry
    del X, Xp
json.dump(res, open(OUT, "w"), indent=2)
print("wrote", OUT)
