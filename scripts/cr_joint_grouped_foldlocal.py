"""The one configuration the audit never computed: source/passage grouping AND fold-local purge together.

Two of the reviewers' corrections push the same numbers in OPPOSITE directions, and each
was only ever measured in isolation:

  Sec 5.4 "residualizing lexical overlap out of the features first drops it to 0.60-0.66"
      published (item-grouped, global purge) : 0.596 / 0.666 / 0.633
      passage-grouped, global purge          : 0.548 / 0.587 / 0.583   (down)
      item-grouped, fold-local purge         : 0.689 / 0.751 / 0.721   (up)
      passage-grouped + fold-local           : <- computed here

  Table 1 "+conf" column
      published (item_id-grouped, global purge)
      source-QA-grouped + fold-local          : <- computed here

Fold-local purge means the OLS residualizer (and the confound standardisation it uses)
is fitted on the training fold only and applied to that fold's test rows, instead of
being fitted once on every row before cross-validation.

Usage:  PYTHONPATH=src python scripts/cr_joint_grouped_foldlocal.py
Writes: runs/cr_joint_grouped_foldlocal.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project        # noqa: E402
from _cr_common import (load, apriori_layer, auroc, auroc_signed,    # noqa: E402
                        seeded_group_folds, collision_map, stat)

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
RT_MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
NSEED = int(os.environ.get("NSEED", "10"))
OUTP = os.environ.get("OUT", "runs/cr_joint_grouped_foldlocal.json")


def norm(x):
    return " ".join(str(x).lower().split())


# ---------------- fold-local residualisation ----------------
def purge_fit(X_tr, C_tr):
    """Fit the OLS residualiser on the TRAINING fold only."""
    C = np.asarray(C_tr, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    mu, sd = C.mean(0), C.std(0) + 1e-8
    A = np.concatenate([np.ones((len(C), 1)), (C - mu) / sd], axis=1)
    beta, *_ = np.linalg.lstsq(A, np.asarray(X_tr, dtype=np.float64), rcond=None)
    return beta, mu, sd


def purge_apply(X, C, fit):
    beta, mu, sd = fit
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    A = np.concatenate([np.ones((len(C), 1)), (C - mu) / sd], axis=1)
    return np.asarray(X, dtype=np.float64) - A @ beta


def cv_massmean_foldlocal(X, y, C, groups, seed, folds=5):
    """Mass-mean probe with the purge fitted inside each training fold."""
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(groups, folds, seed):
        f = purge_fit(X[tr], C[tr])
        Xtr, Xte = purge_apply(X[tr], C[tr], f), purge_apply(X[te], C[te], f)
        oof[te] = project(Xte, fit_direction(Xtr, y[tr]))
    return auroc(oof, y)


def cv_logistic_foldlocal(X, y, C, groups, seed, folds=5, Creg=0.5):
    """L2-logistic probe with the purge fitted inside each training fold."""
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(groups, folds, seed):
        f = purge_fit(X[tr], C[tr])
        Xtr, Xte = purge_apply(X[tr], C[tr], f), purge_apply(X[te], C[te], f)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=Creg, max_iter=2000).fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.decision_function(sc.transform(Xte))
    return auroc(oof, y)


out = {"nseed": NSEED, "numpy": np.__version__, "table1_plus_conf": {}, "rt_in_domain_no_overlap": {}}
t0 = time.time()

# ============ A. Table 1 "+conf": source-QA grouping x fold-local purge ============
print("=== A. Table 1 +conf : source-QA grouped, fold-local purge ===", flush=True)
rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
# source-QA key per item_id, taken from the cell-A row (question, context, answer)
srckey = {}
for r in rows:
    if r["cell"] == "A":
        srckey[r["item_id"]] = (norm(r["question"]) + "||" + norm(r["context"])
                                + "||" + norm(r["asserted_answer"]))
coll = collision_map()

for m in MODELS4:
    data, meta = load(f"{m}_v2")
    L = apriori_layer(meta)
    O = data["overlap_measured"].astype(int)
    cell = data["cell"]; item = data["item_id"]
    is_coll = np.array([bool(coll.get(int(item[k]), False)) for k in range(len(O))])
    keep = ~((cell == "C") & is_coll)
    m1 = (O == 1) & keep
    X = data.layer(L)[m1]
    S = data["support"].astype(int)[m1]
    conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"],
                     data["first_maxsoftmax"]], 1)[m1]
    it = item[m1]
    g_item = it
    g_src = np.array([srckey.get(int(i), f"__{i}") for i in it])

    v_item = [cv_massmean_foldlocal(X, S, conf, g_item, s) for s in range(NSEED)]
    v_src = [cv_massmean_foldlocal(X, S, conf, g_src, s) for s in range(NSEED)]
    out["table1_plus_conf"][m] = {
        "layer": int(L),
        "n_groups_item": int(len(np.unique(g_item))),
        "n_groups_source": int(len(np.unique(g_src))),
        "item_grouped_foldlocal": stat(v_item),
        "source_grouped_foldlocal": stat(v_src),
    }
    r = out["table1_plus_conf"][m]
    print(f"  {m:15s} L{L:<3} groups {r['n_groups_item']}->{r['n_groups_source']}  "
          f"item+foldlocal={r['item_grouped_foldlocal']['mean']:.4f}"
          f"+-{r['item_grouped_foldlocal']['sd']:.4f}   "
          f"SOURCE+foldlocal={r['source_grouped_foldlocal']['mean']:.4f}"
          f"+-{r['source_grouped_foldlocal']['sd']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    data.drop(f"last_{L}")
    json.dump(out, open(OUTP, "w"), indent=2, default=float)

# ============ B. RAGTruth in_domain_no_overlap: passage grouping x fold-local purge ============
print("=== B. RAGTruth in-domain, overlap residualised : passage grouped, fold-local purge ===", flush=True)
rt_rows = [json.loads(l) for l in open("data/ragtruth.jsonl")]
ctx_key = [norm(r.get("context", r.get("passage", ""))) for r in rt_rows]

for m in RT_MODELS:
    rt, rtm = load(f"{m}_rt")
    L = apriori_layer(rtm)
    X = rt.layer(L)
    y = rt["faithful"].astype(int)
    ov = rt["lexical_overlap"].astype(np.float64)
    lenk = "answer_tok_len" if "answer_tok_len" in rt else "answer_tok_len_ws"
    surf = np.stack([ov, rt[lenk].astype(float)], 1)
    g_item = rt["item_id"]
    assert len(ctx_key) == len(y), f"ragtruth.jsonl rows {len(ctx_key)} != features {len(y)}"
    g_pass = np.array(ctx_key)

    # headline (no residualisation) for reference, under both groupings
    def cv_plain(groups, seed):
        oof = np.zeros(len(y))
        for tr, te in seeded_group_folds(groups, 5, seed):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(X[te]))
        return auroc(oof, y)

    head_item = [cv_plain(g_item, s) for s in range(NSEED)]
    head_pass = [cv_plain(g_pass, s) for s in range(NSEED)]
    noov_item = [cv_logistic_foldlocal(X, y, surf, g_item, s) for s in range(NSEED)]
    noov_pass = [cv_logistic_foldlocal(X, y, surf, g_pass, s) for s in range(NSEED)]

    out["rt_in_domain_no_overlap"][m] = {
        "layer": int(L),
        "n_groups_item": int(len(np.unique(g_item))),
        "n_groups_passage": int(len(np.unique(g_pass))),
        "headline_item_grouped": stat(head_item),
        "headline_passage_grouped": stat(head_pass),
        "no_overlap_item_grouped_foldlocal": stat(noov_item),
        "no_overlap_passage_grouped_foldlocal": stat(noov_pass),
        "overlap_baseline_auroc": auroc(ov, y),
    }
    r = out["rt_in_domain_no_overlap"][m]
    print(f"  {m:15s} L{L:<3} groups {r['n_groups_item']}->{r['n_groups_passage']}  "
          f"head item={r['headline_item_grouped']['mean']:.4f} pass={r['headline_passage_grouped']['mean']:.4f} | "
          f"-overlap item+FL={r['no_overlap_item_grouped_foldlocal']['mean']:.4f}"
          f"+-{r['no_overlap_item_grouped_foldlocal']['sd']:.4f}  "
          f"PASS+FL={r['no_overlap_passage_grouped_foldlocal']['mean']:.4f}"
          f"+-{r['no_overlap_passage_grouped_foldlocal']['sd']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(OUTP, "w"), indent=2, default=float)

print(f"\nwrote {OUTP} [{time.time()-t0:.0f}s]")
