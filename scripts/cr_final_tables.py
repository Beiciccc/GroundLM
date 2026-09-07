"""Every camera-ready table number, under ONE corrected protocol.

The area chair asked for exactly this: "Regenerate all figures/tables with one
source-grouped, fold-local, signed-AUROC protocol; deduplicate by source QA before
splitting donor questions."

The protocol, applied uniformly:

  GROUPING       CtrlPairs folds are grouped by the normalised source (question,
                 context, answer) triple, not by item_id. item_id only indexes an
                 NQ-Swap swap record and several records share one source QA, so the
                 1,200 cell-A rows cover 555 distinct triples and 903 of them had a
                 byte-identical twin in another fold. RAGTruth folds are grouped by
                 normalised source passage: the 2,000 released rows cover 450 passages
                 with 4-5 responses each, while item_id is a per-row counter, so the
                 previous "GroupKFold on item_id" was row-level CV.

  FOLD-LOCAL     The confidence / surface residualiser is fitted on the training fold
                 and applied to that fold's test rows. It used to be fitted once on
                 every row before cross-validation, which is transductive.

  SIGNED AUROC   Polarity is fixed by the training fold; the test projection is scored
                 with plain roc_auc_score. No max(AUC, 1-AUC) anywhere.

  SEEDS          Folds come from an explicitly seeded group splitter (sklearn's
                 GroupKFold assigns folds by argsort over group sizes, which is
                 tie-broken differently across numpy versions and is therefore not
                 reproducible here). Every estimate is mean +- sd over NSEED splits.

  HELD-OUT       The cross-strata control trains on A-vs-D and evaluates on A-vs-C from
                 DISJOINT source groups. It previously trained and evaluated on the
                 same 1,200 A rows.

Usage:  PYTHONPATH=src python scripts/cr_final_tables.py
Writes: runs/cr_final_tables.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project          # noqa: E402
from _cr_common import (load, apriori_layer, auroc_signed as auroc,    # noqa: E402
                        seeded_group_folds, collision_map, stat,
                        source_groups, passage_groups)

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
RT_MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
NSEED = int(os.environ.get("NSEED", "10"))
NBOOT = int(os.environ.get("NBOOT", "2000"))
OUTP = os.environ.get("OUT", "runs/cr_final_tables.json")


# ---------- fold-local residualiser ----------
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


def cv_oof(X, y, groups, seed, estimator="massmean", C=None, folds=5, Creg=0.5):
    """Out-of-fold scores under a seeded grouped split; residualiser fitted per fold."""
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(groups, folds, seed):
        if C is None:
            Xtr, Xte = X[tr], X[te]
        else:
            f = purge_fit(X[tr], C[tr])
            Xtr, Xte = purge_apply(X[tr], C[tr], f), purge_apply(X[te], C[te], f)
        if estimator == "massmean":
            oof[te] = project(Xte, fit_direction(Xtr, y[tr]))
        else:
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=Creg, max_iter=2000).fit(sc.transform(Xtr), y[tr])
            oof[te] = clf.decision_function(sc.transform(Xte))
    return oof


def cluster_ci(score, y, groups, B=NBOOT, seed=1):
    """Cluster bootstrap over GROUPS (source QA / source passage), not rows."""
    y = np.asarray(y).astype(int)
    uniq = np.unique(groups)
    idx_of = {u: np.where(groups == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        gs = rng.choice(uniq, len(uniq))
        idx = np.concatenate([idx_of[g] for g in gs])
        if len(np.unique(y[idx])) == 2:
            vals.append(auroc(score[idx], y[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired(a, b, y, groups, B=NBOOT, seed=2):
    y = np.asarray(y).astype(int)
    uniq = np.unique(groups)
    idx_of = {u: np.where(groups == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        gs = rng.choice(uniq, len(uniq))
        idx = np.concatenate([idx_of[g] for g in gs])
        if len(np.unique(y[idx])) == 2:
            d.append(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
    d = np.asarray(d)
    return {"diff": float(d.mean()),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_a_not_greater": float((d <= 0).mean())}


out = {"nseed": NSEED, "n_boot": NBOOT, "numpy": np.__version__,
       "protocol": "source/passage-grouped folds, fold-local purge, signed AUROC",
       "table1": {}, "table3": {}, "table4_pertask": {}, "prose": {}}
t0 = time.time()

# ================= Table 1 : C1 =================
print("=== Table 1 (C1): source-QA grouped, fold-local purge, signed ===", flush=True)
coll = collision_map()
for m in MODELS4:
    data, meta = load(f"{m}_v2")
    L = apriori_layer(meta)
    O = data["overlap_measured"].astype(int)
    cell = data["cell"]; item = data["item_id"]
    is_coll = np.array([bool(coll.get(int(item[k]), False)) for k in range(len(O))])
    keep = ~((cell == "C") & is_coll)
    m1 = (O == 1) & keep
    gsrc_all = source_groups(item)

    X = data.layer(L)[m1]
    S = data["support"].astype(int)[m1]
    g = gsrc_all[m1]
    conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"],
                     data["first_maxsoftmax"]], 1)[m1]

    raw_v, pur_v = [], []
    for s in range(NSEED):
        raw_v.append(auroc(cv_oof(X, S, g, s), S))
        pur_v.append(auroc(cv_oof(X, S, g, s, C=conf), S))
    oof0 = cv_oof(X, S, g, 0)
    ci_raw = cluster_ci(oof0, S, g)

    # held-out cross-strata: train A-vs-D on train groups, evaluate A-vs-C on held-out groups
    Xall = data.layer(L); Sall = data["support"].astype(int)
    mAD = np.isin(cell, ["A", "D"]); mAC = np.isin(cell, ["A", "C"]) & keep
    cross_v = []
    for s in range(NSEED):
        vals = []
        for tr, te in seeded_group_folds(np.unique(gsrc_all), 5, s):
            tr_groups = set(np.unique(gsrc_all)[tr].tolist())
            in_tr = np.array([x in tr_groups for x in gsrc_all])
            fit_mask = mAD & in_tr
            ev_mask = mAC & ~in_tr
            if len(np.unique(Sall[fit_mask])) < 2 or len(np.unique(Sall[ev_mask])) < 2:
                continue
            d = fit_direction(Xall[fit_mask], Sall[fit_mask])
            vals.append(auroc(project(Xall[ev_mask], d), Sall[ev_mask]))
        cross_v.append(float(np.mean(vals)))

    out["table1"][m] = {
        "layer": int(L), "n_A": int((cell[m1] == "A").sum()), "n_C": int((cell[m1] == "C").sum()),
        "n_source_groups": int(len(np.unique(g))),
        "supp_O1_raw": stat(raw_v), "supp_O1_raw_ci": ci_raw,
        "supp_O1_purged": stat(pur_v),
        "cross_strata_heldout": stat(cross_v),
    }
    r = out["table1"][m]
    print(f"  {m:15s} L{L:<3} raw={r['supp_O1_raw']['mean']:.4f}+-{r['supp_O1_raw']['sd']:.4f} "
          f"CI[{ci_raw[0]:.3f},{ci_raw[1]:.3f}]  +conf={r['supp_O1_purged']['mean']:.4f}"
          f"+-{r['supp_O1_purged']['sd']:.4f}  cross(held-out)={r['cross_strata_heldout']['mean']:.4f}"
          f"+-{r['cross_strata_heldout']['sd']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    data.drop(f"last_{L}")
    json.dump(out, open(OUTP, "w"), indent=2, default=float)

# ================= Tables 3 & 4 : C4 =================
print("=== Tables 3/4 (C4): passage grouped, fold-local purge, signed ===", flush=True)
gpass = passage_groups()
for m in RT_MODELS:
    cp, cpm = load(f"{m}_v2")
    rt, rtm = load(f"{m}_rt")
    L = min(set(cpm["layers"]) & set(rtm["layers"]), key=lambda L: abs(L - 0.5 * cpm["n_layers"]))

    d_sup = fit_direction(cp.layer(L), cp["support"].astype(int))   # polarity from CtrlPairs
    Xrt = rt.layer(L)
    y = rt["faithful"].astype(int)
    tt = rt["task_type"]
    ov = rt["lexical_overlap"].astype(np.float64)
    lenk = "answer_tok_len" if "answer_tok_len" in rt else "answer_tok_len_ws"
    surf = np.stack([ov, rt[lenk].astype(float)], 1)
    conf_s = rt["mean_maxsoftmax"].astype(np.float64)
    nli = rt["nli_entail_prob"].astype(np.float64) if "nli_entail_prob" in rt else None

    synth = project(Xrt, d_sup)
    ind_v = [auroc(cv_oof(Xrt, y, gpass, s, estimator="logistic"), y) for s in range(NSEED)]
    noov_v = [auroc(cv_oof(Xrt, y, gpass, s, estimator="logistic", C=surf), y) for s in range(NSEED)]
    oof0 = cv_oof(Xrt, y, gpass, 0, estimator="logistic")

    rec = {"layer": int(L), "n": int(len(y)), "n_passage_groups": int(len(np.unique(gpass))),
           "synth_dS": auroc(synth, y),
           "confidence": auroc(conf_s, y),
           "lexical_overlap": auroc(ov, y),
           "in_domain": stat(ind_v), "in_domain_ci": cluster_ci(oof0, y, gpass),
           "in_domain_no_overlap": stat(noov_v),
           "in_domain_vs_overlap": paired(oof0, ov, y, gpass),
           "synth_vs_overlap": paired(synth, ov, y, gpass),
           "synth_vs_confidence": paired(synth, conf_s, y, gpass)}
    if nli is not None:
        rec["nli"] = auroc(nli, y)
        rec["synth_vs_nli"] = paired(synth, nli, y, gpass)
    out["table3"][m] = rec

    per = {}
    for tk in sorted(set(tt.tolist())):
        mk = tt == tk
        per[tk] = {"n": int(mk.sum()), "synth_dS": auroc(synth[mk], y[mk]),
                   "confidence": auroc(conf_s[mk], y[mk]),
                   "lexical_overlap": auroc(ov[mk], y[mk]),
                   "in_domain": auroc(oof0[mk], y[mk]),
                   "faithful_rate": float(y[mk].mean())}
        if nli is not None:
            per[tk]["nli"] = auroc(nli[mk], y[mk])
    out["table4_pertask"][m] = per

    print(f"  {m:15s} L{L:<3} synth={rec['synth_dS']:.4f} conf={rec['confidence']:.4f} "
          f"in-dom={rec['in_domain']['mean']:.4f}+-{rec['in_domain']['sd']:.4f} "
          f"CI[{rec['in_domain_ci'][0]:.3f},{rec['in_domain_ci'][1]:.3f}] "
          f"-overlap={rec['in_domain_no_overlap']['mean']:.4f}"
          f"+-{rec['in_domain_no_overlap']['sd']:.4f} "
          f"ovl={rec['lexical_overlap']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(OUTP, "w"), indent=2, default=float)

# macro summaries over the three RAGTruth models
tasks = sorted(out["table4_pertask"][RT_MODELS[0]].keys())
sigs = ["synth_dS", "confidence", "in_domain", "lexical_overlap"] + (["nli"] if "nli" in out["table3"][RT_MODELS[0]] else [])
out["prose"]["per_task_mean_over_models"] = {
    s: {**{t: float(np.mean([out["table4_pertask"][m][t][s] for m in RT_MODELS])) for t in tasks},
        "macro": float(np.mean([np.mean([out["table4_pertask"][m][t][s] for m in RT_MODELS]) for t in tasks]))}
    for s in sigs}
out["prose"]["faithful_rate_per_task"] = {
    t: float(np.mean([out["table4_pertask"][m][t]["faithful_rate"] for m in RT_MODELS])) for t in tasks}
for s in sigs:
    d = out["prose"]["per_task_mean_over_models"][s]
    print(f"  {s:18s} " + " ".join(f"{t}={d[t]:.3f}" for t in tasks) + f"  macro={d['macro']:.3f}")

json.dump(out, open(OUTP, "w"), indent=2, default=float)
print(f"\nwrote {OUTP} [{time.time()-t0:.0f}s]")
