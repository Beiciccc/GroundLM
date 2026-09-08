"""Camera-ready control: disentangle TRAINING DOMAIN from ESTIMATOR CAPACITY on RAGTruth.

Table 3 of the submission compares
    synth-$d_S$  = mass-mean direction fitted on CtrlPairs-v2, applied to RAGTruth
    in-domain    = L2-logistic probe fitted on RAGTruth human labels (grouped CV)
which changes the training domain AND the estimator at the same time, so "the
synthetic axis fails / the in-domain probe works" is not attributable to either
factor alone. This script completes the 2x2:

                        mass-mean            logistic
    CtrlPairs -> RT     (= synth-d_S)        NEW
    RAGTruth (CV)       NEW                  (= in-domain)

Fold splits use an explicitly seeded grouped splitter rather than sklearn's
GroupKFold: GroupKFold decides its folds via np.argsort over group sizes, which
is tie-broken differently by different numpy versions (on RAGTruth every
item_id is a singleton, so *all* 2000 groups are tied). Results are reported as
mean +- sd over NSEED splits.

Usage:  PYTHONPATH=src python scripts/cr_estimator_domain_control.py
Writes: runs/cr_estimator_domain_control.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project   # noqa: E402
from _cr_common import load, auroc_signed as auroc, seeded_group_folds, stat, passage_groups     # noqa: E402

RT_MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
NSEED = int(os.environ.get("NSEED", "20"))
OUTP = os.environ.get("OUT", "runs/cr_estimator_domain_control.json")


def cv_score(X, y, groups, estimator, seed, folds=5, C=0.5, max_iter=2000):
    """Out-of-fold scores under a seeded grouped split. estimator in {massmean, logistic}."""
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(groups, folds, seed):
        if estimator == "massmean":
            d = fit_direction(X[tr], y[tr])
            oof[te] = project(X[te], d)
        else:
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(X[te]))
    return oof


out = {"nseed": NSEED, "numpy": np.__version__, "models": RT_MODELS, "per_model": {}}
t0 = time.time()

for m in RT_MODELS:
    cp, cpm = load(f"{m}_v2")
    rt, rtm = load(f"{m}_rt")
    # same shared a-priori layer that reviewer_analyses.py P6d uses for synth-d_S
    L = min(set(cpm["layers"]) & set(rtm["layers"]), key=lambda L: abs(L - 0.5 * cpm["n_layers"]))

    Xcp = cp.layer(L)
    Scp = cp["support"].astype(int)
    Xrt = rt.layer(L)
    faith = rt["faithful"].astype(int)
    # RAGTruth item_id is a per-response counter, so grouping on it is row-level CV.
    # Group by normalized source passage, as every other reported RAGTruth estimate does.
    item = passage_groups()
    ov = rt["lexical_overlap"].astype(np.float64)

    # --- arm 1: CtrlPairs + mass-mean -> RAGTruth (the paper's synth-d_S; deterministic) ---
    d_mm = fit_direction(Xcp, Scp)
    a_cp_mm = auroc(project(Xrt, d_mm), faith)

    # --- arm 2: CtrlPairs + logistic -> RAGTruth (NEW; isolates estimator capacity) ---
    sc_cp = StandardScaler().fit(Xcp)
    clf_cp = LogisticRegression(C=0.5, max_iter=2000).fit(sc_cp.transform(Xcp), Scp)
    a_cp_lr = auroc(clf_cp.decision_function(sc_cp.transform(Xrt)), faith)

    # --- arm 3: RAGTruth + mass-mean, grouped CV (NEW; isolates training domain) ---
    v_rt_mm = [auroc(cv_score(Xrt, faith, item, "massmean", s), faith) for s in range(NSEED)]

    # --- arm 4: RAGTruth + logistic, grouped CV (the paper's in-domain probe) ---
    v_rt_lr = [auroc(cv_score(Xrt, faith, item, "logistic", s), faith) for s in range(NSEED)]

    a_ov = auroc(ov, faith)

    out["per_model"][m] = {
        "layer": int(L),
        "ctrlpairs_massmean_to_rt": a_cp_mm,      # = paper synth-d_S
        "ctrlpairs_logistic_to_rt": a_cp_lr,      # NEW
        "ragtruth_massmean_cv": stat(v_rt_mm),    # NEW
        "ragtruth_logistic_cv": stat(v_rt_lr),    # = paper in-domain
        "lexical_overlap": a_ov,
    }
    print(f"  {m:15s} L={L:2d} | CP+mm->RT={a_cp_mm:.3f}  CP+lr->RT={a_cp_lr:.3f}  "
          f"RT+mm={np.mean(v_rt_mm):.3f}+-{np.std(v_rt_mm, ddof=1):.3f}  "
          f"RT+lr={np.mean(v_rt_lr):.3f}+-{np.std(v_rt_lr, ddof=1):.3f}  overlap={a_ov:.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

# macro over the three RAGTruth models
pm = out["per_model"]
out["mean_over_models"] = {
    "ctrlpairs_massmean_to_rt": float(np.mean([pm[m]["ctrlpairs_massmean_to_rt"] for m in RT_MODELS])),
    "ctrlpairs_logistic_to_rt": float(np.mean([pm[m]["ctrlpairs_logistic_to_rt"] for m in RT_MODELS])),
    "ragtruth_massmean_cv": float(np.mean([pm[m]["ragtruth_massmean_cv"]["mean"] for m in RT_MODELS])),
    "ragtruth_logistic_cv": float(np.mean([pm[m]["ragtruth_logistic_cv"]["mean"] for m in RT_MODELS])),
    "lexical_overlap": float(np.mean([pm[m]["lexical_overlap"] for m in RT_MODELS])),
}
print("\n  mean over models: " + "  ".join(f"{k}={v:.3f}" for k, v in out["mean_over_models"].items()))

json.dump(out, open(OUTP, "w"), indent=2, default=float)
print(f"wrote {OUTP} [{time.time()-t0:.0f}s]")
