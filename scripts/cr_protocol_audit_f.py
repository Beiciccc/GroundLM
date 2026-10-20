"""Camera-ready protocol audit, part 6: (f) transductive surface purge on RAGTruth.

harden_analyses.py:100 computes the "residualize overlap out" number as
    gcv_logistic(purge(X, surf), y, g)
i.e. the OLS residualizer is fitted on ALL 2,000 rows (and the confounds are
standardized with full-sample mean/sd) before the grouped CV. This recomputes it with
the residualizer fitted inside each training fold only.

Writes runs/cr_protocol_audit_f.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from _cr_common import load, apriori_layer, auroc_maxflip as auroc, auroc_signed   # noqa: E402
from groundlm.probe.confidence import purge                       # noqa: E402
from sklearn.linear_model import LogisticRegression               # noqa: E402
from sklearn.preprocessing import StandardScaler                  # noqa: E402
from sklearn.model_selection import GroupKFold                    # noqa: E402

RT = ["qwen25_7b", "llama31_8b", "gemma2_9b"]


def _fit_resid(Xtr, Ctr):
    mu, sd = Ctr.mean(0), Ctr.std(0) + 1e-8
    A = np.concatenate([np.ones((len(Ctr), 1)), (Ctr - mu) / sd], 1)
    beta, *_ = np.linalg.lstsq(A, Xtr, rcond=None)

    def ap(X, C):
        A2 = np.concatenate([np.ones((len(C), 1)), (C - mu) / sd], 1)
        return X - A2 @ beta
    return ap


def gcv_logistic(X, y, groups, conf=None, mode="none", folds=5, C=0.5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    Xuse = purge(X, conf) if mode == "transductive" else X
    for tr, te in GroupKFold(folds).split(X, y, groups):
        if mode == "inductive":
            f = _fit_resid(X[tr], conf[tr]); Xtr, Xte = f(X[tr], conf[tr]), f(X[te], conf[te])
        else:
            Xtr, Xte = Xuse[tr], Xuse[te]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.decision_function(sc.transform(Xte))
    return auroc(oof, y), auroc_signed(oof, y)


out = {}
for m in RT:
    z, meta = load(f"{m}_rt")
    L = apriori_layer(meta)
    X = z.layer(L); y = z["faithful"].astype(int); g = z["item_id"]
    ov = z["lexical_overlap"].astype(float)
    lenk = "answer_tok_len" if "answer_tok_len" in z else "answer_tok_len_ws"
    surf = np.stack([ov, z[lenk].astype(float)], 1)
    t_f, t_s = gcv_logistic(X, y, g, surf, "transductive")
    i_f, i_s = gcv_logistic(X, y, g, surf, "inductive")
    out[m] = {"layer": int(L), "no_overlap_transductive_flipped": t_f,
              "no_overlap_transductive_signed": t_s,
              "no_overlap_inductive_flipped": i_f, "no_overlap_inductive_signed": i_s}
    print(f"{m}: no-overlap transductive={t_f:.4f} inductive={i_f:.4f} (delta {i_f-t_f:+.4f})", flush=True)
    del z, X

json.dump(out, open("runs/cr_protocol_audit_f.json", "w"), indent=2, default=float)
print("wrote runs/cr_protocol_audit_f.json")
