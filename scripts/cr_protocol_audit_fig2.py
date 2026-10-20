"""Camera-ready protocol audit, part 2: Figure 2 (fig_c1_layers).

Reproduces the published curve exactly (cv_auroc: rng.permutation + np.array_split,
no groups, transductive purge) and recomputes the SAME quantity under
GroupKFold(item_id), plus per-fold signed AUROC to test whether max(AUC,1-AUC) binds.

Writes runs/cr_protocol_audit_fig2.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from _cr_common import load, apriori_layer, auroc_maxflip as auroc, auroc_signed          # noqa: E402
from groundlm.probe.directions import cv_auroc, fit_direction, project   # noqa: E402
from groundlm.probe.confidence import purge                             # noqa: E402
from sklearn.model_selection import GroupKFold                          # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]


def cv_auroc_signed_folds(X, y, folds=5, seed=0):
    """Exactly cv_auroc's splits, but report per-fold signed AUROC too."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    sp = np.array_split(idx, folds)
    fl, sg = [], []
    for k in range(folds):
        te = sp[k]; tr = np.concatenate([sp[j] for j in range(folds) if j != k])
        d = fit_direction(X[tr], y[tr])
        s = project(X[te], d)
        fl.append(auroc(s, y[te])); sg.append(auroc_signed(s, y[te]))
    return float(np.mean(fl)), float(np.mean(sg)), min(sg)


def gcv(X, y, groups, folds=5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y)); per = []
    for tr, te in GroupKFold(folds).split(X, y, groups):
        d = fit_direction(X[tr], y[tr]); oof[te] = project(X[te], d)
        per.append(auroc_signed(oof[te], y[te]))
    return auroc(oof, y), auroc_signed(oof, y), float(np.mean(per)), min(per)


out = {}
for m in MODELS4:
    z, meta = load(f"{m}_v2")
    La = apriori_layer(meta)
    O = z["overlap_measured"].astype(int); m1 = O == 1     # figure retains collisions
    S = z["support"].astype(int)[m1]
    g = z["item_id"][m1]
    conf = np.stack([z["mean_logprob"], z["mean_maxsoftmax"], z["first_maxsoftmax"]], 1).astype(float)[m1]
    rows = {}
    for L in meta["layers"]:
        X = z.layer(L)[m1]
        Xp = purge(X, conf)
        pub_raw = cv_auroc(X, S)                     # the exact published curve value
        pub_pur = cv_auroc(Xp, S)
        f_r, s_r, min_r = cv_auroc_signed_folds(X, S)
        f_p, s_p, min_p = cv_auroc_signed_folds(Xp, S)
        gr_f, gr_s, gr_pf, gr_min = gcv(X, S, g)
        gp_f, gp_s, gp_pf, gp_min = gcv(Xp, S, g)
        rows[str(L)] = {
            "published_raw_ungrouped": pub_raw, "published_purged_ungrouped": pub_pur,
            "raw_ungrouped_signed": s_r, "raw_ungrouped_min_fold_signed": min_r,
            "purged_ungrouped_signed": s_p, "purged_ungrouped_min_fold_signed": min_p,
            "raw_grouped_flipped": gr_f, "raw_grouped_signed": gr_s,
            "purged_grouped_flipped": gp_f, "purged_grouped_signed": gp_s,
        }
        del X, Xp
        z.drop(f"last_{L}")
        print(f"  {m} L{L}: pub_raw={pub_raw:.4f} grp_raw={gr_f:.4f} | "
              f"pub_pur={pub_pur:.4f} grp_pur={gp_f:.4f}", flush=True)

    # green star: no-context, same metric, a-priori layer
    znc, mnc = load(f"{m}_v2_nc")
    Lnc = int(list(mnc["layers"])[0])
    Xnc = znc.layer(Lnc)[m1]
    star_pub = cv_auroc(Xnc, S)
    star_grp = gcv(Xnc, S, g)
    out[m] = {"apriori_layer": int(La), "layers": [int(L) for L in meta["layers"]], "per_layer": rows,
              "nc_layer": Lnc, "nocontext_star_published_ungrouped": star_pub,
              "nocontext_star_grouped_flipped": star_grp[0],
              "nocontext_star_grouped_signed": star_grp[1],
              "ctx_at_apriori_published": rows[str(Lnc)]["published_raw_ungrouped"],
              "ctx_at_apriori_grouped": rows[str(Lnc)]["raw_grouped_flipped"]}
    print(f"{m}: star pub={star_pub:.4f} grouped={star_grp[0]:.4f}; "
          f"ctx pub={rows[str(Lnc)]['published_raw_ungrouped']:.4f} grouped={rows[str(Lnc)]['raw_grouped_flipped']:.4f}",
          flush=True)
    del z, znc

json.dump(out, open("runs/cr_protocol_audit_fig2.json", "w"), indent=2, default=float)
print("wrote runs/cr_protocol_audit_fig2.json")
