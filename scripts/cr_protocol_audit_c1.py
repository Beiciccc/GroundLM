"""Camera-ready protocol audit, part 1: C1 / Table 1 / no-context ablation.

Checks, per model, at the a-priori depth-0.5 layer, on the collision-free sample:
  (d) does max(AUC,1-AUC) bind?  -> reports signed AND flipped for every Table-1 entry
  (e) cross-strata A-vs-D -> A-vs-C leakage: reported vs item-disjoint A version
  (f) transductive purge: purge-on-all vs purge fit inside the train fold only

Writes runs/cr_protocol_audit_c1.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from _cr_common import LazyNpz, load, apriori_layer, collision_map, auroc_maxflip as auroc, auroc_signed  # noqa: E402
from groundlm.probe.directions import fit_direction, project                             # noqa: E402
from groundlm.probe.confidence import purge                                              # noqa: E402
from sklearn.model_selection import GroupKFold                                           # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
COLL = collision_map()


def _purge_fit(Xtr, Ctr):
    """Return a callable applying the OLS residualizer fitted on (Xtr, Ctr) only."""
    mu, sd = Ctr.mean(0), Ctr.std(0) + 1e-8
    A = np.concatenate([np.ones((len(Ctr), 1)), (Ctr - mu) / sd], axis=1)
    beta, *_ = np.linalg.lstsq(A, Xtr, rcond=None)

    def apply(X, C):
        A2 = np.concatenate([np.ones((len(C), 1)), (C - mu) / sd], axis=1)
        return X - A2 @ beta
    return apply


def gcv_oof(X, y, groups, folds=5, conf=None, mode="none"):
    """Grouped-CV out-of-fold mass-mean projections.
    mode: 'none' | 'transductive' (purge all rows first) | 'inductive' (purge fit on train fold)
    """
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    Xuse = X
    if mode == "transductive":
        Xuse = purge(X, conf)
    for tr, te in GroupKFold(folds).split(Xuse, y, groups):
        if mode == "inductive":
            f = _purge_fit(X[tr], conf[tr])
            Xtr, Xte = f(X[tr], conf[tr]), f(X[te], conf[te])
        else:
            Xtr, Xte = Xuse[tr], Xuse[te]
        d = fit_direction(Xtr, y[tr])
        oof[te] = project(Xte, d)
    return oof


out = {}
for m in MODELS4:
    z, meta = load(f"{m}_v2")
    L = apriori_layer(meta)
    cell = z["cell"]; item = z["item_id"]; O = z["overlap_measured"].astype(int)
    S = z["support"].astype(int)
    coll = np.array([bool(COLL.get(int(i), False)) for i in item])
    keepC = ~((cell == "C") & coll)
    m1 = (O == 1) & keepC
    Xall = z.layer(L)
    conf_all = np.stack([z["mean_logprob"], z["mean_maxsoftmax"], z["first_maxsoftmax"]], 1).astype(float)

    X, y, g, C = Xall[m1], S[m1], item[m1], conf_all[m1]
    r = {"layer": int(L), "n": int(m1.sum())}

    # ---- Table 1 col supp_O1 (raw) ----
    oof = gcv_oof(X, y, g)
    r["raw_flipped"] = auroc(oof, y); r["raw_signed"] = auroc_signed(oof, y)
    # ---- Table 1 col +conf : transductive (as published) vs inductive ----
    oof_t = gcv_oof(X, y, g, conf=C, mode="transductive")
    r["purged_transductive_flipped"] = auroc(oof_t, y)
    r["purged_transductive_signed"] = auroc_signed(oof_t, y)
    oof_i = gcv_oof(X, y, g, conf=C, mode="inductive")
    r["purged_inductive_flipped"] = auroc(oof_i, y)
    r["purged_inductive_signed"] = auroc_signed(oof_i, y)

    # ---- Table 1 col cross : as published (A|D fit, A|C eval, A shared) ----
    mAD = np.isin(cell, ["A", "D"])
    mAC = np.isin(cell, ["A", "C"]) & keepC
    dS = fit_direction(Xall[mAD], S[mAD])
    sc = project(Xall[mAC], dS)
    r["cross_published_flipped"] = auroc(sc, S[mAC])
    r["cross_published_signed"] = auroc_signed(sc, S[mAC])
    r["n_A_in_fit"] = int((cell[mAD] == "A").sum())
    r["n_A_in_eval"] = int((cell[mAC] == "A").sum())
    r["n_A_shared"] = int(len(set(item[mAD & (cell == 'A')].tolist())
                              & set(item[mAC & (cell == 'A')].tolist())))

    # ---- cross, leak-free: item-disjoint 5-fold. Fit on A|D of train items,
    #      evaluate on A|C of held-out items. Same estimator, same layer.
    items_u = np.unique(item)
    rngperm = np.random.default_rng(0).permutation(len(items_u))
    fold_of = {items_u[rngperm[i]]: i % 5 for i in range(len(items_u))}
    fold = np.array([fold_of[i] for i in item])
    oof_cross = np.full(len(item), np.nan)
    for k in range(5):
        tr = (fold != k) & mAD
        te = (fold == k) & mAC
        d = fit_direction(Xall[tr], S[tr])
        oof_cross[te] = project(Xall[te], d)
    sel = ~np.isnan(oof_cross)
    r["cross_itemdisjoint_flipped"] = auroc(oof_cross[sel], S[sel])
    r["cross_itemdisjoint_signed"] = auroc_signed(oof_cross[sel], S[sel])
    r["cross_itemdisjoint_n"] = int(sel.sum())

    # ---- cross, C-only evaluation (drop the reused A positives entirely):
    #      how much of 0.66-0.88 is carried by the A rows the probe was trained on?
    #      score A(held-out fold) vs C(held-out fold) is above; here instead the
    #      published direction dS scored on A-rows only vs C-rows only separability.
    mC = (cell == "C") & keepC
    mA = cell == "A"
    sA = project(Xall[mA], dS); sC = project(Xall[mC], dS)
    lab = np.concatenate([np.ones(mA.sum()), np.zeros(mC.sum())])
    r["cross_published_AvsC_check"] = auroc_signed(np.concatenate([sA, sC]), lab)

    out[m] = r
    print(f"{m}: L{L} raw={r['raw_flipped']:.4f}(signed {r['raw_signed']:.4f}) "
          f"purged_T={r['purged_transductive_flipped']:.4f} purged_I={r['purged_inductive_flipped']:.4f} "
          f"cross_pub={r['cross_published_flipped']:.4f} cross_disjoint={r['cross_itemdisjoint_flipped']:.4f}",
          flush=True)
    del Xall, X, z

json.dump(out, open("runs/cr_protocol_audit_c1.json", "w"), indent=2, default=float)
print("wrote runs/cr_protocol_audit_c1.json")
