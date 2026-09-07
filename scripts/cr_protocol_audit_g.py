"""Camera-ready protocol audit, part 5:

(g) The manuscript (Methods "Confidence purge" + App. "Confidence scalars") states a
    THREE-scalar confound set (mean log-prob, mean max-softmax, first max-softmax).
    groundlm.analyze.CONFOUND_GROUPS["confidence"] still adds the per-row activation
    L2 norm, and build_confounds(..., "confidence") is what feeds the C2 numbers
    (scripts/cr_c2_and_separability.py) and Fig. 3 panel (b).
    -> recompute the C2 confidence asymmetry under BOTH confound sets.

(d) signed vs flipped for the no-context ablation (all 3 prompt regimes).

Writes runs/cr_protocol_audit_g.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from _cr_common import load, apriori_layer, auroc, auroc_signed, collision_map  # noqa: E402
from groundlm.probe.directions import fit_direction, project                    # noqa: E402
from groundlm.probe.confidence import _proj_r2, purge                           # noqa: E402
from sklearn.model_selection import GroupKFold                                  # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
N_BOOT = 200
out = {"c2_confound_set": {}, "nocontext_signed": {}}
t0 = time.time()


def confsets(z, X):
    three = np.stack([z["mean_logprob"], z["mean_maxsoftmax"], z["first_maxsoftmax"]], 1).astype(float)
    four = np.concatenate([three, np.linalg.norm(X, axis=1)[:, None]], 1)
    return {"three_scalars_as_manuscript": three, "four_with_actnorm_as_code": four}


def asym_point(X, S, F, C):
    aS = _proj_r2(project(X, fit_direction(X, S)), C)
    aF = _proj_r2(project(X, fit_direction(X, F)), C)
    return aF - aS, aS, aF


# ---------------- C2 under both confound sets ----------------
for m in MODELS4:
    z, meta = load(f"{m}_v2")
    S = z["support"].astype(int); F = z["factuality"].astype(int)
    n = len(S)
    rows = {k: [] for k in ["three_scalars_as_manuscript", "four_with_actnorm_as_code"]}
    pvals = {k: [] for k in rows}
    for L in meta["layers"]:
        X = z.layer(L)
        cs = confsets(z, X)
        for k, C in cs.items():
            a, _, _ = asym_point(X, S, F, C)
            rows[k].append(a)
            rng = np.random.default_rng(0)
            mm = max(20, int(0.8 * n)); boot = []
            for _ in range(N_BOOT):
                idx = rng.permutation(n)[:mm]
                b, _, _ = asym_point(X[idx], S[idx], F[idx], C[idx])
                boot.append(b)
            pvals[k].append(float((np.asarray(boot) <= 0).mean()))
        z.drop(f"last_{L}")
        del X
        print(f"  [{time.time()-t0:.0f}s] {m} L{L}: 3-scalar={rows['three_scalars_as_manuscript'][-1]:+.4f} "
              f"4-with-norm={rows['four_with_actnorm_as_code'][-1]:+.4f}", flush=True)
    out["c2_confound_set"][m] = {
        k: {"per_layer": rows[k], "mean": float(np.mean(rows[k])), "sd": float(np.std(rows[k], ddof=1)),
            "n_layers": len(rows[k]), "n_negative": int(sum(1 for v in rows[k] if v < 0)),
            "n_sig_opposite": int(sum(1 for p in pvals[k] if p > 0.95)),
            "n_sig_favouring_factuality": int(sum(1 for p in pvals[k] if p < 0.05))}
        for k in rows}
    del z

for k in ["three_scalars_as_manuscript", "four_with_actnorm_as_code"]:
    allv = [v for m in MODELS4 for v in out["c2_confound_set"][m][k]["per_layer"]]
    out.setdefault("c2_pooled", {})[k] = {
        "n_tests": len(allv), "mean": float(np.mean(allv)), "sd": float(np.std(allv, ddof=1)),
        "n_negative": int(sum(1 for v in allv if v < 0)),
        "per_family_mean": [out["c2_confound_set"][m][k]["mean"] for m in MODELS4],
        "n_sig_opposite": int(sum(out["c2_confound_set"][m][k]["n_sig_opposite"] for m in MODELS4)),
        "n_sig_favouring_factuality": int(sum(out["c2_confound_set"][m][k]["n_sig_favouring_factuality"] for m in MODELS4))}
print(json.dumps(out["c2_pooled"], indent=1), flush=True)
json.dump(out, open("runs/cr_protocol_audit_g.json", "w"), indent=2, default=float)

# ---------------- no-context ablation, signed ----------------
COLL = collision_map()
CP = {"qwen25_7b": 14, "mistral7b_v03": 16, "llama31_8b": 16, "gemma2_9b": 21}


def gcv(X, y, groups, folds=5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        d = fit_direction(X[tr], y[tr]); oof[te] = project(X[te], d)
    return auroc(oof, y), auroc_signed(oof, y)


for m in MODELS4:
    L = CP[m]
    res = {}
    for regime, tag in [("normal", f"{m}_v2"), ("no_context", f"{m}_v2_nc"), ("shuffled", f"{m}_v2_shuf")]:
        z, _ = load(tag)
        O = z["overlap_measured"].astype(int); cell = z["cell"]; item = z["item_id"]
        coll = np.array([bool(COLL.get(int(i), False)) for i in item])
        m1 = (O == 1) & ~((cell == "C") & coll)
        X = z.layer(L)[m1]; y = z["support"].astype(int)[m1]; g = item[m1]
        C = np.stack([z["mean_logprob"], z["mean_maxsoftmax"], z["first_maxsoftmax"]], 1).astype(float)[m1]
        rf, rs = gcv(X, y, g)
        pf, ps = gcv(purge(X, C), y, g)
        res[regime] = {"raw_flipped": rf, "raw_signed": rs, "purged_flipped": pf, "purged_signed": ps}
        del z, X
    out["nocontext_signed"][m] = {"layer": L, **res}
    print(f"{m}: " + " ".join(f"{k}(raw {v['raw_flipped']:.3f}/{v['raw_signed']:.3f}, "
                              f"pur {v['purged_flipped']:.3f}/{v['purged_signed']:.3f})"
                              for k, v in res.items()), flush=True)

json.dump(out, open("runs/cr_protocol_audit_g.json", "w"), indent=2, default=float)
print("wrote runs/cr_protocol_audit_g.json")
