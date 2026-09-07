"""Camera-ready protocol audit, part 4: where does max(AUC,1-AUC) actually bind?

Recomputes, signed and flipped side by side:
  * C3 / Table tab:c3   -- harden_analyses.py section C (12 ordered pairs x 3 seeds)
  * C4 / Table tab:c4   -- RAGTruth in-domain probe + no-overlap variant
  * C4 / tab:c4_pertask -- per-task confidence / in-domain / synth-dS
Writes runs/cr_protocol_audit_flip.json
"""
from __future__ import annotations
import json, os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from _cr_common import load, apriori_layer, auroc, auroc_signed        # noqa: E402
from groundlm.probe.directions import fit_direction, project, _unit    # noqa: E402
from groundlm.probe.confidence import purge                            # noqa: E402
from groundlm.transfer.procrustes import fit_map, transport_direction  # noqa: E402
from sklearn.linear_model import LogisticRegression                    # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402
from sklearn.model_selection import GroupKFold                         # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
RT = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
out = {}

# ---------------- C3 (exact harden_analyses.py section C) ----------------
RUN = {}
for m in MODELS4:
    z, meta = load(f"{m}_v2")
    L = apriori_layer(meta)
    RUN[m] = {"X": z.layer(L), "S": z["support"].astype(int),
              "F": z["factuality"].astype(int), "item": z["item_id"]}
    del z


def split(item, seed):
    u = np.unique(item); r = np.random.default_rng(seed); r.shuffle(u)
    cal = set(u[: len(u) // 2].tolist())
    mask = np.array([i in cal for i in item])
    return mask, ~mask


agg = {k: [] for k in ["auroc", "within", "auroc_fac", "within_fac"]}
sgn = {k: [] for k in agg}
for s, t in itertools.permutations(MODELS4, 2):
    for seed in range(3):
        Sd, Td = RUN[s], RUN[t]
        cal, ev = split(Sd["item"], seed)
        dS = fit_direction(Sd["X"][cal], Sd["S"][cal])
        A = fit_map(Td["X"][cal], Sd["X"][cal], orthogonal=True)
        tr = transport_direction(A, dS)
        dTnat = fit_direction(Td["X"][cal], Td["S"][cal])
        for k, sc, yy in [("auroc", project(Td["X"][ev], tr), Td["S"][ev]),
                          ("within", project(Td["X"][ev], dTnat), Td["S"][ev])]:
            agg[k].append(auroc(sc, yy)); sgn[k].append(auroc_signed(sc, yy))
        dFs = fit_direction(Sd["X"][cal], Sd["F"][cal])
        dFt = fit_direction(Td["X"][cal], Td["F"][cal])
        trF = transport_direction(A, dFs)
        for k, sc, yy in [("auroc_fac", project(Td["X"][ev], trF), Td["F"][ev]),
                          ("within_fac", project(Td["X"][ev], dFt), Td["F"][ev])]:
            agg[k].append(auroc(sc, yy)); sgn[k].append(auroc_signed(sc, yy))
out["c3"] = {k: {"flipped_mean": float(np.mean(agg[k])), "signed_mean": float(np.mean(sgn[k])),
                 "n_below_half_signed": int(sum(1 for v in sgn[k] if v < 0.5)),
                 "min_signed": float(min(sgn[k])), "n_runs": len(sgn[k])} for k in agg}
print("C3:", json.dumps(out["c3"], indent=1), flush=True)
del RUN

# ---------------- C4 RAGTruth ----------------


def gcv_logistic(X, y, groups, folds=5, C=0.5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return oof


out["c4"] = {}
for m in RT:
    zrt, mrt = load(f"{m}_rt")
    zcp, mcp = load(f"{m}_v2")
    L = min(set(mcp["layers"]) & set(mrt["layers"]), key=lambda L: abs(L - 0.5 * mcp["n_layers"]))
    Xrt = zrt.layer(L); y = zrt["faithful"].astype(int); item = zrt["item_id"]; tt = zrt["task_type"]
    ov = zrt["lexical_overlap"].astype(float)
    lenk = "answer_tok_len" if "answer_tok_len" in zrt else "answer_tok_len_ws"
    surf = np.stack([ov, zrt[lenk].astype(float)], 1)
    oof = gcv_logistic(Xrt, y, item)
    oof_no = gcv_logistic(purge(Xrt, surf), y, item)
    d_sup = fit_direction(zcp.layer(L), zcp["support"].astype(int))
    s_synth = project(Xrt, d_sup)
    conf = zrt["mean_maxsoftmax"].astype(float)
    r = {"layer": int(L),
         "in_domain_flipped": auroc(oof, y), "in_domain_signed": auroc_signed(oof, y),
         "in_domain_no_overlap_flipped": auroc(oof_no, y),
         "in_domain_no_overlap_signed": auroc_signed(oof_no, y), "per_task": {}}
    for tk in sorted(set(tt.tolist())):
        mk = tt == tk
        r["per_task"][tk] = {
            "in_domain_flipped": auroc(oof[mk], y[mk]), "in_domain_signed": auroc_signed(oof[mk], y[mk]),
            "confidence_flipped": auroc(conf[mk], y[mk]), "confidence_signed": auroc_signed(conf[mk], y[mk]),
            "synth_dS_flipped": auroc(s_synth[mk], y[mk]), "synth_dS_signed": auroc_signed(s_synth[mk], y[mk])}
    out["c4"][m] = r
    print(f"{m}: in-dom {r['in_domain_flipped']:.4f}/{r['in_domain_signed']:.4f} "
          f"no-ov {r['in_domain_no_overlap_flipped']:.4f}/{r['in_domain_no_overlap_signed']:.4f}", flush=True)
    del zrt, zcp, Xrt

# macro rows exactly as Table tab:c4_pertask builds them (mean over the 3 models)
tasks = sorted(out["c4"][RT[0]]["per_task"].keys())
macro = {}
for sig in ["synth_dS", "confidence", "in_domain"]:
    row_f = {tk: float(np.mean([out["c4"][m]["per_task"][tk][f"{sig}_flipped"] for m in RT])) for tk in tasks}
    row_s = {tk: float(np.mean([out["c4"][m]["per_task"][tk][f"{sig}_signed"] for m in RT])) for tk in tasks}
    row_f["macro"] = float(np.mean(list(row_f.values())))
    row_s["macro"] = float(np.mean([row_s[t] for t in tasks]))
    macro[sig] = {"published_flipped": row_f, "signed": row_s}
out["c4_pertask_table"] = macro
print(json.dumps(macro, indent=1), flush=True)

json.dump(out, open("runs/cr_protocol_audit_flip.json", "w"), indent=2, default=float)
print("wrote runs/cr_protocol_audit_flip.json")
