"""Reviewer-2 hardening analyses — all from cached features, item-grouped CV, no GPU.
Computes the honest confound-controlled numbers and writes runs/hardened.json
(+ wires in-domain into the gate JSONs). Verified targets in comments."""
from __future__ import annotations
import json, os, sys, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project, _unit          # noqa: E402
from groundlm.probe.confidence import purge                                  # noqa: E402
from groundlm.transfer.procrustes import fit_map, transport_direction        # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cr_common import source_groups                                         # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
RT_MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
rng0 = np.random.default_rng(0)


def load(d):
    data = dict(np.load(f"runs/{d}/features.npz", allow_pickle=True))
    meta = json.load(open(f"runs/{d}/features.meta.json"))
    return data, meta


def apriori_layer(m):
    """The a-priori analysis layer = nearest cached layer to relative depth 0.5."""
    meta = json.load(open(f"runs/{m}_v2/features.meta.json"))
    return min(meta["layers"], key=lambda x: abs(x - 0.5 * meta["n_layers"]))


# C1 / cell-C analyses are reported at the a-priori depth-0.5 layer (NOT argmax-peak),
# matching paper Table 1 and the C3/C4 relative-depth convention.
CP = {m: apriori_layer(m) for m in MODELS4}


def auroc(s, y):
    """Signed AUROC (polarity fixed on the train fold). The former max(AUC, 1-AUC)
    convention floored non-predictive directions above 0.5."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, s))


def auroc_signed(s, y):
    """Polarity-honest AUROC (no max(AUC,1-AUC) flip) for near-chance nulls."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, s))


def gcv_massmean(X, y, groups, folds=5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        d = fit_direction(X[tr], y[tr]); oof[te] = project(X[te], d)
    return auroc(oof, y), oof


def gcv_logistic(X, y, groups, folds=5, C=0.5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return auroc(oof, y), oof


def boot_ci(score, y, groups, B=2000):
    y = np.asarray(y).astype(int); uniq = np.unique(groups); aucs = []
    rng = np.random.default_rng(1)
    for _ in range(B):
        gs = rng.choice(uniq, len(uniq)); idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        if len(np.unique(y[idx])) == 2: aucs.append(auroc(score[idx], y[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_p(a, b, y, groups, B=2000):
    y = np.asarray(y).astype(int); uniq = np.unique(groups); diffs = []
    rng = np.random.default_rng(2)
    for _ in range(B):
        gs = rng.choice(uniq, len(uniq)); idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        if len(np.unique(y[idx])) == 2:
            diffs.append(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), float((diffs <= 0).mean())


out = {"in_domain": {}, "c1": {}, "c3": {}, "cellC": {}, "c4_pertask": {}}

# ============ A. in-domain RAGTruth probe (3 models) ============
print("=== A. in-domain RAGTruth probe ===")
for m in RT_MODELS:
    data, meta = load(f"{m}_rt")
    L = min(meta["layers"], key=lambda L: abs(L - 0.5 * meta["n_layers"]))
    X = data[f"last_{L}"].astype(np.float64); y = data["faithful"].astype(int)
    g = data["item_id"]; ov = data["lexical_overlap"].astype(np.float64)
    a_in, oof = gcv_logistic(X, y, g); ci = boot_ci(oof, y, g)
    lenk = "answer_tok_len" if "answer_tok_len" in data else "answer_tok_len_ws"
    surf = np.stack([ov, data[lenk].astype(float)], 1)
    a_no, oof_no = gcv_logistic(purge(X, surf), y, g)
    dmean, dci, p = paired_p(oof, ov, y, g)
    out["in_domain"][m] = {"layer": int(L), "auroc": a_in, "ci": ci, "auroc_no_overlap": a_no,
                           "overlap_auroc": auroc(ov, y), "vs_overlap_diff": dmean, "vs_overlap_ci": dci, "vs_overlap_p": p}
    print(f"  {m}: in-dom={a_in:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] -ov={a_no:.3f} | vs overlap d={dmean:+.3f} p={p:.3f}")
    # Wire into the gate json. NOTE: this block predates the corrected protocol -- it uses
    # GroupKFold over item_id (row-level on RAGTruth) rather than a seeded splitter over
    # source passages, so `in_domain` here (0.767/0.824/0.799) is NOT Table 3's
    # 0.765/0.814/0.797. Table 3 comes from scripts/cr_final_tables.py; these gate reports
    # are kept as the pre-correction record and are cited nowhere in the paper.
    gp = f"runs/{m}_rt/report_gate_ragtruth.json"
    rep = json.load(open(gp)) if os.path.exists(gp) else {}
    rep.setdefault("metrics", {})["in_domain"] = {"auroc": a_in, "ci": ci}
    rep["metrics"]["in_domain_no_overlap"] = {"auroc": a_no}
    rep["in_domain_vs_overlap"] = {"diff": dmean, "ci": dci, "p": p}
    json.dump(rep, open(gp, "w"), indent=2, default=float)

# ============ B. C1 confound-controlled (4 models) ============
print("=== B. C1 confound-controlled ===")
for m, L in CP.items():
    data, meta = load(f"{m}_v2")
    O = data["overlap_measured"].astype(int); m1 = O == 1
    X = data[f"last_{L}"].astype(np.float64)[m1]
    S = data["support"].astype(int)[m1]; F = data["factuality"].astype(int)[m1]
    g = data["item_id"][m1]; cell = data["cell"][m1]
    conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"], data["first_maxsoftmax"]], 1)[m1]
    raw, _ = gcv_massmean(X, S, g)
    purged, _ = gcv_massmean(purge(X, conf), S, g)
    logp = auroc(data["mean_logprob"].astype(float)[m1], S)
    nli = auroc(data["nli_entail_prob"].astype(float)[m1], S) if "nli_entail_prob" in data else float("nan")
    ov_o1 = auroc(data["lexical_overlap"].astype(float)[m1], S)
    sf_ident = bool(np.array_equal(S, F)); sf_corr = float(np.corrcoef(S, F)[0, 1]) if not sf_ident else 1.0
    # cross-strata: train A-vs-D (question fixed = q), eval A-vs-C
    Xall = data[f"last_{L}"].astype(np.float64); cellall = data["cell"]; Sall = data["support"].astype(int)
    mAD = np.isin(cellall, ["A", "D"]); mAC = np.isin(cellall, ["A", "C"])
    dS = fit_direction(Xall[mAD], Sall[mAD]); cross = auroc(project(Xall[mAC], dS), Sall[mAC])
    out["c1"][m] = {"layer": L, "supp_O1_raw": raw, "supp_O1_conf_purged": purged, "logprob_only": logp,
                    "nli_within_O1": nli, "overlap_within_O1": ov_o1, "SF_identical": sf_ident, "SF_corr": sf_corr,
                    "cross_strata_AvsC": cross}
    print(f"  {m}: raw={raw:.3f} conf-purged={purged:.3f} logp={logp:.3f} nli={nli:.3f} ov|O1={ov_o1:.3f} "
          f"S==F={sf_ident} cross-strata={cross:.3f}")

# ============ C. C3 cosine + map-aware/shuffled nulls (source-QA-grouped) ============
# The calibration/evaluation split is grouped by the normalised source QA triple, not by
# item_id: item_id indexes a swap record, and 903 of the 1,200 cell-A rows have an
# identical twin (after case/whitespace normalization) under a different item_id, so an
# item_id split leaves those twins
# on both sides of the boundary. Same grouping as every other reported estimate.
print("=== C. C3 alignment (cosine + nulls, source-QA-grouped) ===")
def c3_runs():
    R = {}
    for m, L in CP.items():
        data, meta = load(f"{m}_v2")
        Lm = min(meta["layers"], key=lambda x: abs(x - 0.5 * meta["n_layers"]))
        R[m] = {"X": data[f"last_{Lm}"].astype(np.float64), "S": data["support"].astype(int),
                "F": data["factuality"].astype(int),
                "item": source_groups(data["item_id"])}
    return R
RUN = c3_runs(); names = list(RUN.keys())
def split(item, seed):
    """Seeded half-split over unique GROUPS (not rows, not GroupKFold's size-sorted folds)."""
    u = np.unique(item); r = np.random.default_rng(seed); r.shuffle(u)
    cal = set(u[: len(u) // 2].tolist()); mask = np.array([i in cal for i in item])
    return mask, ~mask
agg = {k: [] for k in ["cos", "auroc", "within", "rand_map", "rand_lab", "rand_cos",
                        "rand_lab_cos", "cos_fac", "auroc_fac", "within_fac"]}
for s, t in itertools.permutations(names, 2):
    for seed in range(10):
        Sd, Td = RUN[s], RUN[t]
        cal, ev = split(Sd["item"], seed)
        dS = fit_direction(Sd["X"][cal], Sd["S"][cal])
        A = fit_map(Td["X"][cal], Sd["X"][cal], orthogonal=True)        # target->source
        tr = transport_direction(A, dS)                                  # in target space
        dTnat = fit_direction(Td["X"][cal], Td["S"][cal])
        agg["cos"].append(abs(float(np.dot(_unit(tr), _unit(dTnat)))))
        agg["auroc"].append(auroc(project(Td["X"][ev], tr), Td["S"][ev]))
        agg["within"].append(auroc(project(Td["X"][ev], dTnat), Td["S"][ev]))
        # nulls use POLARITY-HONEST (signed) AUROC, matching the methods text; plus cosine-scale nulls
        rr = _unit(np.random.default_rng(seed + 9).standard_normal(Sd["X"].shape[1]))
        tr_r = transport_direction(A, rr)
        agg["rand_map"].append(auroc_signed(project(Td["X"][ev], tr_r), Td["S"][ev]))
        agg["rand_cos"].append(abs(float(np.dot(_unit(tr_r), _unit(dTnat)))))
        ysh = np.random.default_rng(seed + 5).permutation(Sd["S"][cal])
        tr_sh = transport_direction(A, fit_direction(Sd["X"][cal], ysh))
        agg["rand_lab"].append(auroc_signed(project(Td["X"][ev], tr_sh), Td["S"][ev]))
        agg["rand_lab_cos"].append(abs(float(np.dot(_unit(tr_sh), _unit(dTnat)))))
        dFs = fit_direction(Sd["X"][cal], Sd["F"][cal]); dFt = fit_direction(Td["X"][cal], Td["F"][cal])
        trF = transport_direction(A, dFs)
        agg["cos_fac"].append(abs(float(np.dot(_unit(trF), _unit(dFt)))))
        agg["auroc_fac"].append(auroc(project(Td["X"][ev], trF), Td["F"][ev]))
        agg["within_fac"].append(auroc(project(Td["X"][ev], dFt), Td["F"][ev]))
out["c3"] = {k: float(np.mean(v)) for k, v in agg.items()}
print("  " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in agg.items()))

# ============ D. cell-C donor-collision contamination ============
print("=== D. cell-C contamination ===")
rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
byitem = {}
for r in rows: byitem.setdefault(r["item_id"], {})[r["cell"]] = r
norm = lambda x: " ".join(x.lower().split())
collision = {i: (("A" in d and "C" in d) and norm(d["A"]["question"]) == norm(d["C"]["question"])) for i, d in byitem.items()}
nq = len(set(norm(d["A"]["question"]) for d in byitem.values() if "A" in d))
C = [r for r in rows if r["cell"] == "C"]
clean = [r for r in C if not collision.get(r["item_id"])]; cont = [r for r in C if collision.get(r["item_id"])]
fool = lambda L: float(np.mean([r["nli_entail_prob"] > 0.5 for r in L if r.get("nli_entail_prob", -1) >= 0])) if L else float("nan")
ment = lambda L: float(np.mean([r["nli_entail_prob"] for r in L if r.get("nli_entail_prob", -1) >= 0])) if L else float("nan")
out["cellC"] = {"n_items": len(byitem), "n_distinct_q": nq, "n_collision": len(cont),
                "frac_collision": len(cont) / len(C), "fooled_all": fool(C), "fooled_clean": fool(clean),
                "fooled_cont": fool(cont), "mean_ent_cleanC": ment(clean),
                "mean_ent_A": ment([r for r in rows if r["cell"] == "A"]),
                "mean_ent_D": ment([r for r in rows if r["cell"] == "D"])}
# supp_O1 sensitivity (drop contaminated C) per model
for m, L in CP.items():
    data, _ = load(f"{m}_v2"); O = data["overlap_measured"].astype(int)
    keep = np.array([not (data["cell"][k] == "C" and collision.get(int(data["item_id"][k]), False)) for k in range(len(O))])
    m1 = (O == 1) & keep
    raw_clean, _ = gcv_massmean(data[f"last_{L}"].astype(np.float64)[m1], data["support"].astype(int)[m1], data["item_id"][m1])
    out["cellC"][f"supp_O1_clean_{m}"] = raw_clean
S = np.array([r["support"] for r in rows]); O = np.array([r["overlap_measured"] for r in rows])
keepall = np.array([not (r["cell"] == "C" and collision.get(r["item_id"], False)) for r in rows])
out["cellC"]["corr_SO_all"] = float(np.corrcoef(S, O)[0, 1])
out["cellC"]["corr_SO_clean"] = float(np.corrcoef(S[keepall], O[keepall])[0, 1])
print(f"  {len(cont)}/{len(C)} collisions ({len(cont)/len(C)*100:.1f}%), distinct_q={nq}; fooled all={fool(C):.3f} clean={fool(clean):.3f} cont={fool(cont):.3f}")
print(f"  supp_O1 clean: " + " ".join(f"{m}={out['cellC'][f'supp_O1_clean_{m}']:.3f}" for m in CP))

# ============ E. C4 per-task macro (model-independent signals) ============
print("=== E. C4 per-task macro ===")
data, _ = load("qwen25_7b_rt")   # nli/overlap/faithful/task identical across models
tt = data["task_type"]; y = data["faithful"].astype(int)
ov = data["lexical_overlap"].astype(float); nli = data["nli_entail_prob"].astype(float)
tasks = sorted(set(tt.tolist()))
pt = {"faithful_rate": {}, "overlap": {}, "nli": {}}
for tk in tasks:
    mk = tt == tk
    pt["faithful_rate"][tk] = float(y[mk].mean())
    pt["overlap"][tk] = auroc(ov[mk], y[mk]); pt["nli"][tk] = auroc(nli[mk], y[mk])
out["c4_pertask"] = {"tasks": tasks, "per_task": pt,
                     "overlap_pooled": auroc(ov, y), "overlap_macro": float(np.mean(list(pt["overlap"].values()))),
                     "nli_pooled": auroc(nli, y), "nli_macro": float(np.mean(list(pt["nli"].values())))}
print(f"  overlap pooled={out['c4_pertask']['overlap_pooled']:.3f} macro={out['c4_pertask']['overlap_macro']:.3f}; "
      f"nli pooled={out['c4_pertask']['nli_pooled']:.3f} macro={out['c4_pertask']['nli_macro']:.3f}")
print("  faithful-rate:", {k: round(v, 3) for k, v in pt["faithful_rate"].items()})

json.dump(out, open("runs/hardened.json", "w"), indent=2, default=float)
# Legacy pre-correction C1 estimates (item_id folds, transductive purge). Paper Table 1
# comes from scripts/cr_final_tables.py; these values do NOT match it. Kept for the record.
json.dump({m: {"layer": v["layer"], "raw": v["supp_O1_raw"], "purged": v["supp_O1_conf_purged"],
               "logp": v["logprob_only"], "cross": v["cross_strata_AvsC"],
               "clean": out["cellC"].get(f"supp_O1_clean_{m}")} for m, v in out["c1"].items()},
          open("runs/c1_apriori.json", "w"), indent=2, default=float)
print("\nwrote runs/hardened.json + runs/c1_apriori.json (a-priori C1 producer)")
