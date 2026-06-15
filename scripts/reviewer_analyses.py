"""Reviewer-response analyses (all from cached features, item-grouped CV, no GPU).

Reuses harden_analyses.py conventions (a-priori depth-0.5 layer, GroupKFold mass-mean,
orthogonal Procrustes, max-oriented AUROC). Writes runs/reviewer_analyses.json
incrementally (P4 -> P5 -> P6d) so partial results survive interruption.

  P4  clean Table 1: supp_O1 raw / conf-purged / cross-strata with donor-collision
      cell-C items removed (clean becomes the main result);
  P5  Procrustes robustness over the 12 ordered model pairs (one fixed split): the
      transported-to-native cosine for the SUPPORT axis AND for nuisance axes
      (answer length, lexical overlap, confidence, plus factuality). If nuisance axes
      also align at ~0.97, high cosine is a generic matched-stimulus property;
  P6d per-task (Data2txt/QA/Summary) AUROC for synth-d_S, confidence, and the
      in-domain probe (mean over the 3 RAGTruth models), completing Table c4_pertask.
"""
from __future__ import annotations
import json, os, sys, itertools, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project, _unit          # noqa: E402
from groundlm.probe.confidence import purge                                  # noqa: E402
from groundlm.transfer.procrustes import fit_map, transport_direction        # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
RT_MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
OUTP = "runs/reviewer_analyses.json"
_CACHE = {}


def load(d):
    if d not in _CACHE:
        _CACHE[d] = (dict(np.load(f"runs/{d}/features.npz", allow_pickle=True)),
                     json.load(open(f"runs/{d}/features.meta.json")))
    return _CACHE[d]


def apriori_layer(m):
    _, meta = load(f"{m}_v2")
    return min(meta["layers"], key=lambda x: abs(x - 0.5 * meta["n_layers"]))


def auroc(s, y):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def gcv_massmean(X, y, groups, folds=5):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        d = fit_direction(X[tr], y[tr]); oof[te] = project(X[te], d)
    return auroc(oof, y), oof


def gcv_logistic(X, y, groups, folds=5, C=0.5, max_iter=500):
    y = np.asarray(y).astype(int); oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return auroc(oof, y), oof


CP = {m: apriori_layer(m) for m in MODELS4}
out = {"c1_clean": {}, "c3_robust": {}, "c4_pertask_full": {}}


def save():
    json.dump(out, open(OUTP, "w"), indent=2, default=float)


# donor-collision map
rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
byitem = {}
for r in rows:
    byitem.setdefault(r["item_id"], {})[r["cell"]] = r
norm = lambda x: " ".join(x.lower().split())
collision = {i: (("A" in d and "C" in d) and norm(d["A"]["question"]) == norm(d["C"]["question"]))
             for i, d in byitem.items()}

t0 = time.time()
# ============ P4. clean Table 1 ============
print("=== P4. clean C1 (donor-collision cell-C removed) ===", flush=True)
for m, L in CP.items():
    data, _ = load(f"{m}_v2")
    O = data["overlap_measured"].astype(int); cell = data["cell"]; item = data["item_id"]
    coll = np.array([bool(collision.get(int(item[k]), False)) for k in range(len(O))])
    keepC = ~((cell == "C") & coll)
    m1 = (O == 1) & keepC
    X = data[f"last_{L}"].astype(np.float64)[m1]
    S = data["support"].astype(int)[m1]; F = data["factuality"].astype(int)[m1]
    g = item[m1]
    conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"], data["first_maxsoftmax"]], 1)[m1]
    raw, _ = gcv_massmean(X, S, g)
    purged, _ = gcv_massmean(purge(X, conf), S, g)
    Xall = data[f"last_{L}"].astype(np.float64); Sall = data["support"].astype(int)
    mAD = np.isin(cell, ["A", "D"]); mAC = np.isin(cell, ["A", "C"]) & keepC
    dS = fit_direction(Xall[mAD], Sall[mAD]); cross = auroc(project(Xall[mAC], dS), Sall[mAC])
    out["c1_clean"][m] = {"layer": int(L), "supp_O1_raw_clean": raw, "supp_O1_purged_clean": purged,
                          "cross_strata_clean": cross, "SF_identical": bool(np.array_equal(S, F)),
                          "n_A": int((cell[m1] == "A").sum()), "n_C": int((cell[m1] == "C").sum())}
    print(f"  {m}: raw_clean={raw:.3f} purged_clean={purged:.3f} cross_clean={cross:.3f} "
          f"S==F={np.array_equal(S, F)} (nA={int((cell[m1]=='A').sum())}, nC={int((cell[m1]=='C').sum())})", flush=True)
save()

# ============ P5. Procrustes per-pair + nuisance (one fixed split, 12 ordered pairs) ============
print(f"=== P5. Procrustes robustness ({time.time()-t0:.0f}s in) ===", flush=True)
RUN = {}
for m in MODELS4:
    data, meta = load(f"{m}_v2")
    Lm = min(meta["layers"], key=lambda x: abs(x - 0.5 * meta["n_layers"]))
    al = (data["answer_tok_len"] if "answer_tok_len" in data else data["answer_tok_len_ws"]).astype(float)
    mx = data["mean_maxsoftmax"].astype(float)
    RUN[m] = {"X": data[f"last_{Lm}"].astype(np.float64), "S": data["support"].astype(int),
              "F": data["factuality"].astype(int), "item": data["item_id"],
              "len_bin": (al > np.median(al)).astype(int),
              "ov_bin": data["overlap_measured"].astype(int),
              "conf_bin": (mx > np.median(mx)).astype(int)}


def split(item, seed=0):
    u = np.unique(item); r = np.random.default_rng(seed); r.shuffle(u)
    cal = set(u[: len(u) // 2].tolist())
    mask = np.array([i in cal for i in item])
    return mask, ~mask


keymap = {"support": "S", "factuality": "F", "length": "len_bin", "overlap": "ov_bin", "confidence": "conf_bin"}
pair_cos = {}
nuis = {k: [] for k in keymap}
for s, t in itertools.permutations(MODELS4, 2):
    Sd, Td = RUN[s], RUN[t]
    cal, ev = split(Sd["item"], 0)
    A = fit_map(Td["X"][cal], Sd["X"][cal], orthogonal=True)
    row = {}
    for axis, key in keymap.items():
        tr = transport_direction(A, fit_direction(Sd["X"][cal], Sd[key][cal]))
        dT = fit_direction(Td["X"][cal], Td[key][cal])
        c = abs(float(np.dot(_unit(tr), _unit(dT))))
        nuis[axis].append(c); row[axis] = c
    pair_cos[f"{s}->{t}"] = row["support"]
    print(f"  {s}->{t}: support_cos={row['support']:.3f} (len={row['length']:.3f} ov={row['overlap']:.3f} "
          f"conf={row['confidence']:.3f} fac={row['factuality']:.3f})  [{time.time()-t0:.0f}s]", flush=True)
sup = np.array(nuis["support"])
out["c3_robust"] = {"n_ordered_pairs": len(pair_cos),
                    "support_cos_mean": float(sup.mean()), "support_cos_std": float(sup.std()),
                    "support_cos_min": float(sup.min()), "support_cos_max": float(sup.max()),
                    "per_pair_support_cos": pair_cos,
                    "nuisance_cos_mean": {k: float(np.mean(v)) for k, v in nuis.items()}}
print("  support cos over 12 pairs: mean={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(
    sup.mean(), sup.std(), sup.min(), sup.max()), flush=True)
print("  transported-to-native cosine by axis: " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in nuis.items()), flush=True)
save()

# ============ P6d. per-task synth-d_S / confidence / in-domain (mean over 3 RT models) ============
print(f"=== P6d. per-task signals ({time.time()-t0:.0f}s in) ===", flush=True)
acc = {"synth_dS": {}, "confidence": {}, "in_domain": {}}
tasks_ref = None
for m in RT_MODELS:
    cp, cpm = load(f"{m}_v2")
    rt, rtm = load(f"{m}_rt")
    L = min(set(cpm["layers"]) & set(rtm["layers"]), key=lambda L: abs(L - 0.5 * cpm["n_layers"]))
    d_sup = fit_direction(cp[f"last_{L}"].astype(np.float64), cp["support"].astype(int))
    Xrt = rt[f"last_{L}"].astype(np.float64); faith = rt["faithful"].astype(int)
    tt = rt["task_type"]; item = rt["item_id"]; tasks_ref = sorted(set(tt.tolist()))
    s_synth = project(Xrt, d_sup); conf = rt["mean_maxsoftmax"].astype(np.float64)
    _, oof_in = gcv_logistic(Xrt, faith, item)
    for tk in tasks_ref:
        mk = tt == tk
        acc["synth_dS"].setdefault(tk, []).append(auroc(s_synth[mk], faith[mk]))
        acc["confidence"].setdefault(tk, []).append(auroc(conf[mk], faith[mk]))
        acc["in_domain"].setdefault(tk, []).append(auroc(oof_in[mk], faith[mk]))
    print(f"  {m} done [{time.time()-t0:.0f}s]", flush=True)
pt = {}
for sig in acc:
    pt[sig] = {tk: float(np.mean(acc[sig][tk])) for tk in tasks_ref}
    pt[sig]["macro"] = float(np.mean([pt[sig][tk] for tk in tasks_ref]))
out["c4_pertask_full"] = {"tasks": tasks_ref, "per_task_mean_over_models": pt, "models": RT_MODELS}
for sig in pt:
    print(f"  {sig:<11} " + " ".join(f"{tk}={pt[sig][tk]:.3f}" for tk in tasks_ref) + f"  macro={pt[sig]['macro']:.3f}", flush=True)
save()
print(f"\nwrote {OUTP} [{time.time()-t0:.0f}s total]", flush=True)
