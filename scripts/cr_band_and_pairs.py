"""Camera-ready: the four remaining statistics the submission promises or implies but never computes.

(A) MID-LAYER BAND.  sections.tex:271-273 promises "we additionally report the
    mean+-sd across the mid-layer band (0.4--0.6 depth) to show the result is not a
    single-layer artifact". No such statistic exists anywhere. Computed here in the
    Table-1 convention (collision-free cell C, within O=1, item-grouped CV, mass-mean),
    raw and confidence-purged, per family.

(B) PAIRED SIGNIFICANCE FOR THE C4 NEGATIVE.  Sec. 5.4 asserts synth-$d_S$ is "below
    NLI, confidence, and lexical overlap", but the only paired test in the repo is
    in-domain vs overlap (harden_analyses.py:78). This adds item-bootstrap paired
    differences and one-sided p-values for synth-$d_S$ against each baseline.

(C) PER-PAIR NUISANCE COSINES.  sections.tex:384 claims the confidence direction
    transports "at only 0.50 (and unstably, 0.04--0.90 across pairs)". Only the
    12-pair MEAN is stored (reviewer_analyses.json c3_robust.nuisance_cos_mean);
    the per-pair spread is unsupported. Recomputed and stored for every axis.

(D) ACS COSINE.  sections.tex:122/364 compare Procrustes to anchor projection in
    cosine terms ("well above anchor projection"), but no ACS transported-to-native
    cosine is computed anywhere. Added here so the claim can be backed or dropped.

Usage:  PYTHONPATH=src python scripts/cr_band_and_pairs.py
Writes: runs/cr_band_and_pairs.json
"""
from __future__ import annotations
import json, os, sys, time, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project, _unit   # noqa: E402
from groundlm.probe.confidence import purge                           # noqa: E402
from groundlm.transfer.procrustes import fit_map, transport_direction # noqa: E402
from _cr_common import (load, apriori_layer, band_layers,             # noqa: E402
                        auroc_signed as auroc,
                        seeded_group_folds, collision_map, stat,
                        source_groups, passage_groups)

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
RT_MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
NSEED = int(os.environ.get("NSEED", "10"))
NBOOT = int(os.environ.get("NBOOT", "2000"))
OUTP = os.environ.get("OUT", "runs/cr_band_and_pairs.json")

out = {"nseed": NSEED, "n_boot": NBOOT, "numpy": np.__version__,
       "band": {}, "c4_paired": {}, "c3_per_pair": {}, "acs": {}}
t0 = time.time()


def purge_fit(X_tr, C_tr):
    """Fit the residualiser on the TRAINING fold only (never on all rows)."""
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


def cv_massmean_seeded(X, y, groups, seed, C=None):
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(groups, 5, seed):
        if C is None:
            Xtr, Xte = X[tr], X[te]
        else:
            f = purge_fit(X[tr], C[tr])
            Xtr, Xte = purge_apply(X[tr], C[tr], f), purge_apply(X[te], C[te], f)
        oof[te] = project(Xte, fit_direction(Xtr, y[tr]))
    return auroc(oof, y)


# ============ (A) mid-layer band, Table-1 convention ============
print("=== (A) mid-layer band (0.4-0.6 depth), Table-1 convention ===", flush=True)
coll = collision_map()
for m in MODELS4:
    data, meta = load(f"{m}_v2")
    O = data["overlap_measured"].astype(int)
    cell = data["cell"]
    item = data["item_id"]
    is_coll = np.array([bool(coll.get(int(item[k]), False)) for k in range(len(O))])
    keep = ~((cell == "C") & is_coll)
    m1 = (O == 1) & keep
    S = data["support"].astype(int)[m1]
    g = source_groups(item)[m1]   # same grouping as Table 1
    confs = np.stack([data["mean_logprob"], data["mean_maxsoftmax"],
                      data["first_maxsoftmax"]], 1)[m1]      # the 3 scalars used for Table 1
    band = band_layers(meta)
    L0 = apriori_layer(meta)
    raws, purs = [], []
    for L in band:
        X = data.layer(L)[m1]
        r = float(np.mean([cv_massmean_seeded(X, S, g, s) for s in range(NSEED)]))
        p = float(np.mean([cv_massmean_seeded(X, S, g, s, C=confs) for s in range(NSEED)]))
        raws.append(r); purs.append(p)
        data.drop(f"last_{L}")
        print(f"  {m:15s} L{L:>2} (d={L/meta['n_layers']:.2f}) raw={r:.4f} purged={p:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    out["band"][m] = {"a_priori_layer": int(L0), "band_layers": [int(x) for x in band],
                      "raw": stat(raws), "purged": stat(purs),
                      "n_A": int((cell[m1] == "A").sum()), "n_C": int((cell[m1] == "C").sum())}
    b = out["band"][m]
    print(f"  -> {m}: raw {b['raw']['mean']:.3f}+-{b['raw']['sd']:.3f}  "
          f"purged {b['purged']['mean']:.3f}+-{b['purged']['sd']:.3f}  over {len(band)} layers", flush=True)
    json.dump(out, open(OUTP, "w"), indent=2, default=float)


# ============ (B) paired item-bootstrap: synth-d_S vs each baseline ============
print(f"=== (B) paired bootstrap on RAGTruth ({time.time()-t0:.0f}s) ===", flush=True)


def paired(a, b, y, groups, B=NBOOT, seed=2):
    y = np.asarray(y).astype(int)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    idx_of = {u: np.where(groups == u)[0] for u in uniq}
    diffs = []
    for _ in range(B):
        gs = rng.choice(uniq, len(uniq))
        idx = np.concatenate([idx_of[g] for g in gs])
        if len(np.unique(y[idx])) == 2:
            diffs.append(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
    d = np.asarray(diffs)
    return {"diff": float(d.mean()),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_synth_not_worse": float((d >= 0).mean())}   # one-sided: P(synth >= baseline)


for m in RT_MODELS:
    cp, cpm = load(f"{m}_v2")
    rt, rtm = load(f"{m}_rt")
    L = min(set(cpm["layers"]) & set(rtm["layers"]), key=lambda L: abs(L - 0.5 * cpm["n_layers"]))
    d_sup = fit_direction(cp.layer(L), cp["support"].astype(int))
    Xrt = rt.layer(L)
    faith = rt["faithful"].astype(int)
    item = passage_groups()   # cluster on source passage; item_id is a per-response counter
    synth = project(Xrt, d_sup)
    base = {"lexical_overlap": rt["lexical_overlap"].astype(np.float64),
            "confidence": rt["mean_maxsoftmax"].astype(np.float64)}
    if "nli_entail_prob" in rt:
        base["nli"] = rt["nli_entail_prob"].astype(np.float64)
    out["c4_paired"][m] = {"layer": int(L)}
    for k, b in base.items():
        r = paired(synth, b, faith, item)
        out["c4_paired"][m][f"synth_vs_{k}"] = r
        print(f"  {m:15s} synth-dS vs {k:16s} diff={r['diff']:+.4f} "
              f"CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p(synth>=base)={r['p_synth_not_worse']:.4f}", flush=True)
    json.dump(out, open(OUTP, "w"), indent=2, default=float)


# ============ (C)+(D) per-pair transported-to-native cosines, all axes + ACS ============
print(f"=== (C)(D) Procrustes per-pair cosines, all axes + ACS ({time.time()-t0:.0f}s) ===", flush=True)
RUN = {}
for m in MODELS4:
    data, meta = load(f"{m}_v2")
    L = apriori_layer(meta)
    X = data.layer(L)
    lenk = "answer_tok_len" if "answer_tok_len" in data else "answer_tok_len_ws"
    ln = data[lenk].astype(float)
    RUN[m] = {
        "X": X,
        "item": source_groups(data["item_id"]),   # source-QA grouping, as everywhere else
        "axes": {
            "support":    data["support"].astype(int),
            "factuality": data["factuality"].astype(int),
            "length":     (ln > np.median(ln)).astype(int),
            "overlap":    data["overlap_measured"].astype(int),
            "confidence": (data["mean_maxsoftmax"].astype(float)
                           > np.median(data["mean_maxsoftmax"].astype(float))).astype(int),
        },
    }
    print(f"  loaded {m} L{L} X={X.shape} [{time.time()-t0:.0f}s]", flush=True)


def split_items(item, seed=0):
    u = np.unique(item)
    r = np.random.default_rng(seed)
    r.shuffle(u)
    half = set(u[: len(u) // 2].tolist())
    cal = np.array([g in half for g in item])
    return np.where(cal)[0], np.where(~cal)[0]


per_pair = {k: [] for k in RUN[MODELS4[0]]["axes"]}
acs_cos = []
pairs = []
for s, t in itertools.permutations(MODELS4, 2):
    Sd, Td = RUN[s], RUN[t]
    rec = {"pair": f"{s}->{t}"}
    # Mean over the same ten seeded source-grouped splits used everywhere else.
    for k in per_pair:
        cs = []
        for seed in range(NSEED):
            cal, _ = split_items(Sd["item"], seed=seed)
            A = fit_map(Td["X"][cal], Sd["X"][cal], orthogonal=True)  # target -> source
            d_src = fit_direction(Sd["X"][cal], Sd["axes"][k][cal])
            d_tgt_native = fit_direction(Td["X"][cal], Td["axes"][k][cal])
            cs.append(float(abs(np.dot(_unit(transport_direction(A, d_src)), _unit(d_tgt_native)))))
        c = float(np.mean(cs))
        per_pair[k].append(c)
        rec[k] = c
    cal, _ = split_items(Sd["item"], seed=0)
    A = fit_map(Td["X"][cal], Sd["X"][cal], orthogonal=True)
    # --- ACS / anchor projection: represent each row by similarity to shared anchors ---
    rng = np.random.default_rng(7)
    n_anchor = 256
    anchors = rng.choice(len(cal), size=min(n_anchor, len(cal)), replace=False)
    def acs_embed(D):
        Aset = D["X"][cal][anchors]
        Aset = Aset / (np.linalg.norm(Aset, axis=1, keepdims=True) + 1e-9)
        Z = D["X"][cal] / (np.linalg.norm(D["X"][cal], axis=1, keepdims=True) + 1e-9)
        return Z @ Aset.T
    Zs, Zt = acs_embed(Sd), acs_embed(Td)
    d_src_acs = fit_direction(Zs, Sd["axes"]["support"][cal])
    d_tgt_acs = fit_direction(Zt, Td["axes"]["support"][cal])
    acs_c = float(abs(np.dot(_unit(d_src_acs), _unit(d_tgt_acs))))   # shared anchor space: directly comparable
    acs_cos.append(acs_c)
    rec["acs_support"] = acs_c
    pairs.append(rec)
    print(f"  {s}->{t}: " + " ".join(f"{k}={rec[k]:.3f}" for k in per_pair) +
          f" | ACS={acs_c:.3f} [{time.time()-t0:.0f}s]", flush=True)

out["c3_per_pair"] = {"pairs": pairs,
                      "summary": {k: stat(v) for k, v in per_pair.items()}}
out["acs"] = {"support_cosine_in_anchor_space": stat(acs_cos),
              "n_anchors": int(min(256, len(cal))),
              "note": "ACS has no target-space direction to compare against, so the cosine is "
                      "measured between the two families' support directions in the SHARED "
                      "anchor-similarity space; it is not the same quantity as the Procrustes "
                      "transported-to-native cosine and should be labelled as such."}
for k, v in out["c3_per_pair"]["summary"].items():
    print(f"  {k:12s} mean={v['mean']:.4f} sd={v['sd']:.4f} range[{v['min']:.4f},{v['max']:.4f}]")
print(f"  ACS(support, anchor space) mean={out['acs']['support_cosine_in_anchor_space']['mean']:.4f} "
      f"range[{out['acs']['support_cosine_in_anchor_space']['min']:.4f},"
      f"{out['acs']['support_cosine_in_anchor_space']['max']:.4f}]")

json.dump(out, open(OUTP, "w"), indent=2, default=float)
print(f"wrote {OUTP} [{time.time()-t0:.0f}s]")
