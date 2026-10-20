"""Camera-ready fix: RAGTruth AUROCs WITHOUT the max(AUC, 1-AUC) test-fold flip.

The submission states three times that the RAGTruth numbers are scored without the
flip -- sections.tex:438-439 ("we keep the support-polarity sign fixed from CtrlPairs
and score without the test-fold flip, because flipping per-test inflates a near-chance
axis"), the tab:c4 caption at sections.tex:479, and appendix.tex:74-78 ("the test-fold
max(AUC,1-AUC) convention ... inflates a true-chance random direction to ~0.54-0.59,
which matters for the *weak* numbers (the C3 map-aware nulls and the synthetic axis on
RAGTruth), so we report those without the flip").

The code that produced those numbers applies the flip anyway:
  scripts/analyze_gate_ragtruth.py:32-37  _auroc(...) -> float(max(a, 1.0 - a))
  scripts/reviewer_analyses.py:44-49      auroc(...)  -> float(max(a, 1.0 - a))

This script recomputes every RAGTruth signal with plain roc_auc_score, polarity fixed
a priori (support=1 direction learned on CtrlPairs; higher score = predicted faithful),
so Table 3 / Table 5 / Fig. 4 can be stated as the text already claims. Values below
0.5 are reported as such: for a direction transferred with a fixed sign that is the
honest outcome and it strengthens the paper's negative.

Usage:  PYTHONPATH=src python scripts/cr_signed_auroc_ragtruth.py
Writes: runs/cr_signed_auroc_ragtruth.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import fit_direction, project   # noqa: E402
from groundlm.probe.confidence import purge                    # noqa: E402
from groundlm.analyze import build_confounds                   # noqa: E402
from _cr_common import load, auroc_signed as signed                # noqa: E402

RT_MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b"]
OUTP = os.environ.get("OUT", "runs/cr_signed_auroc_ragtruth.json")
NBOOT = int(os.environ.get("NBOOT", "2000"))
def flipped(score, y):
    """The convention the code actually used, for side-by-side comparison."""
    a = signed(score, y)
    return float(max(a, 1.0 - a)) if a == a else a


def boot_ci(score, y, seed=1, B=None):
    B = B or NBOOT
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            vals.append(signed(score[idx], y[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


out = {"numpy": np.__version__, "n_boot": NBOOT, "models": RT_MODELS, "per_model": {}}
t0 = time.time()

for m in RT_MODELS:
    cp, cpm = load(f"{m}_v2")
    rt, rtm = load(f"{m}_rt")
    L = min(set(cpm["layers"]) & set(rtm["layers"]), key=lambda L: abs(L - 0.5 * cpm["n_layers"]))

    Xcp = cp.layer(L)
    Scp = cp["support"].astype(int)
    d_sup = fit_direction(Xcp, Scp)                       # polarity: support=1 side is positive

    Xrt = rt.layer(L)
    faith = rt["faithful"].astype(int)
    tt = rt["task_type"]
    s_raw = project(Xrt, d_sup)

    # confidence-purged variant of the same transferred axis
    conf_cp = build_confounds(cp, Xcp, "confidence")
    d_sup_pur = fit_direction(purge(Xcp, conf_cp), Scp)
    conf_rt = build_confounds(rt, Xrt, "confidence")
    s_pur = project(purge(Xrt, conf_rt), d_sup_pur)

    sigs = {
        "synth_dS_raw": s_raw,
        "synth_dS_purged": s_pur,
        "confidence_maxsoftmax": rt["mean_maxsoftmax"].astype(np.float64),
        "lexical_overlap": rt["lexical_overlap"].astype(np.float64),
    }
    if "nli_entail_prob" in rt:
        sigs["nli_entail"] = rt["nli_entail_prob"].astype(np.float64)

    rec = {"layer": int(L), "n": int(len(faith)), "faithful_rate": float(faith.mean()), "pooled": {}, "per_task": {}}
    for k, s in sigs.items():
        rec["pooled"][k] = {"signed": signed(s, faith), "flipped_as_published": flipped(s, faith),
                            "signed_ci": boot_ci(s, faith)}
    for tk in sorted(set(tt.tolist())):
        mk = tt == tk
        rec["per_task"][tk] = {k: {"signed": signed(s[mk], faith[mk]),
                                   "flipped_as_published": flipped(s[mk], faith[mk]),
                                   "n": int(mk.sum())} for k, s in sigs.items()}
    out["per_model"][m] = rec

    print(f"=== {m} (L{L}, n={rec['n']}, faithful={rec['faithful_rate']:.3f}) [{time.time()-t0:.0f}s]", flush=True)
    for k in sigs:
        p = rec["pooled"][k]
        print(f"    {k:24s} signed={p['signed']:.4f} CI[{p['signed_ci'][0]:.3f},{p['signed_ci'][1]:.3f}]"
              f"   as-published(flipped)={p['flipped_as_published']:.4f}"
              f"{'   <-- FLIP CHANGED THE NUMBER' if abs(p['signed']-p['flipped_as_published'])>1e-9 else ''}", flush=True)
    json.dump(out, open(OUTP, "w"), indent=2, default=float)

# macro summaries
tasks = sorted(set(out["per_model"][RT_MODELS[0]]["per_task"].keys()))
sig_names = list(out["per_model"][RT_MODELS[0]]["pooled"].keys())
out["summary"] = {}
for k in sig_names:
    pooled_signed = [out["per_model"][m]["pooled"][k]["signed"] for m in RT_MODELS]
    pooled_flip = [out["per_model"][m]["pooled"][k]["flipped_as_published"] for m in RT_MODELS]
    macro_signed = [float(np.mean([out["per_model"][m]["per_task"][t][k]["signed"] for t in tasks]))
                    for m in RT_MODELS]
    out["summary"][k] = {
        "pooled_signed_per_model": pooled_signed,
        "pooled_flipped_per_model": pooled_flip,
        "pooled_signed_mean": float(np.mean(pooled_signed)),
        "pooled_flipped_mean": float(np.mean(pooled_flip)),
        "within_task_macro_signed_per_model": macro_signed,
        "within_task_macro_signed_mean": float(np.mean(macro_signed)),
    }
    s = out["summary"][k]
    print(f"  {k:24s} pooled signed {['%.3f' % v for v in pooled_signed]} (mean {s['pooled_signed_mean']:.3f}) "
          f"| as-published {['%.3f' % v for v in pooled_flip]} (mean {s['pooled_flipped_mean']:.3f}) "
          f"| macro signed {s['within_task_macro_signed_mean']:.3f}")

json.dump(out, open(OUTP, "w"), indent=2, default=float)
print(f"wrote {OUTP} [{time.time()-t0:.0f}s]")
