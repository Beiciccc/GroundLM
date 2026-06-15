"""C4 on REAL hallucinations: learn the SUPPORT axis from CtrlPairs, apply it to
RAGTruth responses (same model), and detect hallucination vs NLI / max-softmax.

This is the proper C4 test. We transfer the grounding axis synthetic->real (same
model, cross-dataset) and ask: does a one-pass internal projection detect real
RAGTruth hallucinations as well as / better than an entailment model?
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.analyze import build_confounds                          # noqa: E402
from groundlm.probe.directions import fit_direction, project, _unit   # noqa: E402
from groundlm.probe.confidence import purge                           # noqa: E402
from groundlm.gate import aurc                                        # noqa: E402


def _load(d):
    data = dict(np.load(os.path.join(d, "features.npz"), allow_pickle=True))
    meta = json.load(open(os.path.join(d, "features.meta.json")))
    return data, meta


def _auroc(score, y):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, score)
    return float(max(a, 1.0 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp-dir", required=True, help="CtrlPairs features dir (learn the axis)")
    ap.add_argument("--rt-dir", required=True, help="RAGTruth features dir (apply the axis)")
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--depth", type=float, default=0.5)
    args = ap.parse_args()

    cp, cpm = _load(args.cp_dir)
    rt, rtm = _load(args.rt_dir)
    L = min(set(cpm["layers"]) & set(rtm["layers"]),
            key=lambda L: abs(L - args.depth * cpm["n_layers"]))
    print(f"model={cpm['model_id']} layer L{L}")

    Xcp = cp[f"{args.pooling}_{L}"].astype(np.float64)
    S = cp["support"].astype(int); F = cp["factuality"].astype(int)
    conf_cp = build_confounds(cp, Xcp, "confidence")
    d_sup_raw = fit_direction(Xcp, S)
    d_sup_pur = fit_direction(purge(Xcp, conf_cp), S)
    d_fac = fit_direction(Xcp, F)

    Xrt = rt[f"{args.pooling}_{L}"].astype(np.float64)
    faith = rt["faithful"].astype(int)
    conf_rt = build_confounds(rt, Xrt, "confidence")
    s_sup_raw = project(Xrt, d_sup_raw)
    s_sup_pur = project(purge(Xrt, conf_rt), d_sup_pur)
    s_fac = project(Xrt, d_fac)

    scores = {
        "support_axis_raw": s_sup_raw,
        "support_axis_purged": s_sup_pur,
        "two_axis": (s_sup_raw - s_sup_raw.mean()) / (s_sup_raw.std() + 1e-8)
                    + 0.25 * (s_fac - s_fac.mean()) / (s_fac.std() + 1e-8),
        "max_softmax": rt["mean_maxsoftmax"].astype(np.float64),
        "lexical_overlap": rt["lexical_overlap"].astype(np.float64),
    }
    if "nli_entail_prob" in rt and np.nanmax(rt["nli_entail_prob"]) > 0:
        scores["nli"] = rt["nli_entail_prob"].astype(np.float64)

    print(f"\nRAGTruth n={len(faith)} faithful_rate={faith.mean():.2f}  "
          f"(detect hallucination; AUROC higher=better, AURC lower=better)")
    out = {"model": cpm["model_id"], "layer": int(L), "n": int(len(faith)),
           "faithful_rate": float(faith.mean()), "metrics": {}}
    for name, sc in scores.items():
        au = _auroc(sc, faith)
        ar = aurc(sc, faith)               # treat score as "safe to trust" ordering
        out["metrics"][name] = {"auroc": au, "aurc": ar}
        print(f"  {name:<22} AUROC={au:.3f}  AURC={ar:.3f}")

    # per-task AUROC for the best internal score
    if "task_type" in rt:
        tt = rt["task_type"]
        print("\n  per-task AUROC (support_axis_raw vs nli):")
        for t in sorted(set(tt.tolist())):
            m = tt == t
            if m.sum() > 30 and len(np.unique(faith[m])) == 2:
                a_s = _auroc(s_sup_raw[m], faith[m])
                a_n = _auroc(scores["nli"][m], faith[m]) if "nli" in scores else float("nan")
                print(f"    {t:<10} n={int(m.sum()):4d}  support={a_s:.3f}  nli={a_n:.3f}")
                out.setdefault("per_task", {})[t] = {"support_auroc": a_s, "nli_auroc": a_n, "n": int(m.sum())}

    best_int = max(("support_axis_raw", "support_axis_purged", "two_axis"),
                   key=lambda k: out["metrics"][k]["auroc"])
    nli_au = out["metrics"].get("nli", {}).get("auroc", float("nan"))
    print(f"\n  C4 verdict: best internal = {best_int} AUROC={out['metrics'][best_int]['auroc']:.3f} "
          f"vs NLI {nli_au:.3f} vs max-softmax {out['metrics']['max_softmax']['auroc']:.3f}")
    json.dump(out, open(os.path.join(args.rt_dir, "report_gate_ragtruth.json"), "w"), indent=2, default=float)
    print(f"  wrote {os.path.join(args.rt_dir, 'report_gate_ragtruth.json')}")


if __name__ == "__main__":
    main()
