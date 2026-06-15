"""v2 analysis: is there a SEMANTIC grounding axis beyond lexical overlap?

Headline metric (overlap-stratified): among items where the answer IS present
(O=1: cells A supported vs C unsupported-wrong-question), can we decode SUPPORT
from the residual stream? Overlap is constant in this stratum, so any AUROC>0.5 is
grounding signal that is NOT lexical overlap. Compare to v1, where this was
undefined (all O=1 items were supported). Also: support survives surface-purge,
support-vs-factuality separability, and the confidence asymmetry (confidence-only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.analyze import build_confounds                                    # noqa: E402
from groundlm.probe.directions import cv_auroc                                  # noqa: E402
from groundlm.probe.confidence import purge, asymmetry_test                     # noqa: E402
from groundlm.probe.identification import separability_test                     # noqa: E402


def _auroc_within(X, y, mask):
    yy = y[mask]
    if len(np.unique(yy)) < 2 or mask.sum() < 30:
        return float("nan")
    return cv_auroc(X[mask], yy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/qwen25_7b_v2")
    ap.add_argument("--pooling", default="last")
    args = ap.parse_args()

    data = dict(np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True))
    meta = json.load(open(os.path.join(args.dir, "features.meta.json")))
    layers = meta["layers"]
    S = data["support"].astype(int)
    F = data["factuality"].astype(int)
    O = data["overlap_measured"].astype(int)
    print(f"model={meta['model_id']} n={len(S)} layers={layers} "
          f"| O=1 frac={O.mean():.2f} support frac={S.mean():.2f}")

    rows = []
    for L in layers:
        X = data[f"{args.pooling}_{L}"].astype(np.float64)
        surf = build_confounds(data, X, "surface")
        conf = build_confounds(data, X, "confidence")
        r = {
            "layer": L,
            "support_auroc_all": cv_auroc(X, S),
            "support_auroc_O1": _auroc_within(X, S, O == 1),   # HEADLINE: grounding beyond overlap
            "support_auroc_O0": _auroc_within(X, S, O == 0),
            "support_auroc_purged_surface": cv_auroc(purge(X, surf), S),
            "fact_auroc": cv_auroc(X, F),
        }
        sep = separability_test(X, S, F, n_null=150, n_boot=150)
        r["support_vs_fact_angle"] = sep.cross_angle
        r["separable_p"] = sep.p_value
        asym = asymmetry_test(X, S, F, conf, n_boot=100)
        r["conf_asym_fact_minus_support"] = asym.asymmetry
        r["conf_asym_p"] = asym.p_value
        rows.append(r)
        print(f"  L{L:>2} | supp AUROC all={r['support_auroc_all']:.3f} "
              f"O1={r['support_auroc_O1']:.3f} O0={r['support_auroc_O0']:.3f} "
              f"-surf={r['support_auroc_purged_surface']:.3f} | fact={r['fact_auroc']:.3f} "
              f"| angle={r['support_vs_fact_angle']:.0f}(p={r['separable_p']:.2f}) "
              f"| confasym={r['conf_asym_fact_minus_support']:+.2f}")

    best = max(rows, key=lambda r: (r["support_auroc_O1"] if not np.isnan(r["support_auroc_O1"]) else 0))
    verdict = ("SEMANTIC GROUNDING AXIS EXISTS beyond overlap"
               if best["support_auroc_O1"] > 0.65 else
               "grounding largely reduces to overlap (diagnostic-paper territory)")
    print(f"\nbest O=1 support AUROC (grounding beyond overlap): "
          f"L{best['layer']} = {best['support_auroc_O1']:.3f}  ->  {verdict}")
    json.dump({"rows": rows, "best": best, "verdict": verdict},
              open(os.path.join(args.dir, "report_v2.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
