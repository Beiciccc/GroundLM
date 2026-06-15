"""Corrected C2 + the lexical-overlap control, on the pilot features (local, fast).

The first sweep folded lexical_overlap into the confound set; because overlap ~ the
grounding label by construction (corr 0.99), that absorbed the faithfulness axis for
the wrong reason. Here we separate:
  * CONFIDENCE-only absorption (max-softmax, first-softmax, log-prob, activation norm)
    -> the real C2 asymmetry test;
  * SURFACE absorption (overlap, length) -> how lexical the faithfulness axis is;
  * faithfulness AUROC raw vs after purging surface -> does a faithfulness signal
    survive BEYOND lexical overlap (the key reviewer threat)?
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.analyze import build_confounds                       # noqa: E402
from groundlm.probe.directions import fit_direction, cv_auroc      # noqa: E402
from groundlm.probe.confidence import _proj_r2, purge, asymmetry_test  # noqa: E402
from groundlm.probe.directions import project                      # noqa: E402


def main():
    d = "runs/qwen25_7b"
    data = dict(np.load(os.path.join(d, "features.npz"), allow_pickle=True))
    meta = json.load(open(os.path.join(d, "features.meta.json")))
    layers = [L for L in meta["layers"] if L in (9, 12, 15, 18, 21)]
    fy = data["grounding"].astype(int)
    gy = data["factuality"].astype(int)

    print(f"{'L':>3} | conf-absorb faith/fact  asym(p) | surf-absorb faith | "
          f"faith AUROC raw/-overlap | fact AUROC")
    rows = []
    for L in layers:
        X = data[f"last_{L}"].astype(np.float64)
        conf = build_confounds(data, X, kind="confidence")
        surf = build_confounds(data, X, kind="surface")

        d_faith = fit_direction(X, fy); d_fact = fit_direction(X, gy)
        af_c = _proj_r2(project(X, d_faith), conf)
        ag_c = _proj_r2(project(X, d_fact), conf)
        af_s = _proj_r2(project(X, d_faith), surf)

        asym = asymmetry_test(X, fy, gy, conf, n_boot=120, seed=0)  # confidence-only
        fa_raw = cv_auroc(X, fy)
        fa_noov = cv_auroc(purge(X, surf), fy)      # faithfulness signal beyond surface
        ga_raw = cv_auroc(X, gy)

        print(f"{L:>3} | {af_c:.2f}/{ag_c:.2f}  {asym.asymmetry:+.2f}(p={asym.p_value:.2f}) | "
              f"{af_s:.2f} | {fa_raw:.3f}/{fa_noov:.3f} | {ga_raw:.3f}")
        rows.append({"layer": L, "absorb_faith_conf": af_c, "absorb_fact_conf": ag_c,
                     "conf_asym": asym.asymmetry, "conf_asym_p": asym.p_value,
                     "absorb_faith_surface": af_s, "faith_auroc_raw": fa_raw,
                     "faith_auroc_no_surface": fa_noov, "fact_auroc_raw": ga_raw})

    json.dump(rows, open(os.path.join(d, "reanalyze_c2.json"), "w"), indent=2, default=float)
    # headline reads
    best = max(rows, key=lambda r: r["faith_auroc_no_surface"])
    print(f"\nConfidence-only C2 asymmetry (fact-faith): "
          f"{'POSITIVE (supports hypothesis)' if np.mean([r['conf_asym'] for r in rows])>0 else 'NOT positive'} "
          f"(mean {np.mean([r['conf_asym'] for r in rows]):+.2f})")
    print(f"Faithfulness beyond lexical overlap: best layer L{best['layer']} "
          f"AUROC raw {best['faith_auroc_raw']:.3f} -> after purging surface {best['faith_auroc_no_surface']:.3f}")


if __name__ == "__main__":
    main()
