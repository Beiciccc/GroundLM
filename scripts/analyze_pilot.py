"""Run the C1/C2 layer sweep locally on downloaded pilot features (free CPU)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/qwen25_7b")
    ap.add_argument("--pooling", default="last")
    args = ap.parse_args()

    data = dict(np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True))
    meta = json.load(open(os.path.join(args.dir, "features.meta.json")))
    data["layers"] = meta["layers"]
    pb = dict(np.load(os.path.join(args.dir, "parametric_belief.npz"), allow_pickle=True))
    knows = {int(i): bool(k) for i, k in zip(pb["item_id"], pb["knows_gold"])}
    kg_frac = float(np.mean(pb["knows_gold"]))
    mask = np.array([knows.get(int(i), False) for i in data["item_id"]])

    print(f"model={meta['model_id']}  n={len(data['grounding'])}  layers={meta['layers']}")
    print(f"parametric knows-gold fraction: {kg_frac*100:.1f}%\n")

    def sweep(tag, m):
        print(f"===== {tag} =====")
        t0 = time.time()
        res = analyze.run_layer_sweep(data, pooling=args.pooling, knows_gold_mask=m)
        for r in res:
            print(f"  L{r['layer']:>2} | angle {r['cross_angle']:5.1f} vs null "
                  f"{r['within_faith_null_mean']:4.1f}/{r['within_fact_null_mean']:4.1f} "
                  f"| sep p={r['separable_p']:.3f}{'*' if r['separable'] else ' '} "
                  f"| shared-removed {r['cross_angle_shared_removed']:4.1f} "
                  f"| absorbed faith={r['absorbed_faith']:.2f} fact={r['absorbed_fact']:.2f} "
                  f"asym={r['confidence_asymmetry']:+.2f}(p={r['asymmetry_p']:.3f})")
        best = analyze.best_layer(res)
        print(f"  -> best L{best['layer']}: separable={best['separable']} angle={best['cross_angle']:.1f} "
              f"asym={best['confidence_asymmetry']:+.2f} supported={best['asym_supported']}  [{time.time()-t0:.0f}s]\n")
        return res, best

    res_all, best_all = sweep("ALL ITEMS", None)
    res_kg, best_kg = sweep("KNOWS-GOLD ONLY (parametric label matches model memory)", mask)

    report = {"model": meta["model_id"], "n": int(len(data["grounding"])),
              "knows_gold_frac": kg_frac, "layers": meta["layers"],
              "all_items": {"results": res_all, "best": best_all},
              "knows_gold_only": {"results": res_kg, "best": best_kg}}
    with open(os.path.join(args.dir, "report_local.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"wrote {os.path.join(args.dir, 'report_local.json')}")


if __name__ == "__main__":
    main()
