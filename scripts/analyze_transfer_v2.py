"""C3 — cross-family transfer of the (confidence-purged) SUPPORT direction on v2.

All models were extracted on the SAME ctrlpairs_v2.jsonl, so rows correspond. We
pick each model's layer at a common RELATIVE depth (families differ in #layers),
transport a source-trained probe to each target via orthogonal Procrustes / ACS,
and report target AUROC. The C3 claim: the semantic grounding (support) direction
transfers cross-family; we contrast purged vs raw and support vs factuality, with
within-target (diagonal) as the upper bound and a random direction as the floor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.analyze import build_confounds                       # noqa: E402
from groundlm.transfer import transfer_auroc                       # noqa: E402


def load_run(d, pooling, depth):
    data = dict(np.load(os.path.join(d, "features.npz"), allow_pickle=True))
    meta = json.load(open(os.path.join(d, "features.meta.json")))
    layers, nL = meta["layers"], meta["n_layers"]
    L = min(layers, key=lambda L: abs(L - depth * nL))   # layer nearest target relative depth
    X = data[f"{pooling}_{L}"].astype(np.float64)
    return {"name": os.path.basename(d.rstrip("/")).replace("_v2", ""),
            "X": X, "support": data["support"].astype(int),
            "factuality": data["factuality"].astype(int),
            "conf": build_confounds(data, X, "confidence"),
            "n": len(X), "layer": L, "n_layers": nL,
            "item_id": data["item_id"], "cell": data.get("cell")}


def transfer_table(runs, axis, modes, seed=0):
    names = [r["name"] for r in runs]
    M = {s["name"]: {} for s in runs}
    for s in runs:
        for t in runs:
            mode_aurocs = {}
            for mode in modes:
                m = "within_target" if s["name"] == t["name"] else mode
                mode_aurocs[mode] = transfer_auroc(
                    s["X"], s[axis], s["conf"], t["X"], t[axis], t["conf"],
                    mode=m, seed=seed)["auroc"]
            M[s["name"]][t["name"]] = mode_aurocs
    return names, M


def print_matrix(names, M, mode):
    print(f"\n  [{mode}] rows=source, cols=target")
    print("    " + "".join(f"{n[:10]:>12}" for n in names))
    for s in names:
        print(f"  {s[:10]:>10} " + "".join(
            f"{M[s][t][mode]:>12.3f}" if mode in M[s][t] else f"{M[s][t].get('within_target', float('nan')):>12.3f}"
            for t in names))


def offdiag_mean(names, M, mode):
    vals = [M[s][t].get(mode, M[s][t].get("within_target")) for s in names for t in names if s != t]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="runs/*_v2 dirs (>=2 models)")
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--depth", type=float, default=0.5, help="target relative layer depth")
    args = ap.parse_args()

    runs = [load_run(d, args.pooling, args.depth) for d in args.dirs]
    n0 = runs[0]["n"]
    for r in runs:
        assert r["n"] == n0, f"row mismatch {r['name']}: {r['n']} vs {n0} — models must share ctrlpairs"
        if r["cell"] is not None and runs[0]["cell"] is not None:
            assert np.array_equal(r["cell"], runs[0]["cell"]), f"cell order mismatch for {r['name']}"
    print("models:", ", ".join(f"{r['name']}(L{r['layer']}/{r['n_layers']})" for r in runs))

    modes = ["procrustes_purged", "procrustes_raw", "acs_purged", "random"]
    out = {"depth": args.depth, "pooling": args.pooling, "support": {}, "factuality": {}}
    for axis in ("support", "factuality"):
        print(f"\n===== transfer of {axis.upper()} direction =====")
        names, M = transfer_table(runs, axis, modes)
        for mode in modes:
            print_matrix(names, M, mode)
        out[axis] = {mode: offdiag_mean(names, M, mode) for mode in modes}
        out[axis]["within_target"] = float(np.mean([M[n][n]["procrustes_purged"] for n in names]))
        print(f"  within-target (upper bound) mean AUROC = {out[axis]['within_target']:.3f}")
        print(f"  off-diagonal mean AUROC: " +
              " ".join(f"{m}={out[axis][m]:.3f}" for m in modes))

    print("\n===== C3 SUMMARY =====")
    print(f"  SUPPORT  cross-family (procrustes_purged) off-diag = {out['support']['procrustes_purged']:.3f} "
          f"(raw {out['support']['procrustes_raw']:.3f}, acs {out['support']['acs_purged']:.3f}, "
          f"random {out['support']['random']:.3f})")
    print(f"  FACTUALITY cross-family (procrustes_purged) off-diag = {out['factuality']['procrustes_purged']:.3f}")
    json.dump(out, open("runs/transfer_v2.json", "w"), indent=2, default=float)
    print("  wrote runs/transfer_v2.json")


if __name__ == "__main__":
    main()
