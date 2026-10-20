"""End-to-end pilot for ONE model: build/load CtrlPairs -> extract activations ->
score parametric belief -> C1/C2 layer sweep -> save features + JSON report.

Run on the A100 node, e.g.:
    python scripts/run_pilot.py --model meta-llama/Llama-3.1-8B-Instruct \
        --source auto --max-items 1500 --layers mid --out-dir runs/llama31_8b

Gives the early go/no-go signal: does the matched-pair cross-notion angle clear
the within-notion null, and does the confidence asymmetry appear?
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.data import build_ctrlpairs, load_source, save_statements, audit, format_audit  # noqa: E402
from groundlm.extract import extract, score_parametric_belief, save_features  # noqa: E402
from groundlm import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="auto", help="NQ-Swap .jsonl path or HF id")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-items", type=int, default=1500)
    ap.add_argument("--layers", default="mid", help="all | mid | comma list")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ctrlpairs", default="data/ctrlpairs.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--knows-gold-only", action="store_true",
                    help="restrict C2/C3 to items the model parametrically knows")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. CtrlPairs (load cached jsonl, else build from source and cache)
    if os.path.exists(args.ctrlpairs):
        stmts = [json.loads(l) for l in open(args.ctrlpairs) if l.strip()]
    else:
        raw = load_source(args.source, args.split)
        built = build_ctrlpairs(raw, max_items=args.max_items)
        save_statements(built, args.ctrlpairs)
        from dataclasses import asdict
        stmts = [asdict(s) for s in built]
    rep = audit(stmts)
    print(format_audit(rep))
    if not rep["PASS"]:
        print("WARNING: CtrlPairs audit did not PASS — inspect surface balance before trusting results.")

    # 2. activations
    print(f"\nextracting activations: {args.model} ({len(stmts)} statements)")
    feats = extract(args.model, stmts, layers=args.layers, pooling=(args.pooling,),
                    batch_size=args.batch_size, device=args.device)
    save_features(feats, os.path.join(args.out_dir, "features"))

    # 3. parametric belief
    print("scoring parametric belief (closed-book a_true vs a_swap)")
    pb = score_parametric_belief(args.model, stmts, device=args.device)
    np.savez(os.path.join(args.out_dir, "parametric_belief.npz"), **pb)
    knows = {int(i): bool(k) for i, k in zip(pb["item_id"], pb["knows_gold"])}
    print(f"  model parametrically knows gold on {np.mean(pb['knows_gold'])*100:.1f}% of items")

    mask = None
    if args.knows_gold_only:
        mask = np.array([knows.get(int(i), False) for i in feats["item_id"]])

    # 4. C1/C2 sweep
    print("\nC1/C2 layer sweep:")
    results = analyze.run_layer_sweep(feats, pooling=args.pooling, knows_gold_mask=mask)
    for r in results:
        print(f"  L{r['layer']:>2} | angle {r['cross_angle']:5.1f} vs null "
              f"{r['within_faith_null_mean']:4.1f}/{r['within_fact_null_mean']:4.1f} "
              f"| sep p={r['separable_p']:.3f} {'*' if r['separable'] else ' '} "
              f"| absorbed faith={r['absorbed_faith']:.2f} fact={r['absorbed_fact']:.2f} "
              f"asym={r['confidence_asymmetry']:+.2f} (p={r['asymmetry_p']:.3f})")
    best = analyze.best_layer(results)
    print(f"\nbest layer L{best['layer']}: separable={best['separable']} "
          f"angle={best['cross_angle']:.1f} asym={best['confidence_asymmetry']:+.2f} "
          f"asym_supported={best['asym_supported']}")

    report = {"model": args.model, "audit": rep, "knows_gold_frac": float(np.mean(pb["knows_gold"])),
              "knows_gold_only": args.knows_gold_only, "layer_results": results, "best_layer": best}
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(args.out_dir, 'report.json')}")


if __name__ == "__main__":
    main()
