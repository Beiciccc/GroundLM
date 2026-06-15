"""Extract features + parametric belief for ONE model (model loaded once, batched).
Generic over CtrlPairs v1/v2 (metadata auto-collected)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.extract import load_model, extract, score_parametric_belief, save_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctrlpairs", default="data/ctrlpairs_v2.jsonl")
    ap.add_argument("--belief-items", default="data/belief_items.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stmts = [json.loads(l) for l in open(args.ctrlpairs) if l.strip()]
    print(f"{args.model}: {len(stmts)} statements")
    model, tok = load_model(args.model, device=args.device)   # load ONCE, reuse below

    feats = extract(args.model, stmts, layers=args.layers, pooling=(args.pooling,),
                    batch_size=args.batch_size, max_len=args.max_len,
                    device=args.device, model=model, tok=tok)
    save_features(feats, os.path.join(args.out_dir, "features"))
    L0 = feats["layers"][0]
    print(f"  extracted features: {args.pooling}_{L0} shape {feats[f'{args.pooling}_{L0}'].shape}, layers={feats['layers']}")

    if args.belief_items and os.path.exists(args.belief_items):
        items = [json.loads(l) for l in open(args.belief_items) if l.strip()]
        pb = score_parametric_belief(args.model, items, model=model, tok=tok)
        np.savez(os.path.join(args.out_dir, "parametric_belief.npz"), **pb)
        print(f"  parametric knows-gold: {np.mean(pb['knows_gold'])*100:.1f}% of {len(items)} items")
    print(f"  saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
