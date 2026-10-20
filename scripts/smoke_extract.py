"""Tiny real-model smoke test of the extraction path (CPU, no GPU rental).

Validates that activations.extract + score_parametric_belief + analyze.run_layer
run end-to-end against a REAL Hugging Face model and produce correctly-shaped
outputs — catching tokenization / left-padding / hidden-state-shape / answer-span
bugs for free before paying for A100 time. Signal is meaningless on a tiny model;
we only assert the plumbing.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.data import build_ctrlpairs, audit, format_audit  # noqa: E402
from groundlm.extract import extract, score_parametric_belief    # noqa: E402
from groundlm import analyze                                      # noqa: E402
from test_ctrlpairs import fake_records                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2",
                    help="tiny real model; try Qwen/Qwen2.5-0.5B-Instruct for a target-family tokenizer")
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    stmts = [s.__dict__ for s in build_ctrlpairs(fake_records(args.items))]
    print(format_audit(audit(stmts)))

    print(f"\n[1] extract activations with {args.model} on {args.device}")
    feats = extract(args.model, stmts, layers="all", pooling=("last", "mean"),
                    batch_size=4, device=args.device, dtype="float32")
    n = len(stmts)
    keep = feats["layers"]
    print(f"    layers kept: {keep}; n_layers={feats['n_layers']}")
    for L in keep:
        assert feats[f"last_{L}"].shape[0] == n, feats[f"last_{L}"].shape
        assert feats[f"mean_{L}"].shape[0] == n
        H = feats[f"last_{L}"].shape[1]
    print(f"    feature shapes OK: last/mean per layer = (n={n}, H={H}), dtype={feats[f'last_{keep[0]}'].dtype}")
    for k in ("mean_maxsoftmax", "mean_logprob", "first_maxsoftmax",
              "grounding", "factuality", "item_id"):
        assert len(feats[k]) == n, (k, len(feats[k]))
    assert np.all((feats["mean_maxsoftmax"] >= 0) & (feats["mean_maxsoftmax"] <= 1.0001)), "softmax out of range"
    assert np.all(feats["mean_logprob"] <= 0.0001), "log-probs should be <= 0"
    print(f"    confidence ranges OK: maxsoftmax in [{feats['mean_maxsoftmax'].min():.3f},"
          f"{feats['mean_maxsoftmax'].max():.3f}], logprob<=0")

    print("\n[2] parametric-belief scoring (closed-book a_true vs a_swap)")
    pb = score_parametric_belief(args.model, stmts, device=args.device)
    assert len(pb["item_id"]) == args.items, (len(pb["item_id"]), args.items)
    assert pb["knows_gold"].dtype == bool
    print(f"    scored {len(pb['item_id'])} items; knows_gold frac={np.mean(pb['knows_gold']):.2f}")

    print("\n[3] analyze.run_layer executes on real features")
    r = analyze.run_layer(feats, keep[len(keep) // 2], "last")
    assert np.isfinite(r["cross_angle"]) and np.isfinite(r["confidence_asymmetry"])
    print(f"    run_layer OK: angle={r['cross_angle']:.1f} asym={r['confidence_asymmetry']:+.2f} "
          f"(values meaningless on a tiny model — plumbing only)")

    print("\nSMOKE TEST PASS — extraction path works on a real HF model.")


if __name__ == "__main__":
    main()
