"""Build the RAGTruth eval jsonl (+ optional NLI baseline scoring). Run on the box."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.data.build_ragtruth import build_ragtruth, save, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="wandb/RAGTruth-processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-items", type=int, default=2000)
    ap.add_argument("--task-types", nargs="*", default=None, help="subset e.g. QA Summary Data2txt")
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/ragtruth.jsonl")
    args = ap.parse_args()

    stmts = build_ragtruth(args.source, args.split, max_items=args.max_items, task_types=args.task_types)
    print(json.dumps(summary(stmts), indent=2))
    if not args.skip_nli and stmts:
        from groundlm.data.verify_support import nli_entailment
        probs = nli_entailment(stmts, device=args.device)["entail_prob"]
        for s, p in zip(stmts, probs):
            s.nli_entail_prob = float(p)
        print(f"scored NLI on {len(stmts)} RAGTruth items")
    save(stmts, args.out)
    print(f"saved {len(stmts)} -> {args.out}")


if __name__ == "__main__":
    main()
