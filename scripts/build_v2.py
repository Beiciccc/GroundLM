"""Build the decoupled CtrlPairs-v2 dataset (cells A/C/D + paraphrased cell B) and
the belief-items file. Run once on the GPU box (cell-B paraphrasing needs a model)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.data.build_ctrlpairs import load_source, normalize_record  # noqa: E402
from groundlm.data.build_ctrlpairs_v2 import build_decoupled, decoupling_audit, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pminervini/NQ-Swap")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-items", type=int, default=1500)
    ap.add_argument("--paraphraser", default="Qwen/Qwen2.5-7B-Instruct",
                    help="open instruct model for cell B; '' to skip cell B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/ctrlpairs_v2.jsonl")
    ap.add_argument("--belief-out", default="data/belief_items.jsonl")
    ap.add_argument("--skip-nli", action="store_true", help="skip per-statement NLI baseline scoring")
    args = ap.parse_args()

    raw = load_source(args.source, args.split)
    stmts = build_decoupled(raw, max_items=args.max_items)        # cells A/C/D (model-free)

    if args.paraphraser:
        from groundlm.data.paraphrase import Paraphraser, build_cell_b
        para = Paraphraser(args.paraphraser, device=args.device)
        cellb, stats = build_cell_b(raw, para, nli_device=args.device,
                                    max_items=args.max_items)
        print("cell-B yield:", stats)
        stmts = stmts + cellb

    if not args.skip_nli:
        from groundlm.data.verify_support import nli_entailment
        probs = nli_entailment(stmts, device=args.device)["entail_prob"]
        for s, p in zip(stmts, probs):
            s.nli_entail_prob = float(p)
        print(f"scored NLI entailment for {len(stmts)} statements (the C4 baseline)")

    save(stmts, args.out)
    print(f"saved {len(stmts)} statements -> {args.out}")
    print(json.dumps(decoupling_audit(stmts), indent=2))

    # belief items (item_id, question, a_true, a_swap) for parametric-belief scoring
    recs = [r for r in (normalize_record(x) for x in raw) if r is not None][:args.max_items]
    used = sorted({s.item_id for s in stmts})
    os.makedirs(os.path.dirname(args.belief_out) or ".", exist_ok=True)
    with open(args.belief_out, "w") as fh:
        for i in used:
            r = recs[i]
            fh.write(json.dumps({"item_id": i, "question": r["question"],
                                 "a_true": r["a_true"], "a_swap": r["a_swap"]}) + "\n")
    print(f"saved {len(used)} belief items -> {args.belief_out}")


if __name__ == "__main__":
    main()
