"""Committed producer for the v1 (extractive NQ-Swap) confound number used in the paper.

Reports corr(SUPPORT, OVERLAP) on the *extractive* construction (no decoupling), using the
SAME exact-answer-string overlap definition as the decoupled v2 cells (overlap_measured),
so the v1->v2 drop is apples-to-apples. Writes runs/v1_corr.json; no GPU/model needed."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from groundlm.data.build_ctrlpairs import build_ctrlpairs, load_source

stmts = build_ctrlpairs(load_source("pminervini/NQ-Swap", "dev"))
g = np.array([s.grounding for s in stmts])
o_exact = np.array([1 if s.asserted_answer.strip().lower() in s.context.lower() else 0 for s in stmts])
o_token = np.array([1 if s.lexical_overlap > 0 else 0 for s in stmts])
out = {
    "n": int(len(stmts)),
    "corr_support_overlap_exactstring": round(float(np.corrcoef(g, o_exact)[0, 1]), 4),
    "corr_support_overlap_tokenpresent": round(float(np.corrcoef(g, o_token)[0, 1]), 4),
    "overlap_def": "exactstring",  # the value the paper cites (matches v2 overlap_measured)
}
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "runs"), exist_ok=True)
p = os.path.join(os.path.dirname(__file__), "..", "runs", "v1_corr.json")
with open(p, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
