import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.data.build_ctrlpairs_v2 import build_decoupled
from groundlm.data.build_ctrlpairs import load_source
from groundlm.data.verify_support import verify_and_audit

raw = load_source("pminervini/NQ-Swap", "dev")
stmts = build_decoupled(raw, max_items=250)   # cells A/C/D, model-free
rep = verify_and_audit(stmts, device="cpu")
print("agreement NLI vs by-construction:", round(rep["agreement_nli_vs_byconstruction"], 3))
print("corr(by-constr support, overlap):", round(rep["corr_byconstr_support_overlap"], 3))
print("corr(NLI support, overlap):      ", round(rep["corr_nli_support_overlap"], 3))
print("per cell (byconstr S / NLI S / mean entail prob):")
for c, d in rep["per_cell"].items():
    print(f"  {c}: n={d['n']:4d}  byS={d['byconstr_support_rate']:.2f}  nliS={d['nli_support_rate']:.2f}  P(entail)={d['mean_entail_prob']:.2f}")
