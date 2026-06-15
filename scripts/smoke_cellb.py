import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.data.build_ctrlpairs import load_source
from groundlm.data.build_ctrlpairs_v2 import build_decoupled, decoupling_audit
from groundlm.data.paraphrase import Paraphraser, build_cell_b

raw = load_source("pminervini/NQ-Swap", "dev")
N = 8
para = Paraphraser("Qwen/Qwen2.5-0.5B-Instruct", device="cpu", max_new_tokens=110)
acd = build_decoupled(raw, max_items=N)                       # cells A/C/D
cellb, stats = build_cell_b(raw, para, nli_device="cpu", max_items=N)
print("cell-B yield stats:", stats)
if cellb:
    s = cellb[0]
    print("\nexample cell-B context (answer '%s' should be ABSENT):" % s.asserted_answer)
    print(" ", s.context[:300].replace("\n", " "))
print("\ncorr(support,overlap) A/C/D only:", round(decoupling_audit(acd)["corr_support_overlapbin"], 3))
allcells = acd + cellb
print("corr(support,overlap) A/B/C/D    :", round(decoupling_audit(allcells)["corr_support_overlapbin"], 3),
      "| cell counts:", decoupling_audit(allcells)["cell_counts"])
