"""Draft paper/references.bib from the verified citation data (runs/cite_verified.json).

Superseded: paper/references.bib is now maintained by hand. The camera-ready reference
audit fixed things this generator cannot reproduce -- brace-protected acronyms (acl_natbib
lowercases unprotected capitals, so an unprotected title renders as "Debertav3"),
LaTeX-escaped diacritics, and three 2026 venue upgrades verified against the ACL
Anthology. Re-running this would revert them, so it writes to references.bib.generated
and leaves references.bib alone; diff the two if you want to see what the ledger holds.
"""
import json, os

c = json.load(open(os.path.join(os.path.dirname(__file__), "..", "runs", "cite_verified.json")))

# short, formal booktitle/journal per verified_venue entry
SHORT = {
    "longpre2021nqswap": ("inproceedings", "EMNLP"),
    "niu2024ragtruth": ("inproceedings", "ACL"),
    "azaria2023saplma": ("inproceedings", "Findings of EMNLP"),
    "marks2024geometry": ("inproceedings", "COLM"),
    "burger2024truthuniversal": ("inproceedings", "NeurIPS"),
    "bao2025probing": ("inproceedings", "Findings of ACL"),
    "ming2025faitheval": ("inproceedings", "ICLR"),
    "manakul2023selfcheckgpt": ("inproceedings", "EMNLP"),
    "shi2024cad": ("inproceedings", "NAACL"),
    "zha2023alignscore": ("inproceedings", "ACL"),
    "he2023debertav3": ("inproceedings", "ICLR"),
    "sun2025redeep": ("inproceedings", "ICLR"),
    "wang2025space": ("inproceedings", "NeurIPS"),
    "zhao2024residualconflict": ("inproceedings", "NeurIPS Workshop on Foundation Model Interventions"),
    "mnli": ("inproceedings", "NAACL-HLT"),
    "fever": ("inproceedings", "NAACL-HLT"),
    "anli": ("inproceedings", "ACL"),
    "nli_checkpoint": ("article", "Political Analysis"),
    "schonemann1966": ("article", "Psychometrika"),
}
JOURNAL_EXTRA = {"nli_checkpoint": "volume={32}, number={1}, ", "schonemann1966": "volume={31}, number={1}, "}
DOWNGRADE = {"adarsh2026context", "gao2026proberag"}   # unconfirmed future venue -> arXiv
# Mega-author technical reports: use the official corporate authorship so the ACL bst
# renders a clean "Team. 2024." instead of the truncated "and 1 others".
OVERRIDE_AUTHORS = {"qwen25": "{Qwen Team}", "gemma2": "{Gemma Team}", "llama31": "{Llama Team}"}

LABAN = """@article{laban2022summac,
  title={{SummaC}: Re-Visiting {NLI}-based Models for Inconsistency Detection in Summarization},
  author={Laban, Philippe and Schnabel, Tobias and Bennett, Paul N. and Hearst, Marti A.},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10}, pages={163--177}, year={2022}, publisher={MIT Press}}"""


def gen(k, d):
    title, yr, arx = d["title"], d["year"], d["arxiv_id"]
    auth = OVERRIDE_AUTHORS.get(k, d["authors"])
    if k in DOWNGRADE or d["status"] == "arxiv_only":
        return (f"@article{{{k},\n  title={{{title}}},\n  author={{{auth}}},\n"
                f"  journal={{arXiv preprint arXiv:{arx}}}, year={{{yr}}}}}")
    typ, ven = SHORT[k]
    pages = d.get("pages", "")
    if typ == "inproceedings":
        s = f"@inproceedings{{{k},\n  title={{{title}}},\n  author={{{auth}}},\n  booktitle={{{ven}}}, year={{{yr}}}"
        if pages:
            s += f", pages={{{pages}}}"
        return s + "}"
    extra = JOURNAL_EXTRA.get(k, "")
    s = f"@article{{{k},\n  title={{{title}}},\n  author={{{auth}}},\n  journal={{{ven}}}, {extra}year={{{yr}}}"
    if pages:
        s += f", pages={{{pages}}}"
    return s + "}"


GROUPS = [
    ("datasets / benchmarks", ["longpre2021nqswap", "ming2025faitheval", "niu2024ragtruth"]),
    ("truth / factuality geometry", ["azaria2023saplma", "marks2024geometry", "burger2024truthuniversal",
                                     "azizian2025orthogonal", "bao2025probing", "cho2026confidencemanifold", "noanswer2025"]),
    ("faithfulness/factuality as joint geometric objects", ["wang2025space", "adarsh2026context", "gao2026proberag"]),
    ("residual stream / knowledge conflict / attribution", ["zhao2024residualconflict", "sun2025redeep", "brink2026attribution"]),
    ("cross-family transfer (orthogonal Procrustes + anchor projection)", ["schonemann1966", "puri2025atlas", "kim2026crossfamily"]),
    ("faithfulness-vs-factuality gates / contextual probes", ["fadeeva2025franq", "zhu2026saber", "oneill2025singledirection"]),
    ("NLI baseline: DeBERTaV3 checkpoint + training data + related scorers", ["he2023debertav3", "nli_checkpoint", "mnli", "fever", "anli", "laban2022summac", "zha2023alignscore"]),
    ("other tools / baselines", ["manakul2023selfcheckgpt", "shi2024cad"]),
    ("probed model families", ["llama31", "qwen25", "mistral7b", "gemma2"]),
]

out = ["% References for the GroundLM paper.",
       "% Verified against ACL Anthology / arXiv / OpenReview / Cambridge Core; per-entry sources in runs/cite_verified.json.", ""]
n = 0
for gtitle, keys in GROUPS:
    out.append(f"% ---- {gtitle} ----")
    for k in keys:
        out.append(LABAN if k == "laban2022summac" else gen(k, c[k]))
        out.append("")
        n += 1
dest = os.path.join(os.path.dirname(__file__), "..", "paper", "references.bib.generated")
open(dest, "w").write("\n".join(out))
print(f"wrote {n} entries to {dest}")
print("references.bib is hand-maintained and was not touched; diff against it if needed.")
