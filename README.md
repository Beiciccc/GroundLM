# Grounding or Lexical Overlap?

Code and data for *Grounding or Lexical Overlap? A Matched-Pair Study of
Internals-Based Faithfulness Probes* (GroundLM 2026 workshop, EMNLP).

Reference-free hallucination detectors increasingly read a linear
"context-faithfulness" direction out of the residual stream. In the standard extractive
QA setup that direction is almost perfectly confounded with **lexical overlap**
(*r* = 0.99): an answer is context-faithful essentially *iff* its string occurs in the
context, so an apparent grounding probe may be a string-matching detector.

We build a matched **Support × Overlap** construction that halves the confound, and
then ask, control by control, whether anything grounding-specific is left. The answer is
no:

- Within the overlap-controlled stratum a support contrast is decodable (≈0.99 raw,
  0.76–0.88 after purging confidence), but a **no-context ablation** leaves it almost
  unchanged — the probe is reading question–answer compatibility, not the context.
- The direction aligns across four model families up to a linear map, but answer-length,
  overlap and factuality directions align just as well: a generic matched-stimulus
  property, not a grounding axis.
- The synthetic direction does **not** transfer to real hallucinations. On RAGTruth it
  scores below NLI, below model confidence, and below plain lexical overlap. Only an
  in-domain probe trained on human labels recovers a usable signal.

The reusable contribution is the construction itself, in particular the **wrong-question
trap** (cell C): the answer is present in the context but answers a *different*
question. An off-the-shelf NLI model is fooled by ~25% of clean cell-C items.

## Layout

```
paper/                     main.tex, sections.tex, appendix.tex, references.bib, figs/
MANIFEST.md                every paper number -> its producer script and files
reproduce.sh               two-stage reproduction (CPU analysis / GPU extraction)

src/groundlm/
  data/build_ctrlpairs_v2.py   the Support x Overlap construction (cells A/C/D)
  data/build_ragtruth.py       the RAGTruth evaluation subset
  extract/activations.py       residual-stream + confidence extraction (GPU)
  probe/directions.py          mass-mean / logistic directions, signed AUROC
  probe/identification.py      null-calibrated separability
  probe/confidence.py          confidence purge and the asymmetry test
  transfer/procrustes.py       cross-family orthogonal Procrustes, ACS baseline

scripts/
  cr_final_tables.py       every number in Tables 1, 3 and 4, one protocol
  _cr_common.py            grouping, seeded folds, lazy feature loading
  make_figures.py          Figures 1-4
  harden_analyses.py       C3 cosine, cell-C contamination, per-task C4
  analyze_nocontext.py     the no-context / shuffled-context ablation
  cr_*.py                  the audit analyses released with the camera-ready
```

## Reproducing

```bash
pip install -r requirements.txt
bash reproduce.sh
```

Stage B (the default) recomputes every number, table and figure on CPU from cached
residual-stream features. Those caches are 2.4 GB and are not redistributed;
`bash reproduce.sh extract` regenerates them from the released datasets on one
A100-40G, and `reproduce.sh` checks for them and says so if they are missing.

`MANIFEST.md` maps each table, figure and quoted number to the script that produces it.

## Evaluation protocol

Every reported estimate uses one protocol, and it is worth stating because two earlier
choices were wrong:

- **Grouping is by source, not by record.** CtrlPairs folds are grouped by the source QA
  triple, not by `item_id`: `item_id` indexes an NQ-Swap *swap record*, and several
  records share one source QA, so the 1,200 cell-A rows cover only 555 distinct triples.
  RAGTruth folds are grouped by source passage (2,000 responses over 450 passages), not
  by response id.
- **Residualization is fold-local**, fitted on the training fold and applied to its test
  rows.
- **AUROC is signed.** Polarity is fixed on the training fold. No reported number uses
  `max(AUC, 1-AUC)`, since it floors a non-predictive direction above 0.5; the helper is
  named `auroc_maxflip` and only the `cr_protocol_audit_*` scripts call it, to report
  what the old convention cost.
- Folds come from an explicitly seeded splitter rather than `GroupKFold`, whose fold
  assignment depends on `np.argsort` tie-breaking and therefore on the numpy version.

## Data

`data/ctrlpairs_v2.jsonl` (3,593 statements over 1,200 NQ-Swap records / 555 source QAs)
and `data/ragtruth.jsonl` (2,000 responses over 450 passages) are released. Cell B of the
2×2, the paraphrase cell, has zero yield in this release, so the realized design is the
three model-free cells A/C/D.
