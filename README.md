# Faithfulness Axis — GroundLM @ EMNLP 2026

Disentangling **context-faithfulness** from **parametric-factuality** in the
residual stream, with a matched-pair identification test, a confidence asymmetry,
cross-family transfer, and a one-pass two-axis conformal grounding gate.
**API-free, no human annotation, ≤4×A100-40G, <1 day.** Full plan: [`DESIGN_DOC.md`](DESIGN_DOC.md).

## Layout
```
DESIGN_DOC.md              locked blueprint (sections 1–10)
src/groundlm/
  data/build_ctrlpairs.py  Stage 0: matched 2×2 CtrlPairs from NQ-Swap
  data/surface_stats.py    Stage 0: surface-balance audit + matched guarantees
  extract/activations.py   Stage 1: residual-stream + confidence extraction; parametric-belief
  probe/directions.py      Stage 2: mass-mean / logistic probes
  probe/identification.py  Stage 3: null-calibrated separability test (C1)
  probe/confidence.py      Stage 4: confidence-asymmetry test (C2)
  analyze.py               Stage 3/4 driver over cached features
scripts/
  synth_validation.py      CPU method-validation on known geometry (C1+C2)
  test_ctrlpairs.py        CPU validation of the 2×2 construction
  run_pilot.py             one-model pilot: build→extract→belief→sweep→report
  run_pilot.sh             all families, one model per GPU
paper/                     main.tex, sections.tex (Intro/Related/Method), references.bib
configs/                   pilot.yaml, models.yaml
```
Stages 5–6 (cross-family Procrustes + ACS baseline; two-axis conformal gate) are next.

## Install
```bash
pip install -r requirements.txt          # CPU analysis needs only numpy/scipy/sklearn
```

## Validate the method locally (no GPU, no downloads)
```bash
python scripts/test_ctrlpairs.py         # 2×2 construction + matched guarantees
python scripts/synth_validation.py       # C1 rejects iff separable; C2 recovers asymmetry
```
Both should print `ALL ... PASS`. `synth_validation.py` doubles as the paper's
"method on known ground-truth" sanity figure.

## Run the pilot on the cluster
```bash
# single model
PYTHONPATH=src python scripts/run_pilot.py \
  --model meta-llama/Llama-3.1-8B-Instruct --source auto \
  --max-items 1500 --layers mid --out-dir runs/llama31_8b

# all four families + scale control, one model per GPU
bash scripts/run_pilot.sh
```
`--source auto` tries known NQ-Swap HF mirrors; if none resolve, pass a local
NQ-Swap `.jsonl` (fields: `question`, `org_answer`, `sub_answer`, `org_context`,
`sub_context`) — source: github.com/apple/ml-knowledge-conflicts.

The go/no-go signal lands in `runs/<model>/report.json`: per-layer cross-notion
angle vs within-notion null (C1) and confidence asymmetry (C2). Use
`--knows-gold-only` to condition C2/C3 on the per-model parametric-belief check.

## Build the paper
Add the ACL style files (`acl.sty`, `acl_natbib.bst`) from
github.com/acl-org/acl-style-files into `paper/`, then `latexmk -pdf paper/main.tex`.
Open reviewer risks tracked in `paper/REVIEWER_RISK_NOTES.md`.
