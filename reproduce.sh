#!/usr/bin/env bash
# Canonical reproduction of every number and figure in the paper.
# Two stages: (A) GPU extraction from scratch [optional, needs A100],
# (B) CPU analysis from cached features [reproduces all paper numbers + figures + PDF].
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

STAGE="${1:-analysis}"   # 'extract' (GPU, from scratch) or 'analysis' (default, from cached runs/*/features.npz)

if [ "$STAGE" = "extract" ]; then
  echo "### STAGE A: GPU extraction (build datasets + extract 4 families + RAGTruth) ###"
  # If huggingface.co is unreachable, set HF_ENDPOINT to a mirror and point the HF caches at a large disk first.
  bash scripts/run_strengthen.sh   # build_v2 + build_ragtruth + run_extract per model (+disk cleanup)
fi

echo "### STAGE B: analysis from cached features (no GPU) ###"
# 1. C1 (a-priori depth-0.5 layers) + C3 (cosine + signed nulls + factuality-within) +
#    cell-C contamination + C4 per-task + in-domain probe (wired into gate JSONs).
python3 scripts/harden_analyses.py        # -> runs/c1_apriori.json, runs/hardened.json, runs/*_rt/report_gate_ragtruth.json
# 1b. Reviewer-response analyses: clean (collision-free) Table 1, Procrustes nuisance
#     baseline (per-pair + length/overlap/confidence cosine), per-task synth/conf/in-domain.
python3 scripts/reviewer_analyses.py      # -> runs/reviewer_analyses.json
# 1c. No-context ablation (Point 3): supp_O1 under normal / no_context / shuffled prompts.
#     Needs runs/*_v2_nc + *_v2_shuf from the GPU extract stage (scripts/run_nocontext_ablation.sh).
python3 scripts/analyze_nocontext.py      # -> runs/nocontext_ablation.json
# 2. Cross-family transfer matrix (for the C3 heatmap) and per-model C1/C2.
python3 scripts/analyze_transfer_v2.py --dirs runs/qwen25_7b_v2 runs/mistral7b_v03_v2 runs/llama31_8b_v2 runs/gemma2_9b_v2 || true
# 2b. v1 (extractive) confound number for the decoupling figure (CPU; no model).
python3 scripts/v1_corr.py                # -> runs/v1_corr.json
# 2c. Sentence-level (SummaC-style) NLI baseline on RAGTruth -> Table c4_pertask rows.
#     Uses the small DeBERTa-MNLI model only (no LLM); ~30 min on CPU,
#     checkpoints to runs/_nli_sent_cache/ and resumes if interrupted.
python3 scripts/nli_sentence_ragtruth.py  # -> runs/nli_sentence_ragtruth.json
# 3. Figures (read the JSONs above; no hardcoded results).
python3 scripts/make_figures.py           # -> paper/figs/*.pdf
# 4. Paper (needs the ACL style files acl.sty + acl_natbib.bst already in paper/).
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "### DONE -> paper/main.pdf ###"
# Local method-validation sanity checks (synthetic, fast):
cd .. && python3 scripts/synth_validation.py >/dev/null && python3 scripts/test_ctrlpairs.py >/dev/null && echo "method-validation checks PASS"
