#!/usr/bin/env bash
# Canonical reproduction of every number and figure in the paper.
# Two stages: (A) GPU extraction from scratch [optional, needs A100],
# (B) CPU analysis from cached features [reproduces all paper numbers + figures + PDF].
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

STAGE="${1:-analysis}"   # 'extract' (GPU, from scratch) or 'analysis' (default, from cached runs/*/features.npz)

if [ "$STAGE" = "extract" ]; then
  echo "### STAGE A: GPU extraction (datasets + 4 families + RAGTruth + no-context/shuffled) ###"
  # If huggingface.co is unreachable, set HF_ENDPOINT to a mirror and point the HF caches at a large disk first.
  bash scripts/run_strengthen.sh          # build_v2 + build_ragtruth + run_extract per model (+disk cleanup)
  bash scripts/run_nocontext_ablation.sh  # runs/*_v2_nc + runs/*_v2_shuf, needed by analyze_nocontext.py
fi

# Stage B consumes runs/*/features.npz (~2.4 GB), which are too large for the source
# repository. Fail loudly rather than half-way through if they are absent.
if [ "$STAGE" != "extract" ] && ! ls runs/*/features.npz >/dev/null 2>&1; then
  echo "ERROR: no runs/*/features.npz found." >&2
  echo "  The 2.4 GB caches are not redistributed; regenerate them with:" >&2
  echo "    bash reproduce.sh extract     (needs one A100-40G)" >&2
  exit 1
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
# 2d. CANONICAL: every reported table number under one source-grouped, fold-local,
#     signed-AUROC protocol. Tables 1, 3 and 4 come from here.
python3 scripts/cr_final_tables.py               # -> runs/cr_final_tables.json
python3 scripts/cr_joint_grouped_foldlocal.py    # -> runs/cr_joint_grouped_foldlocal.json
# 2e. Supporting analyses released with the camera-ready (see MANIFEST.md).
python3 scripts/cr_signed_auroc_ragtruth.py     # -> runs/cr_signed_auroc_ragtruth.json
python3 scripts/cr_estimator_domain_control.py  # -> runs/cr_estimator_domain_control.json
python3 scripts/cr_c2_and_separability.py       # -> runs/cr_c2_and_separability.json
python3 scripts/cr_band_and_pairs.py            # -> runs/cr_band_and_pairs.json
python3 scripts/heldout_procrustes.py           # -> runs/heldout_procrustes.json
# 2f. Leakage audits released in response to review (cheap; no features needed for the first two).
python3 scripts/cr_dupleak_audit.py             # -> runs/cr_dupleak_audit.json
python3 scripts/cr_dupleak_table1.py            # -> runs/cr_dupleak_table1.json
python3 scripts/cr_passage_leakage.py           # -> runs/cr_passage_leakage.json
# 3. Figures (read the JSONs above; no hardcoded results).
python3 scripts/make_figures.py           # -> paper/figs/*.pdf
# 4. Paper (needs the ACL style files acl.sty + acl_natbib.bst already in paper/).
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "### DONE -> paper/main.pdf ###"
# Local method-validation sanity checks (synthetic, fast):
cd .. && python3 scripts/synth_validation.py >/dev/null && python3 scripts/test_ctrlpairs.py >/dev/null && echo "method-validation checks PASS"
