#!/usr/bin/env bash
# One-command v2 GPU session: build decoupled dataset (cell-B paraphrase) ->
# extract per model (one GPU each, model loaded once) -> per-model C1/C2 analysis.
# Cross-family transfer (C3) + gate (C4) are a follow-up step, run only if C1
# confirms a grounding axis beyond overlap.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh || true     # optional: HF_ENDPOINT (mirror), HF_HOME, HF_TOKEN

SOURCE="${SOURCE:-pminervini/NQ-Swap}"
SPLIT="${SPLIT:-dev}"
MAX_ITEMS="${MAX_ITEMS:-1500}"
PARAPHRASER="${PARAPHRASER:-Qwen/Qwen2.5-7B-Instruct}"   # '' to skip cell B (cells A/C/D only)
LAYERS="${LAYERS:-mid}"

echo "=== Step 1: build decoupled CtrlPairs-v2 (cell-B paraphrase on GPU) ==="
PYTHONPATH=src python3 scripts/build_v2.py --source "$SOURCE" --split "$SPLIT" \
  --max-items "$MAX_ITEMS" --paraphraser "$PARAPHRASER" --device cuda \
  --out data/ctrlpairs_v2.jsonl --belief-out data/belief_items.jsonl

declare -A MODELS=(
  [llama31_8b]="meta-llama/Llama-3.1-8B-Instruct"
  [qwen25_7b]="Qwen/Qwen2.5-7B-Instruct"
  [mistral7b_v03]="mistralai/Mistral-7B-Instruct-v0.3"
  [gemma2_9b]="google/gemma-2-9b-it"
  [llama32_3b]="meta-llama/Llama-3.2-3B-Instruct"
)

echo "=== Step 2: extract per model (one GPU each) ==="
n_gpu=$(nvidia-smi -L | wc -l); gpu=0
for short in "${!MODELS[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src python3 scripts/run_extract.py \
    --model "${MODELS[$short]}" --ctrlpairs data/ctrlpairs_v2.jsonl \
    --belief-items data/belief_items.jsonl --layers "$LAYERS" \
    --out-dir "runs/${short}_v2" > "runs_${short}_v2.log" 2>&1 &
  gpu=$(( (gpu + 1) % n_gpu ))
done
wait

echo "=== Step 3: per-model C1/C2 analysis ==="
for short in "${!MODELS[@]}"; do
  echo "--- $short ---"
  PYTHONPATH=src python3 scripts/analyze_v2.py --dir "runs/${short}_v2" || true
done

echo "=== Step 4: C3 cross-family transfer (all models) ==="
PYTHONPATH=src python3 scripts/analyze_transfer_v2.py --dirs runs/*_v2 || true

echo "=== Step 5: C4 two-axis conformal gate (per model) ==="
for short in "${!MODELS[@]}"; do
  echo "--- $short ---"
  PYTHONPATH=src python3 scripts/analyze_gate_v2.py --dir "runs/${short}_v2" || true
done

echo "v2 done."
echo "  C1/C2: runs/*_v2/report_v2.json   (support_auroc_O1 = grounding beyond overlap)"
echo "  C3:    runs/transfer_v2.json       (support procrustes_purged off-diag)"
echo "  C4:    runs/*_v2/report_gate_v2.json (support gate vs NLI on overlap=1 subset)"
