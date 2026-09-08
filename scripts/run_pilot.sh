#!/usr/bin/env bash
# Pilot + full extraction across the four families (+ scale control).
# One model per GPU on a 4xA100-40G node. CtrlPairs are built once and reused.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${SOURCE:-auto}"          # or a local NQ-Swap .jsonl path
MAX_ITEMS="${MAX_ITEMS:-1500}"
LAYERS="${LAYERS:-mid}"
POOLING="${POOLING:-last}"

declare -A MODELS=(
  [llama31_8b]="meta-llama/Llama-3.1-8B-Instruct"
  [qwen25_7b]="Qwen/Qwen2.5-7B-Instruct"
  [mistral7b_v03]="mistralai/Mistral-7B-Instruct-v0.3"
  [gemma2_9b]="google/gemma-2-9b-it"
  [llama32_3b]="meta-llama/Llama-3.2-3B-Instruct"
)

# Step 1: build CtrlPairs once (shared across models).
python -m groundlm.data.build_ctrlpairs --source "$SOURCE" --max-items "$MAX_ITEMS" \
  --out data/ctrlpairs.jsonl || true

# Step 2: one model per GPU (assign CUDA_VISIBLE_DEVICES round-robin).
gpu=0
for short in "${!MODELS[@]}"; do
  echo "=== $short on GPU $gpu ==="
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src python3 scripts/run_pilot.py \
    --model "${MODELS[$short]}" --source "$SOURCE" --max-items "$MAX_ITEMS" \
    --layers "$LAYERS" --pooling "$POOLING" \
    --ctrlpairs data/ctrlpairs.jsonl --out-dir "runs/$short" &
  gpu=$(( (gpu + 1) % 4 ))
done
wait
echo "all extractions done; reports in runs/*/report.json"
