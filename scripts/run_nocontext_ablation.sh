#!/usr/bin/env bash
# Point-3 ablation: is the within-O=1 support signal (A-vs-C) actually context-grounding,
# or just question-answer compatibility / answer plausibility that needs NO context?
#
# Re-extracts CtrlPairs-v2 activations under two ablated prompts, same models/layers as main:
#   no_context       : "Question: {q}\nAnswer: {a}"            (context removed entirely)
#   shuffled_context : unrelated donor context + {q} + {a}     (context present but irrelevant)
# If supp_O1 (A-vs-C) stays high under no_context, the signal is NOT grounding.
# If it drops toward chance vs the normal run, the signal genuinely uses the context.
#
# Needs a GPU (4x A100-40G plenty; one GPU fine). If huggingface.co is unreachable, set
#   export HF_ENDPOINT=<your-mirror-endpoint>
# and point HF_HOME/caches/runs at the big data disk first (see scripts/run_strengthen.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

MODELS=(qwen25_7b mistral7b_v03 llama31_8b gemma2_9b)
declare -A HF=(
  [qwen25_7b]="Qwen/Qwen2.5-7B-Instruct"
  [mistral7b_v03]="mistralai/Mistral-7B-Instruct-v0.3"
  [llama31_8b]="meta-llama/Llama-3.1-8B-Instruct"
  [gemma2_9b]="google/gemma-2-9b-it"
)
declare -A BS=( [qwen25_7b]=4 [mistral7b_v03]=16 [llama31_8b]=6 [gemma2_9b]=2 )  # large-vocab OOM-safe
declare -A LAYER=( [qwen25_7b]=14 [mistral7b_v03]=16 [llama31_8b]=16 [gemma2_9b]=21 )  # a-priori depth-0.5

for m in "${MODELS[@]}"; do
  for mode in no_context shuffled_context; do
    suf=$([ "$mode" = "no_context" ] && echo nc || echo shuf)
    out="runs/${m}_v2_${suf}"
    if [ -f "$out/features.npz" ]; then echo "### skip $out (already done) ###"; continue; fi
    echo "### $m / $mode -> $out ###"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/run_extract.py \
      --model "${HF[$m]}" --ctrlpairs data/ctrlpairs_v2.jsonl \
      --layers "${LAYER[$m]}" --pooling last --batch-size "${BS[$m]}" --max-len 1024 \
      --prompt-mode "$mode" --skip-belief --out-dir "$out"
  done
done
echo "### DONE. Pull runs/*_v2_nc and runs/*_v2_shuf, then: python3 scripts/analyze_nocontext.py ###"
