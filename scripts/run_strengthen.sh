#!/usr/bin/env bash
# Strengthening GPU session: 4 families (per-model batch size; disk-cleanup after
# each to fit the 49G data disk) + RAGTruth real-hallucination C4 eval + C3 transfer.
# Builds CtrlPairs-v2 (cells A/C/D, NLI-scored) and RAGTruth once.
set -uo pipefail        # NOT -e: gated/OOM failures must not abort the run
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh || true

MAX_ITEMS=${MAX_ITEMS:-1200}; RT_ITEMS=${RT_ITEMS:-2000}; LAYERS=${LAYERS:-mid}

echo "### BUILD datasets (CtrlPairs-v2 + RAGTruth, with NLI baselines) ###"
[ -f data/ctrlpairs_v2.jsonl ] || PYTHONPATH=src python3 scripts/build_v2.py \
  --paraphraser "" --max-items "$MAX_ITEMS" \
  --out data/ctrlpairs_v2.jsonl --belief-out data/belief_items.jsonl || { echo BUILD_CP_FAILED; exit 1; }
[ -f data/ragtruth.jsonl ] || PYTHONPATH=src python3 scripts/build_ragtruth.py \
  --max-items "$RT_ITEMS" --out data/ragtruth.jsonl || echo "RAGTruth build failed (continuing)"

# short:hf_id:batch_size  (large vocab -> smaller batch; Gemma 256k vocab -> 2)
MODELS=(
  "qwen25_7b:Qwen/Qwen2.5-7B-Instruct:4"
  "mistral7b_v03:mistralai/Mistral-7B-Instruct-v0.3:16"
  "llama31_8b:meta-llama/Llama-3.1-8B-Instruct:6"
  "gemma2_9b:google/gemma-2-9b-it:2"
)
CP_DIRS=()
for entry in "${MODELS[@]}"; do
  IFS=: read -r short mid bs <<< "$entry"
  echo "### EXTRACT $short ($mid, bs=$bs) ###"
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src python3 scripts/run_extract.py \
       --model "$mid" --ctrlpairs data/ctrlpairs_v2.jsonl --belief-items data/belief_items.jsonl \
       --layers "$LAYERS" --batch-size "$bs" --out-dir "runs/${short}_v2"; then
    CP_DIRS+=("runs/${short}_v2")
    if [ -f data/ragtruth.jsonl ]; then
      echo "### EXTRACT $short RAGTruth (max-len 2048) ###"
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src python3 scripts/run_extract.py \
        --model "$mid" --ctrlpairs data/ragtruth.jsonl --belief-items "" \
        --layers "$LAYERS" --batch-size "$bs" --max-len 2048 --out-dir "runs/${short}_rt" \
        && PYTHONPATH=src python3 scripts/analyze_gate_ragtruth.py --cp-dir "runs/${short}_v2" --rt-dir "runs/${short}_rt" || echo "rt failed $short"
    fi
    PYTHONPATH=src python3 scripts/analyze_v2.py --dir "runs/${short}_v2" || true
  else
    echo "### SKIP $short (extract failed — gated/OOM) ###"
  fi
  # free this model's weights from the 49G data disk before the next one
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${mid//\//--}" 2>/dev/null && echo "freed cache: $short"
done

if [ "${#CP_DIRS[@]}" -ge 2 ]; then
  echo "### C3 TRANSFER across ${CP_DIRS[*]} ###"
  PYTHONPATH=src python3 scripts/analyze_transfer_v2.py --dirs "${CP_DIRS[@]}" || true
fi
echo "### STRENGTHEN_DONE (models ok: ${CP_DIRS[*]:-none}) ###"
