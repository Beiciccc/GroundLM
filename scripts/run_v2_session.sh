#!/usr/bin/env bash
# Bounded v2 GPU session: build (no paraphrase by default) -> extract chosen models
# (gated failures non-blocking) -> per-model C1/C2 + C4 -> C3 transfer if >=2 models.
set -uo pipefail        # NOT -e: one model's failure must not abort the run
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh || true

SOURCE="${SOURCE:-pminervini/NQ-Swap}"; SPLIT="${SPLIT:-dev}"
MAX_ITEMS="${MAX_ITEMS:-1200}"; PARAPHRASER="${PARAPHRASER:-}"; LAYERS="${LAYERS:-mid}"
MODELS_STR="${MODELS_STR:-qwen25_7b:Qwen/Qwen2.5-7B-Instruct mistral7b_v03:mistralai/Mistral-7B-Instruct-v0.3 llama31_8b:meta-llama/Llama-3.1-8B-Instruct}"

echo "### BUILD v2 (paraphraser='$PARAPHRASER', max_items=$MAX_ITEMS) ###"
PYTHONPATH=src python3 scripts/build_v2.py --source "$SOURCE" --split "$SPLIT" \
  --max-items "$MAX_ITEMS" --paraphraser "$PARAPHRASER" --device cuda \
  --out data/ctrlpairs_v2.jsonl --belief-out data/belief_items.jsonl \
  || { echo "BUILD_FAILED"; exit 1; }

OK_DIRS=()
for pair in $MODELS_STR; do
  short="${pair%%:*}"; mid="${pair##*:}"
  echo "### EXTRACT $short ($mid) ###"
  if PYTHONPATH=src python3 scripts/run_extract.py --model "$mid" \
       --ctrlpairs data/ctrlpairs_v2.jsonl --belief-items data/belief_items.jsonl \
       --layers "$LAYERS" --out-dir "runs/${short}_v2"; then
    OK_DIRS+=("runs/${short}_v2")
    echo "### ANALYZE C1/C2 $short ###"
    PYTHONPATH=src python3 scripts/analyze_v2.py --dir "runs/${short}_v2" || true
    echo "### ANALYZE C4 gate $short ###"
    PYTHONPATH=src python3 scripts/analyze_gate_v2.py --dir "runs/${short}_v2" || true
  else
    echo "### SKIP $short (extract/download failed — likely gated) ###"
  fi
done

if [ "${#OK_DIRS[@]}" -ge 2 ]; then
  echo "### C3 TRANSFER across ${OK_DIRS[*]} ###"
  PYTHONPATH=src python3 scripts/analyze_transfer_v2.py --dirs "${OK_DIRS[@]}" || true
fi
echo "### V2_SESSION_DONE (models ok: ${OK_DIRS[*]:-none}) ###"
