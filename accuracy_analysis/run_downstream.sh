#!/bin/bash
# =============================================================================
# Downstream Evaluation — Mixed-Precision MoE
#
# Runs MMLU-CoT, GSM8k-CoT, HellaSwag, WinoGrande with lm-evaluation-harness.
#
# Requirements:
#   - Blackwell GPU (RTX 6000 Pro / SM120, or B200 / SM100)
#   - CUDA 12.8+, cuDNN 9.x
#   - lm-eval (pip install lm-eval)
#
# Usage:
#   bash run_downstream.sh                                  # full pipeline
#   bash run_downstream.sh --skip-download                  # skip download
#   bash run_downstream.sh --resume                         # resume from checkpoint
#   bash run_downstream.sh --tasks=hellaswag,winogrande     # subset of tasks
#   bash run_downstream.sh --limit=100                      # quick smoke test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
OUTPUT_DIR="${SCRIPT_DIR}/downstream_results"
BACKEND="cudnn"
BATCH_SIZE="auto"
LIMIT=""
TASKS=""
NUM_FEWSHOT=""

QWEN3_DIR="${MODEL_DIR}/Qwen3-30B-A3B"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

SKIP_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --backend=*) BACKEND="${arg#*=}" ;;
        --batch-size=*) BATCH_SIZE="${arg#*=}" ;;
        --limit=*) LIMIT="${arg#*=}" ;;
        --tasks=*) TASKS="${arg#*=}" ;;
        --num-fewshot=*) NUM_FEWSHOT="${arg#*=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
    esac
done

# =============================================================================
# STEP 0: Environment check
# =============================================================================
echo ""
echo "============================================================"
echo " Downstream Evaluation Pipeline"
echo "============================================================"

log "Checking environment..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'  SM:      {cap[0]}{cap[1]}')
" || err "PyTorch not available"

python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" \
    || err "transformers not installed"

python -c "import lm_eval; print(f'  lm-eval: {lm_eval.__version__}')" \
    || err "lm-eval not installed. Run: pip install lm-eval"

echo ""

# =============================================================================
# STEP 1: Download model (if needed)
# =============================================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "============================================================"
    echo " Step 1: Download model"
    echo "============================================================"

    python "${SCRIPT_DIR}/downloader.py" --base_dir "${MODEL_DIR}"

    log "Download complete"
else
    warn "Skipping download (--skip-download)"
fi

# =============================================================================
# STEP 2: Verify model
# =============================================================================
echo ""
echo "============================================================"
echo " Step 2: Verify model"
echo "============================================================"

if [ ! -d "$QWEN3_DIR" ] || [ ! -f "${QWEN3_DIR}/config.json" ]; then
    err "Model not found at ${QWEN3_DIR}. Run without --skip-download first."
fi

python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('${QWEN3_DIR}', trust_remote_code=True)
print(f'  Qwen3-30B-A3B:')
print(f'    model_type:    {config.model_type}')
print(f'    num_layers:    {config.num_hidden_layers}')
print(f'    hidden_size:   {config.hidden_size}')
print(f'    num_experts:   {getattr(config, \"num_experts\", \"N/A\")}')
" || err "Failed to load model config"

log "Model verified"

# =============================================================================
# STEP 3: Pre-download evaluation datasets
# =============================================================================
echo ""
echo "============================================================"
echo " Step 3: Pre-download evaluation datasets"
echo "============================================================"

python -c "
from datasets import load_dataset
for name, config in [
    ('cais/mmlu', 'all'),
    ('openai/gsm8k', 'main'),
    ('Rowan/hellaswag', None),
    ('allenai/winogrande', 'winogrande_xl'),
]:
    print(f'  Caching {name} ({config or \"default\"})...', end=' ', flush=True)
    load_dataset(name, config, trust_remote_code=True)
    print('OK')
print('  All datasets cached.')
"

log "Datasets ready"

# =============================================================================
# STEP 4: Run downstream evaluation
# =============================================================================
echo ""
echo "============================================================"
echo " Step 4: Downstream Evaluation"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

EXTRA_FLAGS="--resume"
if [ -n "$LIMIT" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --limit ${LIMIT}"
fi
if [ -n "$NUM_FEWSHOT" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --num_fewshot ${NUM_FEWSHOT}"
fi

TASK_FLAGS=""
if [ -n "$TASKS" ]; then
    IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
    TASK_FLAGS="--tasks ${TASK_ARRAY[*]}"
fi

python "${SCRIPT_DIR}/downstream.py" \
    --model "${QWEN3_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --backend "${BACKEND}" \
    --batch_size "${BATCH_SIZE}" \
    ${TASK_FLAGS} \
    ${EXTRA_FLAGS}

log "Downstream evaluation complete"

# =============================================================================
# STEP 5: Summary
# =============================================================================
echo ""
echo "============================================================"
echo " Results"
echo "============================================================"

for f in "${OUTPUT_DIR}"/downstream_*.json; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  ${f}  (${size})"
        python -c "
import json
with open('$f') as fp:
    results = json.load(fp)
for r in results:
    tasks_str = ', '.join(
        f\"{info['display_name']}={info['primary_metric']:.4f}\"
        for t, info in r['tasks'].items()
        if info['primary_metric'] is not None
    )
    print(f\"    {r['config']:<25} {tasks_str}\")
" 2>/dev/null || true
    fi
done

echo ""
log "All done!"
