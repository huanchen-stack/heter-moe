#!/bin/bash
# =============================================================================
# Perplexity Experiment — Mixed-Precision MoE
#
# Runs WikiText-2 perplexity evaluation across different quantization configs.
#
# Requirements:
#   - Blackwell GPU (RTX 6000 Pro / SM120, or B200 / SM100)
#   - CUDA 12.8+, cuDNN 9.x
#
# Usage:
#   bash run_perplexity.sh                        # full pipeline
#   bash run_perplexity.sh --skip-download        # skip download
#   bash run_perplexity.sh --resume               # resume from checkpoint
#   bash run_perplexity.sh --mode=compute_bound   # seq_len=8192
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
OUTPUT_DIR="${SCRIPT_DIR}/experiment_results"
BACKEND="cudnn"
MODE="memory_bound"
MAX_TOKENS=""
RESUME=false

# Model path
QWEN3_DIR="${MODEL_DIR}/Qwen3-30B-A3B"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ─── Parse args ──────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --resume) RESUME=true ;;
        --mode=*) MODE="${arg#*=}" ;;
        --backend=*) BACKEND="${arg#*=}" ;;
        --max-tokens=*) MAX_TOKENS="${arg#*=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
    esac
done

# =============================================================================
# STEP 0: Environment check
# =============================================================================
echo ""
echo "============================================================"
echo " Perplexity Experiment Pipeline"
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

python -c "from datasets import load_dataset; print('  Datasets: OK')" \
    || err "datasets not installed"

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
# STEP 3: Quick sanity check — mixed precision model
# =============================================================================
echo ""
echo "============================================================"
echo " Step 3: Mixed-precision sanity check"
echo "============================================================"

log "Testing mixed-precision model (bf16 baseline + nvfp4 quantized)..."
python -c "
import torch, logging
logging.basicConfig(level=logging.INFO)
from mixed_precision_model import MultiPrecisionMoEModel

model = MultiPrecisionMoEModel(
    model_path='${QWEN3_DIR}',
    quant_modes=['nvfp4'],
    backend='${BACKEND}',
)

prompts = ['The quick brown fox', 'Once upon a time in a land far away']

# Baseline (bf16)
print('  --- BF16 baseline ---')
for text in model.generate(prompts, max_new_tokens=20):
    print(f'    {text}')

# Quantized (nvfp4, 100% experts)
print('  --- NVFP4 100% quantized ---')
model.quantize_all_layers(quant_mode='nvfp4', quantize_ratio=1.0)
for text in model.generate(prompts, max_new_tokens=20):
    print(f'    {text}')
model.clear_hooks()

# Quantized (nvfp4, 50% experts, coldest)
print('  --- NVFP4 50% quantized (coldest) ---')
from mixed_precision_model import QuantizationCriteria
model.quantize_all_layers(quant_mode='nvfp4', quantize_ratio=0.5, criteria=QuantizationCriteria.COLDEST_COUNT)
for text in model.generate(prompts, max_new_tokens=20):
    print(f'    {text}')
model.clear_hooks()

print('  Sanity check passed!')
del model
torch.cuda.empty_cache()
" || err "Mixed-precision sanity check FAILED"

log "Sanity check passed"

# =============================================================================
# STEP 4: Run perplexity experiment
# =============================================================================
echo ""
echo "============================================================"
echo " Step 4: Perplexity Experiment (mode=${MODE})"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

EXTRA_FLAGS=""
if [ "$RESUME" = true ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --resume"
fi
if [ -n "$MAX_TOKENS" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --max_tokens ${MAX_TOKENS}"
fi

python "${SCRIPT_DIR}/perplexity.py" \
    --model "${QWEN3_DIR}" \
    --mode "${MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --backend "${BACKEND}" \
    ${EXTRA_FLAGS}

log "Perplexity experiment complete"

# =============================================================================
# STEP 5: Summary
# =============================================================================
echo ""
echo "============================================================"
echo " Results"
echo "============================================================"

for f in "${OUTPUT_DIR}"/ppl_*.json; do
    if [ -f "$f" ]; then
        # Skip chunk checkpoint files
        if [[ "$f" == *_chunks.json ]]; then
            continue
        fi
        size=$(du -h "$f" | cut -f1)
        echo "  ${f}  (${size})"
        python -c "
import json
with open('$f') as fp:
    results = json.load(fp)
for r in results:
    print(f\"    {r['config']:<25} PPL={r['perplexity']:.4f}\")
" 2>/dev/null || true
    fi
done

echo ""
log "All done!"
