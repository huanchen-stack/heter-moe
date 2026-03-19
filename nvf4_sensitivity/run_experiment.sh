#!/bin/bash
# =============================================================================
# NVFP4 Sensitivity Analysis — Full Experiment Pipeline
#
# Downloads models, verifies, runs sanity inference, then runs NVFP4
# sensitivity analysis on both models.
#
# Requirements:
#   - Blackwell GPU (RTX 6000 Pro / SM120, or B200 / SM100)
#   - CUDA 12.8+, cuDNN 9.x
#   - ~100GB disk (Qwen1.5-MoE ~29GB + Qwen3-30B ~60GB)
#
# Usage:
#   bash run_experiment.sh              # full pipeline (both models)
#   bash run_experiment.sh --skip-download   # skip download, run experiments only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
CALIB_DIR="${SCRIPT_DIR}/calib"
NSAMPLES=128
SEQLEN=4096
BACKEND="cudnn"

# Model paths
QWEN15_DIR="${MODEL_DIR}/Qwen1.5-MoE-A2.7B"
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
    esac
done

# =============================================================================
# STEP 0: Environment check
# =============================================================================
echo ""
echo "============================================================"
echo " NVFP4 Sensitivity Analysis Pipeline"
echo "============================================================"

log "Installing dependencies..."
pip install -r "${SCRIPT_DIR}/requirements_nvfp4.txt" --quiet 2>&1 | tail -3 || true

log "Checking environment..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'  SM:      {cap[0]}{cap[1]}')
" || err "PyTorch not available"

python -c "from flashinfer import mm_fp4; print('  FlashInfer: OK')" 2>/dev/null \
    || warn "FlashInfer not installed — run: pip install flashinfer-python>=0.6.5"

python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" \
    || err "transformers not installed"

python -c "from safetensors.torch import load_file; print('  Safetensors: OK')" \
    || err "safetensors not installed"

python -c "from datasets import load_dataset; print('  Datasets: OK')" \
    || err "datasets not installed"

echo ""

# =============================================================================
# STEP 1: Download models
# =============================================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "============================================================"
    echo " Step 1: Download models"
    echo "============================================================"

    python "${SCRIPT_DIR}/downloader.py" --base_dir "${MODEL_DIR}"

    log "Downloads complete"
else
    warn "Skipping download (--skip-download)"
fi

# =============================================================================
# STEP 2: Verify downloads
# =============================================================================
echo ""
echo "============================================================"
echo " Step 2: Verify model downloads"
echo "============================================================"

verify_model() {
    local model_dir=$1
    local model_name=$2

    if [ ! -d "$model_dir" ]; then
        err "Model directory not found: ${model_dir}"
    fi

    if [ ! -f "${model_dir}/config.json" ]; then
        err "${model_name}: config.json missing"
    fi

    # Check safetensors exist
    local n_safetensors
    n_safetensors=$(find "$model_dir" -name "*.safetensors" | wc -l)
    if [ "$n_safetensors" -eq 0 ]; then
        err "${model_name}: no .safetensors files found"
    fi

    # Verify with Python
    python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('${model_dir}', trust_remote_code=True)
print(f'  ${model_name}:')
print(f'    model_type:    {config.model_type}')
print(f'    num_layers:    {config.num_hidden_layers}')
print(f'    hidden_size:   {config.hidden_size}')
print(f'    num_experts:   {getattr(config, \"num_experts\", \"N/A\")}')
print(f'    safetensors:   ${n_safetensors} files')
" || err "Failed to load ${model_name} config"

    log "${model_name} verified"
}

verify_model "${QWEN15_DIR}" "Qwen1.5-MoE-A2.7B"
verify_model "${QWEN3_DIR}"  "Qwen3-30B-A3B"

# =============================================================================
# STEP 3: Sanity inference test
# =============================================================================
echo ""
echo "============================================================"
echo " Step 3: Sanity inference test"
echo "============================================================"

sanity_test() {
    local model_dir=$1
    local model_name=$2

    log "Testing ${model_name} inference..."
    python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '${model_dir}'
print(f'  Loading {model_path}...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    attn_implementation='sdpa',
)
model.eval()

input_ids = tokenizer('Hello, world!', return_tensors='pt').input_ids.to(model.device)
with torch.no_grad():
    out = model(input_ids)
    logits = out.logits
    print(f'  Output logits shape: {logits.shape}')
    print(f'  First token logit mean: {logits[0,0,:].float().mean().item():.4f}')
    print(f'  Inference OK!')

del model
torch.cuda.empty_cache()
" || err "Inference test FAILED for ${model_name}"
    log "${model_name} inference passed"
}

sanity_test "${QWEN15_DIR}" "Qwen1.5-MoE-A2.7B"
sanity_test "${QWEN3_DIR}"  "Qwen3-30B-A3B"

# =============================================================================
# STEP 4: Run NVFP4 sensitivity analysis
# =============================================================================
echo ""
echo "============================================================"
echo " Step 4: NVFP4 Sensitivity Analysis"
echo "============================================================"

mkdir -p "${CALIB_DIR}"

run_sensitivity() {
    local model_dir=$1
    local model_name=$2
    local metric=$3
    local save_name="${model_name}-nvfp4-${metric}"
    local save_path="${CALIB_DIR}/${save_name}.json"

    if [ -f "$save_path" ]; then
        warn "SKIP: ${save_path} already exists"
        return
    fi

    log "Running: ${model_name} metric=${metric} nsamples=${NSAMPLES}"
    echo "  Save: ${save_path}"

    python "${SCRIPT_DIR}/nf4_sensitivity.py" \
        --model_path "${model_dir}" \
        --metric "${metric}" \
        --nsamples "${NSAMPLES}" \
        --seqlen "${SEQLEN}" \
        --backend "${BACKEND}" \
        --save_path "${save_path}"

    if [ $? -eq 0 ] && [ -f "$save_path" ]; then
        local n_layers
        n_layers=$(python -c "import json; d=json.load(open('${save_path}')); print(len(d))")
        log "Done: ${save_name} (${n_layers} layers saved)"
    else
        err "FAILED: ${save_name}"
    fi
}

# # ── Qwen1.5-MoE-A2.7B (14.3B, 24 layers, 60 experts + 1 shared) ────────────
# echo ""
# echo "--- Qwen1.5-MoE-A2.7B ---"
# run_sensitivity "${QWEN15_DIR}" "qwen15-moe-a2.7b" "layer_out_norm"

# ── Qwen3-30B-A3B (30B, 48 layers, 128 experts, top-8) ──────────────────────
echo ""
echo "--- Qwen3-30B-A3B ---"
run_sensitivity "${QWEN3_DIR}" "qwen3-30b-a3b" "layer_out_norm"

# =============================================================================
# STEP 5: Summary
# =============================================================================
echo ""
echo "============================================================"
echo " Results Summary"
echo "============================================================"

for f in "${CALIB_DIR}"/*.json; do
    if [ -f "$f" ]; then
        n_layers=$(python -c "import json; d=json.load(open('$f')); print(len(d))" 2>/dev/null || echo "?")
        size=$(du -h "$f" | cut -f1)
        echo "  ${f}  (${n_layers} layers, ${size})"
    fi
done

echo ""
log "All experiments complete!"
