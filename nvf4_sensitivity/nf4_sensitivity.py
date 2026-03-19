"""
NVFP4 (FP4 E2M1) Quantization Sensitivity Analysis for MoE Models.

Uses FlashInfer's native FP4 tensor core GEMM (mm_fp4) on Blackwell GPUs
for real hardware-matched sensitivity measurement — NOT fake quantization.

Both weights AND activations are quantized to NVFP4 (W4A4).

Memory-efficient: loads one transformer layer at a time from safetensors.
Never holds the full model in memory.

Usage:
    python nf4_sensitivity.py \\
        --model_path /path/to/Qwen1.5-MoE-A2.7B \\
        --nsamples 128 \\
        --save_path ./calib/nvfp4_sensitivity.json

Output JSON format (same as MxMoE calibration):
    {layer_idx: {expert_idx: [gate_err, up_err, down_err]}}

Dependencies:
    pip install torch transformers safetensors flashinfer-python>=0.6.5 datasets tqdm

Requires: Blackwell GPU (SM120/SM100) with CUDA 12.8+ and cuDNN 9.x
"""

import os
import gc
import json
import time
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import Tensor
from safetensors.torch import load_file as safetensors_load
from flashinfer import nvfp4_quantize, mm_fp4, SfLayout


# ============================================================================
# NVFP4 Quantization + Native FP4 GEMM
# ============================================================================

NVFP4_BLOCK_SIZE = 16  # NVFP4 spec: 16 elements per FP8-scaled block


def compute_nvfp4_global_scale(tensor: Tensor) -> Tensor:
    """
    Compute NVFP4 global scale factor for a tensor.

    Formula: (max_fp8_e4m3 * max_fp4_e2m1) / max(|tensor|)
             = (448 * 6) / max(|tensor|)

    This maps the tensor's full dynamic range into the representable
    range of the two-level NVFP4 scaling scheme:
        value ≈ global_scale * block_scale_fp8 * fp4_value
    """
    absmax = tensor.float().abs().nan_to_num().max()
    return (448.0 * 6.0) / absmax.clamp(min=1e-12)


@torch.no_grad()
def make_nvfp4_forward(
    linear: nn.Linear,
    backend: str = "cudnn",
) -> tuple:
    """
    Create an NVFP4 replacement for nn.Linear.forward().

    Pre-quantizes the weight to NVFP4. Returns (nvfp4_forward_fn, cleanup_data).
    The forward_fn quantizes activations on-the-fly and uses mm_fp4 for
    native FP4×FP4 → BF16 tensor core GEMM.

    Args:
        linear: the nn.Linear module to replace
        backend: FlashInfer mm_fp4 backend ('cudnn', 'cutlass', 'auto')

    Returns:
        (nvfp4_forward_fn, w_global_sf)
    """
    w = linear.weight.data  # [out_features, in_features]
    bias = linear.bias

    # Pre-quantize weight to NVFP4
    w_global_sf = compute_nvfp4_global_scale(w)
    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    # mm_fp4 expects B in column-major: pass w_fp4.T
    w_fp4_t = w_fp4.T.contiguous()
    w_sf_t = w_sf.T.contiguous()

    def nvfp4_forward(input: Tensor) -> Tensor:
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])  # [M, K]

        # Quantize activation to NVFP4 on-the-fly
        x_global_sf = compute_nvfp4_global_scale(x)
        x_fp4, x_sf = nvfp4_quantize(
            x, x_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )

        # Combined dequantization scale
        alpha = torch.tensor(
            1.0 / (x_global_sf.item() * w_global_sf.item()),
            device=x.device, dtype=torch.float32,
        )

        # Native FP4 × FP4 → BF16 tensor core GEMM
        out = mm_fp4(
            x_fp4, w_fp4_t,
            x_sf, w_sf_t,
            alpha,
            out_dtype=torch.bfloat16,
            block_size=NVFP4_BLOCK_SIZE,
            backend=backend,
            use_nvfp4=True,
        )

        if bias is not None:
            out = out + bias

        return out.reshape(*orig_shape[:-1], out.shape[-1])

    return nvfp4_forward, w_global_sf


# ============================================================================
# Memory-efficient layer loader from safetensors
# ============================================================================

class SafetensorsLayerLoader:
    """Load model weights layer-by-layer from safetensors. Uses mmap."""

    def __init__(self, model_path: str):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map: dict[str, str] = json.load(f)["weight_map"]
            self.sharded = True
        else:
            self.weight_map = None
            self.single_file = os.path.join(model_path, "model.safetensors")
            self.sharded = False
        self.model_path = model_path

    def _load_by_prefix(self, prefix: str, device: str = "cpu") -> dict[str, Tensor]:
        if not self.sharded:
            shard = safetensors_load(self.single_file, device=device)
            result = {k[len(prefix):]: v.clone() for k, v in shard.items() if k.startswith(prefix)}
            del shard
            return result

        matching = {k: v for k, v in self.weight_map.items() if k.startswith(prefix)}
        result = {}
        for shard_file in set(matching.values()):
            shard = safetensors_load(os.path.join(self.model_path, shard_file), device=device)
            for full_name, sf in matching.items():
                if sf == shard_file and full_name in shard:
                    result[full_name[len(prefix):]] = shard[full_name].clone()
            del shard
        return result

    def get_embedding_weight(self, device="cpu") -> Tensor:
        return self._load_by_prefix("model.embed_tokens.", device)["weight"]

    def get_layer_state_dict(self, layer_idx: int, device="cpu") -> dict[str, Tensor]:
        return self._load_by_prefix(f"model.layers.{layer_idx}.", device)


# ============================================================================
# Calibration data
# ============================================================================

def get_calibration_data(tokenizer_path: str, nsamples: int, seed: int, seqlen: int) -> list[Tensor]:
    """Load WikiText-2 calibration data."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        trainloader.append(trainenc.input_ids[:, i:i + seqlen])
    return trainloader


# ============================================================================
# Model architecture helpers
# ============================================================================

# Maps model_type → (expert list accessor, linear block names, labels)
MODEL_EXPERT_CONFIG = {
    "qwen2_moe": {
        "block_names": ["gate_proj", "up_proj", "down_proj"],
        "labels": ["gate", "up", "down"],
    },
    "qwen3_moe": {
        "block_names": ["gate_proj", "up_proj", "down_proj"],
        "labels": ["gate", "up", "down"],
    },
    "deepseek_v2": {
        "block_names": ["gate_proj", "up_proj", "down_proj"],
        "labels": ["gate", "up", "down"],
    },
    "mixtral": {
        "block_names": ["w1", "w3", "w2"],
        "labels": ["gate", "up", "down"],
    },
}


def get_experts(layer: nn.Module, model_type: str) -> list[nn.Module]:
    """Get all expert modules from a layer (routed + shared)."""
    experts = []
    if model_type in ("qwen2_moe", "qwen3_moe"):
        experts.extend(layer.mlp.experts)
        if hasattr(layer.mlp, "shared_expert") and layer.mlp.shared_expert is not None:
            experts.append(layer.mlp.shared_expert)
    elif model_type == "deepseek_v2":
        if hasattr(layer.mlp, "experts"):
            experts.extend(layer.mlp.experts)
            if hasattr(layer.mlp, "shared_experts"):
                experts.append(layer.mlp.shared_experts)
        else:
            experts.append(layer.mlp)
    elif model_type == "mixtral":
        experts.extend(layer.block_sparse_moe.experts)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return experts


def get_decoder_layer_class(model_type: str):
    """Import and return the correct DecoderLayer class for a model type."""
    if model_type == "qwen2_moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
            Qwen2MoeDecoderLayer, Qwen2MoeRotaryEmbedding,
        )
        return Qwen2MoeDecoderLayer, Qwen2MoeRotaryEmbedding
    elif model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding,
        )
        return Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
    elif model_type == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralDecoderLayer, MixtralRotaryEmbedding,
        )
        return MixtralDecoderLayer, MixtralRotaryEmbedding
    elif model_type == "deepseek_v2":
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2DecoderLayer, DeepseekV2RotaryEmbedding,
        )
        return DeepseekV2DecoderLayer, DeepseekV2RotaryEmbedding
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ============================================================================
# NVFP4 Sensitivity Analysis
# ============================================================================

@torch.no_grad()
def get_model_quant_error(
    model_path: str,
    dataloader: list[Tensor],
    metric: str,  # "layer_out_norm" or "model_out_norm"
    save_path: str,
    max_layers: int = -1,
    backend: str = "cudnn",
):
    """
    NVFP4 quantization sensitivity analysis — follows the exact same structure
    as MoeModelQuantizer.get_model_quant_error() in quant.py.

    For each layer × expert × linear block (gate/up/down):
      1. Compute full-precision output (BF16)
      2. Patch that ONE linear block with NVFP4 (mm_fp4)
      3. Compute NVFP4-quantized output
      4. Error = ‖quantized_output - fp_output‖₂
      5. Unpatch, restore original forward

    Two metrics:
      "layer_out_norm": error at single-layer output (one layer on GPU at a time)
      "model_out_norm": error at full-model last hidden state (needs full model in memory)
    """
    from transformers import AutoConfig
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model_type = config.model_type
    num_layers = config.num_hidden_layers if max_layers < 0 else min(max_layers, config.num_hidden_layers)
    hidden_size = config.hidden_size
    expert_cfg = MODEL_EXPERT_CONFIG[model_type]
    block_names = expert_cfg["block_names"]
    block_labels = expert_cfg["labels"]

    DecoderLayerCls, RotaryEmbCls = get_decoder_layer_class(model_type)
    loader = SafetensorsLayerLoader(model_path)
    num_samples = len(dataloader)

    print(f"NVFP4 Sensitivity Analysis (metric={metric})")
    print(f"  Model: {model_path} ({model_type})")
    print(f"  Layers: {num_layers}, Hidden: {hidden_size}")
    print(f"  Backend: {backend}, Samples: {num_samples}")

    # allocate for quant error (same structure as quant.py)
    layer_loss: list[list[list[float]]] = [[] for _ in range(num_layers)]
    layer_loss_save = {}

    ################# METRIC 1: LAYER-LOCAL SENSITIVITY (layer_out_norm) #################
    # Matches quant.py get_model_quant_error() layer_out_norm path exactly.
    # One layer on GPU at a time. For each layer:
    #   Step 1: compute FP output for all nsamples
    #   Step 2: for each expert × block, NVFP4-patch → recompute → measure ‖diff‖₂ → unpatch
    #   Step 3: propagate FP outputs as inputs to next layer
    #####################################################################################
    if metric == "layer_out_norm":

        # ── Prepare inputs (equivalent to prepare_inps Catcher trick) ────
        # Load embedding + rotary_emb from safetensors, compute hidden states
        embed_weight = loader.get_embedding_weight(device="cpu")
        embed = nn.Embedding(config.vocab_size, hidden_size, getattr(config, "pad_token_id", None))
        embed.weight = nn.Parameter(embed_weight.to(dtype))
        embed = embed.to(dev)

        inps = torch.zeros(num_samples, dataloader[0].shape[-1], hidden_size, dtype=dtype, device=dev)
        for i, batch in enumerate(dataloader):
            inps[i] = embed(batch.to(dev)).squeeze(0)
        del embed, embed_weight
        gc.collect(); torch.cuda.empty_cache()

        seqlen = inps.shape[1]
        rotary_emb = RotaryEmbCls(config=config).to(dev)
        position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)
        pos_emb = rotary_emb(inps[0:1], position_ids)
        del rotary_emb; torch.cuda.empty_cache()

        # SDPA handles causal masking internally when attention_mask=None
        attn_mask = None

        full_precision_outs: Tensor = torch.zeros_like(inps).to(torch.float64)
        quantized_outs: Tensor = torch.zeros_like(inps).to(torch.float64)

        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            # Load layer from safetensors → GPU
            layer_sd = loader.get_layer_state_dict(layer_idx, device="cpu")
            layer = DecoderLayerCls(config, layer_idx)
            layer.load_state_dict(layer_sd, strict=False)
            layer = layer.to(dtype).to(dev).eval()
            del layer_sd; gc.collect(); torch.cuda.empty_cache()

            with torch.inference_mode():
                # 1. get the output of the full precision layer
                for i in range(num_samples):
                    full_precision_outs[i] = layer(
                        inps[i].unsqueeze(0),
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        position_embeddings=pos_emb,
                    )[0].to(torch.float64)

            # 2. get the output of the NVFP4-quantized layer
            experts = get_experts(layer, model_type)
            num_layer_experts = len(experts)

            for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                expert = experts[exp_id]
                expert_err = []

                for qlinear_block, block_name in zip(block_labels, block_names):
                    linear: nn.Linear = getattr(expert, block_name, None)
                    if linear is None or not hasattr(linear, "weight"):
                        expert_err.append(0.0)
                        continue

                    with torch.inference_mode():
                        # Patch: replace this ONE linear with NVFP4 compute
                        orig_forward = linear.forward
                        nvfp4_fwd, _ = make_nvfp4_forward(linear, backend=backend)
                        linear.forward = nvfp4_fwd

                        for i in range(num_samples):
                            quantized_outs[i] = layer(
                                inps[i].unsqueeze(0),
                                attention_mask=attn_mask,
                                position_ids=position_ids,
                                position_embeddings=pos_emb,
                            )[0].to(torch.float64)

                        # 3. calculate the quantization error
                        quant_err = torch.norm(quantized_outs - full_precision_outs).item()
                        expert_err.append(quant_err)

                        # 4. recover the FULL precision layer (unpatch)
                        linear.forward = orig_forward

                layer_loss[layer_idx].append(expert_err)
                print(f"  L{layer_idx}-E{exp_id} NVFP4 (layer_out_norm): {expert_err}")

            # Unload layer
            del layer, experts
            gc.collect(); torch.cuda.empty_cache()

            # 5. serialization (incremental, same format as quant.py)
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            with open(save_path, "w") as f:
                layer_loss_save[layer_idx] = {e: layer_loss[layer_idx][e] for e in range(len(layer_loss[layer_idx]))}
                json.dump(layer_loss_save, f)
            print(f"  Layer-{layer_idx} quant error(layer_out_norm):\n{layer_loss[layer_idx]}")

            # 6. prepare the inputs for next layer
            inps, full_precision_outs = full_precision_outs.to(inps.dtype), full_precision_outs

    ################# METRIC 2: FULL-MODEL SENSITIVITY (model_out_norm) #################
    # Matches quant.py get_model_quant_error() model_out_norm path.
    # Measures error at the model's LAST HIDDEN STATE. For each perturbation,
    # runs a full forward pass through ALL layers.
    # NOTE: Requires enough memory to hold the full model (loads all layers).
    ####################################################################################
    elif metric == "model_out_norm":
        from transformers import AutoModelForCausalLM

        # Load full model (needs enough RAM)
        print("Loading full model for model_out_norm (requires full model in memory)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.config.use_cache = False
        model.eval()

        layers = model.model.layers

        # inps: [num_samples, seqlen]
        inps = torch.vstack([d.to(model.device) for d in dataloader])

        # outs: [num_samples, seqlen, hidden_size]
        full_precision_outs: Tensor = torch.zeros(
            (num_samples, inps.shape[-1], hidden_size),
            device=next(layers[-1].parameters()).device, dtype=torch.float64,
        )
        quantized_outs: Tensor = torch.zeros_like(full_precision_outs)

        # 1. get the output of full-precision model (hidden_states of the last layer)
        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc="Getting full precision output"):
                full_precision_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

        # 2. get the model output with NVFP4-quantized layer
        for layer_idx in tqdm(range(num_layers), desc="Getting quantized output"):
            experts = get_experts(layers[layer_idx], model_type)
            num_layer_experts = len(experts)

            for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                expert = experts[exp_id]
                expert_err = []

                for qlinear_block, block_name in zip(block_labels, block_names):
                    linear: nn.Linear = getattr(expert, block_name, None)
                    if linear is None or not hasattr(linear, "weight"):
                        expert_err.append(0.0)
                        continue

                    # Patch: replace this ONE linear with NVFP4 compute
                    orig_forward = linear.forward
                    nvfp4_fwd, _ = make_nvfp4_forward(linear, backend=backend)
                    linear.forward = nvfp4_fwd

                    with torch.inference_mode():
                        for i in range(num_samples):
                            quantized_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

                        # 3. calculate the quantization error
                        quant_err = torch.norm(quantized_outs.sub_(full_precision_outs)).item()
                        expert_err.append(quant_err)

                        # 4. recover the FULL precision layer (unpatch)
                        linear.forward = orig_forward

                layer_loss[layer_idx].append(expert_err)
                print(f"  L{layer_idx}-E{exp_id} NVFP4 (model_out_norm): {expert_err}")

            # 5. serialization
            with open(save_path, "w") as f:
                layer_loss_save[layer_idx] = {e: layer_loss[layer_idx][e] for e in range(len(layer_loss[layer_idx]))}
                json.dump(layer_loss_save, f)
            print(f"  Layer-{layer_idx} quant error(model_out_norm): {layer_loss[layer_idx]}")

        del model

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return layer_loss


# ============================================================================
# Seed
# ============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NVFP4 (W4A4) Quantization Sensitivity Analysis for MoE Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # layer_out_norm (memory-efficient, one layer at a time)
  python nf4_sensitivity.py --model_path /path/to/model --metric layer_out_norm

  # model_out_norm (needs full model in memory)
  python nf4_sensitivity.py --model_path /path/to/model --metric model_out_norm

  # Quick test (1 layer)
  python nf4_sensitivity.py --model_path /path/to/model --metric layer_out_norm --max_layers 1
        """,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--metric", type=str, default="layer_out_norm",
                        choices=["layer_out_norm", "model_out_norm"],
                        help="Sensitivity metric (default: layer_out_norm)")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--seqlen", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save_path", type=str, default="./calib/nvfp4_sensitivity.json",
                        help="Output JSON path")
    parser.add_argument("--max_layers", type=int, default=-1,
                        help="Max layers to process (-1 = all)")
    parser.add_argument("--backend", type=str, default="cudnn",
                        choices=["cudnn", "cutlass", "auto"],
                        help="FlashInfer mm_fp4 backend (default: cudnn — most stable on SM120)")

    args = parser.parse_args()
    seed_everything(args.seed)

    dataloader = get_calibration_data(args.model_path, args.nsamples, args.seed, args.seqlen)

    get_model_quant_error(
        model_path=args.model_path,
        dataloader=dataloader,
        metric=args.metric,
        save_path=args.save_path,
        max_layers=args.max_layers,
        backend=args.backend,
    )
