"""
NVFP4 (FP4 E2M1) Quantization Sensitivity Analysis for MoE Models.

Uses FlashInfer's native FP4 tensor core GEMM (mm_fp4) on Blackwell GPUs
for real hardware-matched sensitivity measurement — NOT fake quantization.

Both weights AND activations are quantized to NVFP4 (W4A4).

Loads the full model via from_pretrained (requires sufficient VRAM).

Usage:
    python nf4_sensitivity.py \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --nsamples 128 \\
        --save_path ./calib/nvfp4_sensitivity.json

Output JSON format (same as MxMoE calibration):
    {layer_idx: {expert_idx: [gate_err, up_err, down_err]}}

Dependencies:
    pip install torch transformers flashinfer-python>=0.6.5 datasets tqdm

Requires: Blackwell GPU (SM120/SM100) with CUDA 12.8+ and cuDNN 9.x
"""

import os
import gc
import json
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import Tensor


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


def make_nvfp4_forward(
    linear: nn.Linear,
    backend: str = "cudnn",
) -> tuple:
    """Create an NVFP4 (W4A4) replacement for nn.Linear.forward().

    Weight and activation are quantized to NVFP4 on every call and
    immediately discarded — no pre-computed tensors stored in the closure.

    Requires: flashinfer-python >= 0.6.5, Blackwell GPU (SM120/SM100).
    """
    from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
    effective_backend = [backend]

    def nvfp4_forward(input: Tensor) -> Tensor:
        w = linear.weight.data
        bias = linear.bias
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])

        # Quantize weight
        w_global_sf = compute_nvfp4_global_scale(w)
        w_fp4, w_sf = nvfp4_quantize(
            w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
        )
        w_fp4_t = w_fp4.T.contiguous()
        w_sf_t = w_sf.T.contiguous()

        # Quantize activation
        x_global_sf = compute_nvfp4_global_scale(x)
        x_fp4, x_sf = nvfp4_quantize(
            x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
        )

        alpha = torch.tensor(
            1.0 / (x_global_sf.item() * w_global_sf.item()),
            device=x.device, dtype=torch.float32,
        )

        try:
            out = mm_fp4(
                x_fp4, w_fp4_t, x_sf, w_sf_t, alpha,
                out_dtype=torch.bfloat16, block_size=NVFP4_BLOCK_SIZE,
                backend=effective_backend[0], use_nvfp4=True,
            )
        except RuntimeError:
            if effective_backend[0] != "cutlass":
                effective_backend[0] = "cutlass"
            out = mm_fp4(
                x_fp4, w_fp4_t, x_sf, w_sf_t, alpha,
                out_dtype=torch.bfloat16, block_size=NVFP4_BLOCK_SIZE,
                backend="cutlass", use_nvfp4=True,
            )

        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    return nvfp4_forward, None


# ============================================================================
# FP8 (W8A8) — Real FP8 Tensor Core GEMM via FlashInfer
# ============================================================================

def make_fp8_forward(linear: nn.Linear) -> tuple:
    """Create a W8A8 FP8 replacement using FlashInfer gemm_fp8_nt_groupwise.

    Activation: per-token groupwise FP8 (flashinfer.testing.utils.per_token_cast_to_fp8).
    Weight: per-block FP8 128×128 (flashinfer.testing.utils.per_block_cast_to_fp8).
    GEMM: real FP8 tensor core compute via flashinfer.gemm.gemm_fp8_nt_groupwise.
    All quantized tensors computed on-the-fly and discarded.
    """
    from flashinfer.gemm import gemm_fp8_nt_groupwise
    from flashinfer.testing.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

    def fp8_forward(input: Tensor) -> Tensor:
        w = linear.weight.data     # [N, K]
        bias = linear.bias
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])  # [M, K]
        M, K = x.shape
        N = w.shape[0]

        # ── Quantize activation: per-token, 128-element groups along K ──
        x_fp8, x_sf = per_token_cast_to_fp8(x)    # x_fp8: [M,K], x_sf: [M, K//128]
        a_scale = x_sf.T.contiguous()              # [K//128, M] for "MN" mode

        # ── Quantize weight: 128×128 block-wise ─────────────────────────
        w_fp8, w_sf = per_block_cast_to_fp8(w)     # w_fp8: [N,K], w_sf: [N//128, K//128]
        b_scale = w_sf.T.contiguous()              # [K//128, N//128] for "MN" mode

        # ── Real FP8 tensor core GEMM ───────────────────────────────────
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        gemm_fp8_nt_groupwise(
            x_fp8, w_fp8, a_scale, b_scale,
            out=out, scale_major_mode="MN",
        )

        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    return fp8_forward, None


# ============================================================================
# A16W4 (Weight-only FP4) Fake Quantization
# ============================================================================

# FP4 E2M1 representable magnitudes and their midpoint breakpoints
_FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_BREAKS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def fake_quantize_fp4_e2m1(tensor: Tensor, block_size: int = 16) -> Tensor:
    """Fake-quantize to FP4 E2M1 with per-block scaling (matches NVFP4 scheme)."""
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, block_size)

    # Per-block scale: max(|block|) / max_fp4 (6.0)
    amax = flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
    scale = amax / 6.0

    # Scale into FP4 range, split sign
    scaled = flat / scale
    sign = scaled.sign()
    abs_scaled = scaled.abs()

    # Bucket into nearest FP4 value
    breaks = torch.tensor(_FP4_BREAKS, device=tensor.device, dtype=tensor.dtype)
    vals = torch.tensor(_FP4_VALUES, device=tensor.device, dtype=tensor.dtype)
    idx = torch.bucketize(abs_scaled, breaks)
    quantized_abs = vals[idx]

    return (sign * quantized_abs * scale).reshape(orig_shape)


def make_a16w4_forward(linear: nn.Linear) -> tuple:
    """Create a W4A16 replacement for nn.Linear.forward().

    Weight is fake-quantized to FP4 E2M1 on every call.
    Activation stays in BF16. No pre-computed tensors stored.
    """
    def a16w4_forward(input: Tensor) -> Tensor:
        w_fq = fake_quantize_fp4_e2m1(linear.weight.data)
        out = torch.nn.functional.linear(input, w_fq, linear.bias)
        return out

    return a16w4_forward, None


# ============================================================================
# Dispatch
# ============================================================================

def make_quant_forward(linear: nn.Linear, quant_mode: str, backend: str = "cudnn") -> tuple:
    """Create a quantized forward function for the given mode.

    Returns (forward_fn, extra_data).
    """
    if quant_mode == "nvfp4":
        return make_nvfp4_forward(linear, backend=backend)
    elif quant_mode == "fp8":
        return make_fp8_forward(linear)
    elif quant_mode == "a16w4":
        return make_a16w4_forward(linear)
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")


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

# Maps model_type → (linear block names, labels)
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


class ExpertMLP(nn.Module):
    """Individual expert module created from slices of fused 3D weight tensors."""

    def __init__(self, gate_weight: Tensor, up_weight: Tensor,
                 down_weight: Tensor, act_fn: nn.Module):
        super().__init__()
        inter_dim, hidden_dim = gate_weight.shape
        self.gate_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_dim, bias=False)
        # Point to slices of the fused tensor (views — no extra memory)
        self.gate_proj.weight = nn.Parameter(gate_weight, requires_grad=False)
        self.up_proj.weight = nn.Parameter(up_weight, requires_grad=False)
        self.down_proj.weight = nn.Parameter(down_weight, requires_grad=False)
        self.act_fn = act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def unfuse_moe_experts(moe_block: nn.Module) -> None:
    """Replace fused Qwen3MoeExperts (3D weight tensors) with individual ExpertMLP modules.

    After this, moe_block.experts is nn.ModuleList[ExpertMLP] and the forward
    routes through individual expert modules, so monkey-patching nn.Linear.forward
    on a single expert's projection actually affects the layer computation.

    Uses views of the original fused tensors — no extra GPU memory.
    """
    fused = moe_block.experts

    # Already unfused (nn.ModuleList) — nothing to do
    if isinstance(fused, nn.ModuleList):
        return

    # Not a fused format we recognise
    if not hasattr(fused, "gate_up_proj"):
        return

    num_experts = fused.num_experts
    inter_dim = fused.intermediate_dim
    act_fn = fused.act_fn
    gate = moe_block.gate

    # Build individual expert modules from slices of fused 3D params
    experts_list = nn.ModuleList()
    for i in range(num_experts):
        gate_up = fused.gate_up_proj[i]            # [2*inter, hidden] (view)
        expert = ExpertMLP(
            gate_weight=gate_up[:inter_dim],        # [inter, hidden]   (view)
            up_weight=gate_up[inter_dim:],           # [inter, hidden]   (view)
            down_weight=fused.down_proj[i],          # [hidden, inter]   (view)
            act_fn=act_fn,
        )
        experts_list.append(expert)

    # Move to same device as original weights
    dev = fused.gate_up_proj.device
    experts_list = experts_list.to(dev)

    # Replace the fused experts with our ModuleList
    moe_block.experts = experts_list

    # Replace forward to route through individual expert modules
    def unfused_forward(hidden_states: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = gate(flat)

        final = torch.zeros_like(flat)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=num_experts,
        ).permute(2, 1, 0)

        with torch.no_grad():
            expert_hit = expert_mask.sum(dim=(-1, -2)).nonzero().squeeze(-1)

        for idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[idx])
            if token_idx.numel() == 0:
                continue
            out = experts_list[idx](flat[token_idx])
            out = out * routing_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        return final.reshape(batch_size, seq_len, hidden_dim)

    moe_block.forward = unfused_forward


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
    max_experts: int = -1,
    backend: str = "cudnn",
    quant_mode: str = "nvfp4",
    joint: bool = False,
    resume: bool = False,
):
    """
    NVFP4 quantization sensitivity analysis.

    For each layer × expert × linear block (gate/up/down):
      1. Compute full-precision output (BF16)
      2. Patch that ONE linear block with NVFP4 (mm_fp4)
      3. Compute NVFP4-quantized output
      4. Error = ‖quantized_output - fp_output‖₂
      5. Unpatch, restore original forward

    Two metrics:
      "layer_out_norm": error at single-layer output
      "model_out_norm": error at full-model last hidden state
    """
    from transformers import AutoModelForCausalLM

    dev = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ── Load full model on GPU ───────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.eval()

    config = model.config
    model_type = config.model_type
    num_layers = config.num_hidden_layers if max_layers < 0 else min(max_layers, config.num_hidden_layers)
    hidden_size = config.hidden_size
    num_experts = getattr(config, "num_experts", getattr(config, "num_local_experts", 1))
    expert_cfg = MODEL_EXPERT_CONFIG[model_type]
    block_names = expert_cfg["block_names"]
    block_labels = expert_cfg["labels"]
    num_samples = len(dataloader)
    layers = model.model.layers

    print(f"Quantization Sensitivity Analysis (metric={metric}, quant={quant_mode})")
    print(f"  Model: {model_path} ({model_type})")
    print(f"  Layers: {num_layers}, Hidden: {hidden_size}, Experts: {num_experts}")
    print(f"  Backend: {backend}, Samples: {num_samples}")
    print(f"  Quant mode: {quant_mode}")

    # ── Unfuse MoE experts (3D fused tensors → individual nn.Module) ──
    # Newer transformers stores expert weights as fused 3D nn.Parameter
    # (Qwen3MoeExperts). We need individual ExpertMLP modules so that
    # monkey-patching a single nn.Linear.forward actually affects the
    # layer computation during sensitivity measurement.
    for layer in layers:
        if hasattr(layer.mlp, "experts"):
            unfuse_moe_experts(layer.mlp)
    print("  Experts unfused: individual modules ready for patching")

    layer_loss: list[list[list[float]]] = [[] for _ in range(num_layers)]
    layer_loss_save = {}

    # ── Resume: load existing results from save_path ─────────────────
    if resume and os.path.exists(save_path):
        with open(save_path) as f:
            layer_loss_save = json.load(f)
        # Convert string keys back to int
        layer_loss_save = {int(k): v for k, v in layer_loss_save.items()}
        n_existing = sum(len(v) for v in layer_loss_save.values())
        print(f"  Resume: loaded {n_existing} existing expert results from {save_path}")

    ################# METRIC 1: LAYER-LOCAL SENSITIVITY (layer_out_norm) #################
    if metric == "layer_out_norm":

        # ── Prepare inputs using model's own embedding ───────────────────
        embed = model.model.embed_tokens
        inps = torch.zeros(num_samples, dataloader[0].shape[-1], hidden_size, dtype=dtype, device=dev)
        for i, batch in enumerate(dataloader):
            inps[i] = embed(batch.to(dev)).squeeze(0)

        seqlen = inps.shape[1]
        rotary_emb = model.model.rotary_emb
        position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)
        pos_emb = rotary_emb(inps[0:1], position_ids)

        attn_mask = None  # SDPA handles causal masking internally
        full_precision_outs: Tensor = torch.zeros_like(inps).to(torch.float64)
        quantized_outs: Tensor = torch.zeros_like(inps).to(torch.float64)

        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            layer = layers[layer_idx]

            # ── Capture per-expert routing distribution ──────────────────
            expert_token_counts = torch.zeros(num_experts, dtype=torch.long, device=dev)

            def _routing_hook(module, input, output,
                              _counts=expert_token_counts):
                # output layout: (router_logits, router_scores, router_indices)
                # router_indices: [num_tokens, top_k]
                indices = output[2].flatten()
                _counts.index_add_(
                    0, indices,
                    torch.ones_like(indices, dtype=torch.long),
                )

            hook_handle = None
            if hasattr(layer.mlp, "gate"):
                hook_handle = layer.mlp.gate.register_forward_hook(_routing_hook)

            with torch.inference_mode():
                # 1. get the output of the full precision layer
                for i in range(num_samples):
                    full_precision_outs[i] = layer(
                        inps[i].unsqueeze(0),
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        position_embeddings=pos_emb,
                    )[0].to(torch.float64)

            if hook_handle is not None:
                hook_handle.remove()

            # Print routing stats
            total_assignments = expert_token_counts.sum().item()
            total_tokens = num_samples * seqlen
            if total_assignments > 0:
                mean_c = expert_token_counts.float().mean().item()
                std_c = expert_token_counts.float().std().item()
                cv = std_c / mean_c if mean_c > 0 else 0
                print(f"  Layer {layer_idx}: {total_tokens} tokens × top-{total_assignments // total_tokens} "
                      f"= {total_assignments} total assignments over {num_samples} samples")
                print(f"    per-expert tokens: min={expert_token_counts.min().item()} "
                      f"max={expert_token_counts.max().item()} "
                      f"mean={mean_c:.1f} std={std_c:.1f} CV={cv:.3f}")

            # 2. get the output of the quantized layer
            experts = get_experts(layer, model_type)
            num_layer_experts = len(experts) if max_experts < 0 else min(max_experts, len(experts))

            for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                # ── Resume: skip if this layer-expert already exists ─────
                if resume and layer_idx in layer_loss_save:
                    existing = layer_loss_save[layer_idx]
                    if str(exp_id) in existing or exp_id in existing:
                        entry = existing.get(str(exp_id), existing.get(exp_id))
                        cached_err = entry["error"] if isinstance(entry, dict) else entry
                        layer_loss[layer_idx].append(cached_err)
                        exp_tokens = expert_token_counts[exp_id].item() if exp_id < len(expert_token_counts) else 0
                        print(f"  L{layer_idx}-E{exp_id} SKIP (resumed) [{exp_tokens} tokens]: {cached_err}")
                        continue

                expert = experts[exp_id]

                if joint:
                    # ── Joint mode: patch ALL linears at once, single error ───
                    orig_forwards = {}
                    for block_name in block_names:
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            continue
                        orig_forwards[block_name] = linear.forward
                        quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                        linear.forward = quant_fwd

                    with torch.inference_mode():
                        for i in range(num_samples):
                            quantized_outs[i] = layer(
                                inps[i].unsqueeze(0),
                                attention_mask=attn_mask,
                                position_ids=position_ids,
                                position_embeddings=pos_emb,
                            )[0].to(torch.float64)

                    joint_err = torch.norm(quantized_outs - full_precision_outs).item()

                    # Restore all
                    for block_name, orig_fwd in orig_forwards.items():
                        getattr(expert, block_name).forward = orig_fwd

                    expert_err = [joint_err]
                else:
                    # ── Per-linear mode: patch one linear at a time ───────────
                    expert_err = []
                    for qlinear_block, block_name in zip(block_labels, block_names):
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            expert_err.append(0.0)
                            continue

                        with torch.inference_mode():
                            orig_forward = linear.forward
                            quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                            linear.forward = quant_fwd

                            for i in range(num_samples):
                                quantized_outs[i] = layer(
                                    inps[i].unsqueeze(0),
                                    attention_mask=attn_mask,
                                    position_ids=position_ids,
                                    position_embeddings=pos_emb,
                                )[0].to(torch.float64)

                            quant_err = torch.norm(quantized_outs - full_precision_outs).item()
                            expert_err.append(quant_err)
                            linear.forward = orig_forward

                layer_loss[layer_idx].append(expert_err)
                label = "joint" if joint else "per-linear"
                exp_tokens = expert_token_counts[exp_id].item() if exp_id < len(expert_token_counts) else 0
                print(f"  L{layer_idx}-E{exp_id} {quant_mode} ({label}) "
                      f"[{exp_tokens} tokens]: {expert_err}")

            # 5. serialization (incremental)
            mean_tokens = expert_token_counts.float().mean().item() if total_assignments > 0 else 1.0
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            layer_save = {}
            for e in range(len(layer_loss[layer_idx])):
                tok = expert_token_counts[e].item() if e < len(expert_token_counts) else 0
                layer_save[e] = {
                    "error": layer_loss[layer_idx][e],
                    "tokens": tok,
                    "token_ratio": round(tok / mean_tokens, 4) if mean_tokens > 0 else 0.0,
                }
            layer_loss_save[layer_idx] = layer_save
            with open(save_path, "w") as f:
                json.dump(layer_loss_save, f)
            print(f"  Layer-{layer_idx} saved ({len(layer_save)} experts)")

            # 6. prepare the inputs for next layer
            inps = full_precision_outs.to(inps.dtype)

    ################# METRIC 2: FULL-MODEL SENSITIVITY (model_out_norm) #################
    elif metric == "model_out_norm":

        inps = torch.vstack([d.to(dev) for d in dataloader])  # [num_samples, seqlen]

        full_precision_outs: Tensor = torch.zeros(
            (num_samples, inps.shape[-1], hidden_size),
            device=dev, dtype=torch.float64,
        )
        quantized_outs: Tensor = torch.zeros_like(full_precision_outs)

        # ── Capture per-layer routing during full-precision pass ────
        per_layer_counts = {li: torch.zeros(num_experts, dtype=torch.long, device=dev)
                            for li in range(num_layers)}

        def _make_model_routing_hook(layer_idx):
            counts = per_layer_counts[layer_idx]
            def hook(module, input, output, _counts=counts):
                indices = output[2].flatten()
                _counts.index_add_(
                    0, indices,
                    torch.ones_like(indices, dtype=torch.long),
                )
            return hook

        hook_handles = []
        for li in range(num_layers):
            if hasattr(layers[li].mlp, "gate"):
                h = layers[li].mlp.gate.register_forward_hook(_make_model_routing_hook(li))
                hook_handles.append(h)

        # 1. get the output of full-precision model (hidden_states of the last layer)
        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc="Full precision output"):
                full_precision_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

        for h in hook_handles:
            h.remove()

        # Print routing stats per layer
        seqlen_model = inps.shape[-1]
        total_tokens_model = num_samples * seqlen_model
        for li in range(num_layers):
            counts = per_layer_counts[li]
            total = counts.sum().item()
            if total > 0:
                mean_c = counts.float().mean().item()
                std_c = counts.float().std().item()
                cv = std_c / mean_c if mean_c > 0 else 0
                print(f"  Layer {li}: {total_tokens_model} tokens × top-{total // total_tokens_model} "
                      f"= {total} total assignments over {num_samples} samples")
                print(f"    per-expert tokens: min={counts.min().item()} "
                      f"max={counts.max().item()} "
                      f"mean={mean_c:.1f} std={std_c:.1f} CV={cv:.3f}")

        # 2. get the model output with quantized layer
        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            experts = get_experts(layers[layer_idx], model_type)
            num_layer_experts = len(experts) if max_experts < 0 else min(max_experts, len(experts))

            for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                # ── Resume: skip if this layer-expert already exists ─────
                if resume and layer_idx in layer_loss_save:
                    existing = layer_loss_save[layer_idx]
                    if str(exp_id) in existing or exp_id in existing:
                        entry = existing.get(str(exp_id), existing.get(exp_id))
                        cached_err = entry["error"] if isinstance(entry, dict) else entry
                        layer_loss[layer_idx].append(cached_err)
                        exp_tokens = per_layer_counts[layer_idx][exp_id].item() if exp_id < len(per_layer_counts[layer_idx]) else 0
                        print(f"  L{layer_idx}-E{exp_id} SKIP (resumed) [{exp_tokens} tokens]: {cached_err}")
                        continue

                expert = experts[exp_id]

                if joint:
                    # ── Joint mode: patch ALL linears at once, single error ───
                    orig_forwards = {}
                    for block_name in block_names:
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            continue
                        orig_forwards[block_name] = linear.forward
                        quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                        linear.forward = quant_fwd

                    with torch.inference_mode():
                        for i in range(num_samples):
                            quantized_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

                    joint_err = torch.norm(quantized_outs - full_precision_outs).item()

                    for block_name, orig_fwd in orig_forwards.items():
                        getattr(expert, block_name).forward = orig_fwd

                    expert_err = [joint_err]
                else:
                    # ── Per-linear mode: patch one linear at a time ───────────
                    expert_err = []
                    for qlinear_block, block_name in zip(block_labels, block_names):
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            expert_err.append(0.0)
                            continue

                        orig_forward = linear.forward
                        quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                        linear.forward = quant_fwd

                        with torch.inference_mode():
                            for i in range(num_samples):
                                quantized_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

                            quant_err = torch.norm(quantized_outs.sub_(full_precision_outs)).item()
                            expert_err.append(quant_err)
                            linear.forward = orig_forward

                layer_loss[layer_idx].append(expert_err)
                label = "joint" if joint else "per-linear"
                exp_tokens = per_layer_counts[layer_idx][exp_id].item() if exp_id < len(per_layer_counts[layer_idx]) else 0
                print(f"  L{layer_idx}-E{exp_id} {quant_mode} ({label}) "
                      f"[{exp_tokens} tokens]: {expert_err}")

            # 5. serialization
            counts = per_layer_counts[layer_idx]
            mean_tokens = counts.float().mean().item() if counts.sum().item() > 0 else 1.0
            layer_save = {}
            for e in range(len(layer_loss[layer_idx])):
                tok = counts[e].item() if e < len(counts) else 0
                layer_save[e] = {
                    "error": layer_loss[layer_idx][e],
                    "tokens": tok,
                    "token_ratio": round(tok / mean_tokens, 4) if mean_tokens > 0 else 0.0,
                }
            layer_loss_save[layer_idx] = layer_save
            with open(save_path, "w") as f:
                json.dump(layer_loss_save, f)
            print(f"  Layer-{layer_idx} saved ({len(layer_save)} experts)")

    ################# METRIC 3: KL DIVERGENCE ON LOGITS (kl_divergence) #################
    elif metric == "kl_divergence":
        # Loop order: (samples → experts) so we only hold ONE sample's logits
        # on GPU at a time. No CPU storage of reference log-probs.
        #
        # For each sample:
        #   1. Reference forward (BF16) → ref_log_probs  [1, seqlen, vocab] on GPU
        #   2. For each expert: patch → quantized forward → KL vs ref → unpatch
        #   3. Discard ref_log_probs, next sample

        inps = torch.vstack([d.to(dev) for d in dataloader])  # [num_samples, seqlen]

        # ── Capture per-layer routing in a pre-pass ──────────────────────
        per_layer_counts = {li: torch.zeros(num_experts, dtype=torch.long, device=dev)
                            for li in range(num_layers)}

        def _make_model_routing_hook(layer_idx):
            counts = per_layer_counts[layer_idx]
            def hook(module, input, output, _counts=counts):
                indices = output[2].flatten()
                _counts.index_add_(
                    0, indices,
                    torch.ones_like(indices, dtype=torch.long),
                )
            return hook

        hook_handles = []
        for li in range(num_layers):
            if hasattr(layers[li].mlp, "gate"):
                h = layers[li].mlp.gate.register_forward_hook(_make_model_routing_hook(li))
                hook_handles.append(h)

        print("Routing pre-pass (reference forwards)...")
        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc="Routing pre-pass"):
                _ = model(inps[i:i+1, :])

        for h in hook_handles:
            h.remove()

        # Print routing stats
        seqlen_model = inps.shape[-1]
        total_tokens_model = num_samples * seqlen_model
        for li in range(num_layers):
            counts = per_layer_counts[li]
            total = counts.sum().item()
            if total > 0:
                mean_c = counts.float().mean().item()
                std_c = counts.float().std().item()
                cv = std_c / mean_c if mean_c > 0 else 0
                print(f"  Layer {li}: {total_tokens_model} tokens × top-{total // total_tokens_model} "
                      f"= {total} total assignments over {num_samples} samples")
                print(f"    per-expert tokens: min={counts.min().item()} "
                      f"max={counts.max().item()} "
                      f"mean={mean_c:.1f} std={std_c:.1f} CV={cv:.3f}")

        # ── Per-layer KL divergence ──────────────────────────────────────
        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            experts = get_experts(layers[layer_idx], model_type)
            num_layer_experts = len(experts) if max_experts < 0 else min(max_experts, len(experts))

            # Resume: skip entire layer if all experts already done
            if resume and layer_idx in layer_loss_save:
                existing = layer_loss_save[layer_idx]
                all_done = all(
                    (str(e) in existing or e in existing)
                    for e in range(num_layer_experts)
                )
                if all_done:
                    for e in range(num_layer_experts):
                        entry = existing.get(str(e), existing.get(e))
                        cached_err = entry["error"] if isinstance(entry, dict) else entry
                        layer_loss[layer_idx].append(cached_err)
                    print(f"  Layer {layer_idx}: SKIP (all {num_layer_experts} experts resumed)")
                    continue

            # KL accumulators for this layer
            kl_accum = [0.0] * num_layer_experts
            n_tokens_accum = [0] * num_layer_experts
            # Track which experts need computation
            needs_compute = [True] * num_layer_experts
            if resume and layer_idx in layer_loss_save:
                existing = layer_loss_save[layer_idx]
                for e in range(num_layer_experts):
                    if str(e) in existing or e in existing:
                        needs_compute[e] = False

            for i in tqdm(range(num_samples), desc=f"L{layer_idx} samples", leave=False):
                with torch.inference_mode():
                    # 1. Reference forward (BF16, no patching) — one sample
                    ref_logits = model(inps[i:i+1, :]).logits.float()  # [1, seq, vocab]
                    ref_lp = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    ref_p = ref_lp.exp()
                    del ref_logits

                    # 2. For each expert that needs computation
                    for exp_id in range(num_layer_experts):
                        if not needs_compute[exp_id]:
                            continue

                        expert = experts[exp_id]

                        # Patch all linears
                        orig_forwards = {}
                        for block_name in block_names:
                            linear: nn.Linear = getattr(expert, block_name, None)
                            if linear is None or not hasattr(linear, "weight"):
                                continue
                            orig_forwards[block_name] = linear.forward
                            quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                            linear.forward = quant_fwd

                        # Quantized forward
                        q_logits = model(inps[i:i+1, :]).logits.float()
                        q_lp = torch.nn.functional.log_softmax(q_logits, dim=-1)
                        del q_logits

                        # KL(P_ref || Q_quant) = sum(P * (log P - log Q))
                        kl_per_token = (ref_p * (ref_lp - q_lp)).sum(dim=-1)  # [1, seqlen]
                        kl_accum[exp_id] += kl_per_token.sum().item()
                        n_tokens_accum[exp_id] += kl_per_token.numel()
                        del q_lp, kl_per_token

                        # Unpatch
                        for bn, orig_fwd in orig_forwards.items():
                            getattr(expert, bn).forward = orig_fwd

                    del ref_lp, ref_p

            # Finalize per-expert KL for this layer
            for exp_id in range(num_layer_experts):
                if not needs_compute[exp_id]:
                    existing = layer_loss_save[layer_idx]
                    entry = existing.get(str(exp_id), existing.get(exp_id))
                    cached_err = entry["error"] if isinstance(entry, dict) else entry
                    layer_loss[layer_idx].append(cached_err)
                    exp_tokens = per_layer_counts[layer_idx][exp_id].item() if exp_id < len(per_layer_counts[layer_idx]) else 0
                    print(f"  L{layer_idx}-E{exp_id} SKIP (resumed) [{exp_tokens} tokens]: {cached_err}")
                else:
                    mean_kl = kl_accum[exp_id] / max(n_tokens_accum[exp_id], 1)
                    expert_err = [mean_kl]
                    layer_loss[layer_idx].append(expert_err)
                    exp_tokens = per_layer_counts[layer_idx][exp_id].item() if exp_id < len(per_layer_counts[layer_idx]) else 0
                    print(f"  L{layer_idx}-E{exp_id} {quant_mode} (KL) "
                          f"[{exp_tokens} tokens]: mean_KL={mean_kl:.6f}")

            # Serialization
            counts = per_layer_counts[layer_idx]
            mean_tokens = counts.float().mean().item() if counts.sum().item() > 0 else 1.0
            layer_save = {}
            for e in range(len(layer_loss[layer_idx])):
                tok = counts[e].item() if e < len(counts) else 0
                layer_save[e] = {
                    "error": layer_loss[layer_idx][e],
                    "tokens": tok,
                    "token_ratio": round(tok / mean_tokens, 4) if mean_tokens > 0 else 0.0,
                }
            layer_loss_save[layer_idx] = layer_save
            with open(save_path, "w") as f:
                json.dump(layer_loss_save, f)
            print(f"  Layer-{layer_idx} saved ({len(layer_save)} experts)")

    ################# QUANTIZE ALL: patch every expert, single measurement #################
    elif metric == "quantize_all":

        inps = torch.vstack([d.to(dev) for d in dataloader])  # [num_samples, seqlen]

        # ── Patch ALL experts in ALL layers ───────────────────────────────
        print("Patching all experts with quantized forwards...")
        all_orig_forwards: list[tuple[nn.Linear, object]] = []
        patched_count = 0
        for layer_idx in range(num_layers):
            experts = get_experts(layers[layer_idx], model_type)
            n = len(experts) if max_experts < 0 else min(max_experts, len(experts))
            for exp_id in range(n):
                expert = experts[exp_id]
                for block_name in block_names:
                    linear: nn.Linear = getattr(expert, block_name, None)
                    if linear is None or not hasattr(linear, "weight"):
                        continue
                    all_orig_forwards.append((linear, linear.forward))
                    quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                    linear.forward = quant_fwd
                    patched_count += 1
        print(f"  Patched {patched_count} linears across {num_layers} layers")

        # ── Compute reference (BF16) and quantized outputs ───────────────
        # Use perplexity + KL divergence as aggregate metrics
        print("Computing reference (BF16) outputs...")
        # Unpatch temporarily for reference
        for linear, orig_fwd in all_orig_forwards:
            linear.forward = orig_fwd

        ref_nll_sum = 0.0
        ref_n_tokens = 0
        kl_sum = 0.0

        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc="Reference + quantized"):
                input_ids = inps[i:i+1, :]  # [1, seqlen]

                # Reference forward
                ref_logits = model(input_ids).logits.float()  # [1, seqlen, vocab]
                ref_lp = torch.nn.functional.log_softmax(ref_logits, dim=-1)

                # Reference perplexity (next-token prediction)
                # logits[i] predicts token[i+1]
                shift_lp = ref_lp[:, :-1, :]       # [1, seqlen-1, vocab]
                shift_targets = input_ids[:, 1:]    # [1, seqlen-1]
                nll = -shift_lp.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
                ref_nll_sum += nll.sum().item()
                ref_n_tokens += nll.numel()

                ref_p = ref_lp.exp()

                # Repatch for quantized forward
                for linear, _ in all_orig_forwards:
                    # Find and apply the quant forward again
                    pass
                # More efficient: just re-apply all patches
                del ref_logits  # free before quantized forward

        # Actually, let's do it properly: reference pass, then quantized pass
        # Reference pass (already unpatched above)
        ref_nll_sum = 0.0
        ref_n_tokens = 0
        ref_log_probs_list = []  # store ONE at a time below

        print("  Reference pass...")
        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc="BF16 reference"):
                ref_logits = model(inps[i:i+1, :]).logits.float()
                ref_lp = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                shift_lp = ref_lp[:, :-1, :]
                shift_targets = inps[i:i+1, 1:]
                nll = -shift_lp.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
                ref_nll_sum += nll.sum().item()
                ref_n_tokens += nll.numel()
                del ref_logits, ref_lp, shift_lp, nll

        ref_ppl = torch.exp(torch.tensor(ref_nll_sum / max(ref_n_tokens, 1))).item()
        print(f"  Reference perplexity: {ref_ppl:.4f}")

        # Re-patch all
        patch_idx = 0
        for layer_idx in range(num_layers):
            experts = get_experts(layers[layer_idx], model_type)
            n = len(experts) if max_experts < 0 else min(max_experts, len(experts))
            for exp_id in range(n):
                expert = experts[exp_id]
                for block_name in block_names:
                    linear: nn.Linear = getattr(expert, block_name, None)
                    if linear is None or not hasattr(linear, "weight"):
                        continue
                    quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                    linear.forward = quant_fwd

        # Quantized pass
        quant_nll_sum = 0.0
        quant_n_tokens = 0
        kl_sum = 0.0
        kl_n_tokens = 0

        print(f"  Quantized pass ({quant_mode}, all experts)...")
        # Need reference log probs for KL — compute both in lockstep
        # Unpatch → ref, repatch → quant, per sample
        for linear, orig_fwd in all_orig_forwards:
            linear.forward = orig_fwd  # unpatch all first

        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc=f"All-{quant_mode}"):
                input_ids = inps[i:i+1, :]

                # Reference (unpatched)
                ref_logits = model(input_ids).logits.float()
                ref_lp = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                del ref_logits

                # Repatch all
                patch_idx = 0
                for layer_idx in range(num_layers):
                    exps = get_experts(layers[layer_idx], model_type)
                    n = len(exps) if max_experts < 0 else min(max_experts, len(exps))
                    for exp_id in range(n):
                        expert = exps[exp_id]
                        for block_name in block_names:
                            linear = getattr(expert, block_name, None)
                            if linear is None or not hasattr(linear, "weight"):
                                continue
                            quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                            linear.forward = quant_fwd

                # Quantized forward
                q_logits = model(input_ids).logits.float()
                q_lp = torch.nn.functional.log_softmax(q_logits, dim=-1)
                del q_logits

                # Quantized perplexity
                shift_qlp = q_lp[:, :-1, :]
                shift_targets = input_ids[:, 1:]
                nll = -shift_qlp.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
                quant_nll_sum += nll.sum().item()
                quant_n_tokens += nll.numel()

                # KL(P_ref || Q_quant)
                ref_p = ref_lp.exp()
                kl_per_token = (ref_p * (ref_lp - q_lp)).sum(dim=-1)
                kl_sum += kl_per_token.sum().item()
                kl_n_tokens += kl_per_token.numel()

                del ref_lp, ref_p, q_lp, kl_per_token, nll

                # Unpatch all for next reference forward
                for linear, orig_fwd in all_orig_forwards:
                    linear.forward = orig_fwd

        quant_ppl = torch.exp(torch.tensor(quant_nll_sum / max(quant_n_tokens, 1))).item()
        mean_kl = kl_sum / max(kl_n_tokens, 1)

        print(f"\n{'='*60}")
        print(f"  FULL MODEL QUANTIZATION RESULTS ({quant_mode})")
        print(f"{'='*60}")
        print(f"  Reference perplexity (BF16):  {ref_ppl:.4f}")
        print(f"  Quantized perplexity ({quant_mode:>5}):  {quant_ppl:.4f}")
        print(f"  Perplexity increase:          {quant_ppl - ref_ppl:+.4f} ({(quant_ppl/ref_ppl - 1)*100:+.2f}%)")
        print(f"  Mean KL divergence:           {mean_kl:.6f}")
        print(f"  Samples: {num_samples}, Tokens: {kl_n_tokens}")
        print(f"{'='*60}")

        # Save results
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        results = {
            "quant_mode": quant_mode,
            "ref_perplexity": ref_ppl,
            "quant_perplexity": quant_ppl,
            "ppl_increase": quant_ppl - ref_ppl,
            "ppl_increase_pct": (quant_ppl / ref_ppl - 1) * 100,
            "mean_kl_divergence": mean_kl,
            "num_samples": num_samples,
            "num_tokens": kl_n_tokens,
            "num_layers": num_layers,
            "num_experts": num_experts,
        }
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {save_path}")

        # Restore all
        for linear, orig_fwd in all_orig_forwards:
            linear.forward = orig_fwd

    else:
        raise ValueError(f"Unknown metric: {metric}")

    del model
    gc.collect(); torch.cuda.empty_cache()

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
        description="Quantization Sensitivity Analysis for MoE Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization modes:
  nvfp4   W4A4  — real FP4 GEMM via FlashInfer mm_fp4 (Blackwell GPU required)
  fp8     W8A8  — real FP8 GEMM via FlashInfer gemm_fp8_nt_groupwise
  a16w4   W4A16 — fake-quantize weight to FP4 E2M1 (per-block), BF16 activation

Metrics:
  layer_out_norm  — L2 norm of single-layer output difference (fast, per-layer)
  model_out_norm  — L2 norm of full-model hidden state difference
  kl_divergence   — KL(P_ref || Q_quant) on output logits (per-expert sensitivity)
  quantize_all    — Quantize ALL experts, report perplexity + KL (whole-model baseline)

Examples:
  # Full-model baseline: quantize everything, get perplexity
  python nf4_sensitivity.py --model_path /path/to/model --quant_mode nvfp4 --metric quantize_all
  python nf4_sensitivity.py --model_path /path/to/model --quant_mode fp8 --metric quantize_all

  # Per-expert KL sensitivity
  python nf4_sensitivity.py --model_path /path/to/model --quant_mode nvfp4 --metric kl_divergence

  # Quick test (1 layer, 4 experts)
  python nf4_sensitivity.py --model_path /path/to/model --max_layers 1 --max_experts 4
        """,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--quant_mode", type=str, default="nvfp4",
                        choices=["nvfp4", "fp8", "a16w4"],
                        help="Quantization mode (default: nvfp4)")
    parser.add_argument("--metric", type=str, default="layer_out_norm",
                        choices=["layer_out_norm", "model_out_norm", "kl_divergence", "quantize_all"],
                        help="Sensitivity metric (default: layer_out_norm). "
                             "kl_divergence compares output logit distributions. "
                             "quantize_all patches ALL experts and reports perplexity + KL.")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--seqlen", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save_path", type=str, default="./calib/sensitivity.json",
                        help="Output JSON path")
    parser.add_argument("--max_layers", type=int, default=-1,
                        help="Max layers to process (-1 = all)")
    parser.add_argument("--max_experts", type=int, default=-1,
                        help="Max experts to analyze per layer (-1 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip layer-expert pairs that already exist in the output JSON. "
                             "Useful for resuming interrupted runs.")
    parser.add_argument("--joint", action="store_true",
                        help="Quantize all three linears (gate/up/down) simultaneously "
                             "instead of one at a time. Produces a single error per expert "
                             "and runs ~3x faster.")
    parser.add_argument("--backend", type=str, default="cudnn",
                        choices=["cudnn", "cutlass", "auto"],
                        help="FlashInfer mm_fp4 backend for nvfp4 mode (default: cudnn)")

    args = parser.parse_args()
    seed_everything(args.seed)

    dataloader = get_calibration_data(args.model_path, args.nsamples, args.seed, args.seqlen)

    get_model_quant_error(
        model_path=args.model_path,
        dataloader=dataloader,
        metric=args.metric,
        save_path=args.save_path,
        max_layers=args.max_layers,
        max_experts=args.max_experts,
        backend=args.backend,
        quant_mode=args.quant_mode,
        joint=args.joint,
        resume=args.resume,
    )
