"""
Quantization forward functions and MoE model architecture helpers.

Provides:
  - NVFP4 (W4A4) real FP4 GEMM via FlashInfer mm_fp4
  - FP8 (W8A8) real FP8 GEMM via FlashInfer gemm_fp8_nt_groupwise
  - A16W4 (weight-only FP4) fake quantization
  - ExpertMLP, unfuse_moe_experts, get_experts for MoE expert manipulation
"""

import torch
import torch.nn as nn
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
    absmax = tensor.abs().max()
    if torch.isnan(absmax):
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

        w_global_sf = compute_nvfp4_global_scale(w)
        w_fp4, w_sf = nvfp4_quantize(
            w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
        )
        w_fp4 = w_fp4.T
        w_sf = w_sf.T

        # Quantize activation
        x_global_sf = compute_nvfp4_global_scale(x)
        x_fp4, x_sf = nvfp4_quantize(
            x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
        )

        alpha = (1.0 / (x_global_sf * w_global_sf)).to(dtype=torch.float32)

        try:
            out = mm_fp4(
                x_fp4, w_fp4, x_sf, w_sf, alpha,
                out_dtype=torch.bfloat16, block_size=NVFP4_BLOCK_SIZE,
                backend=effective_backend[0], use_nvfp4=True,
            )
        except RuntimeError:
            if effective_backend[0] != "cutlass":
                effective_backend[0] = "cutlass"
            out = mm_fp4(
                x_fp4, w_fp4, x_sf, w_sf, alpha,
                out_dtype=torch.bfloat16, block_size=NVFP4_BLOCK_SIZE,
                backend="cutlass", use_nvfp4=True,
            )

        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    return nvfp4_forward, None


def make_nvfp4_forward_prequantized(
    linear: nn.Linear,
    backend: str = "cudnn",
) -> tuple:
    """Create an NVFP4 forward with pre-quantized weights.

    Weights are quantized once and stored in the closure. The original BF16
    weight is deleted from the linear module to free GPU memory.
    Only activation is quantized on each forward call.

    Returns (forward_fn, cleanup_fn). Call cleanup_fn() to delete BF16 weight.
    """
    from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
    effective_backend = [backend]

    w = linear.weight.data
    bias = linear.bias

    w_global_sf = compute_nvfp4_global_scale(w)
    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
    )
    w_fp4 = w_fp4.T.contiguous()
    w_sf = w_sf.T.contiguous()
    w_global_sf_inv = (1.0 / w_global_sf).to(dtype=torch.float32)

    def nvfp4_forward_preq(input: Tensor) -> Tensor:
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])

        x_global_sf = compute_nvfp4_global_scale(x)
        x_fp4, x_sf = nvfp4_quantize(
            x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
        )

        alpha = (w_global_sf_inv / x_global_sf).to(dtype=torch.float32)

        try:
            out = mm_fp4(
                x_fp4, w_fp4, x_sf, w_sf, alpha,
                out_dtype=torch.bfloat16, block_size=NVFP4_BLOCK_SIZE,
                backend=effective_backend[0], use_nvfp4=True,
            )
        except RuntimeError:
            if effective_backend[0] != "cutlass":
                effective_backend[0] = "cutlass"
            out = mm_fp4(
                x_fp4, w_fp4, x_sf, w_sf, alpha,
                out_dtype=torch.bfloat16, block_size=NVFP4_BLOCK_SIZE,
                backend="cutlass", use_nvfp4=True,
            )

        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    def cleanup():
        linear.weight = None

    return nvfp4_forward_preq, cleanup


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


def make_fp8_forward_prequantized(linear: nn.Linear) -> tuple:
    """Create a W8A8 FP8 forward with pre-quantized weights.

    Weights are quantized once and stored in the closure. The original BF16
    weight is deleted from the linear module to free GPU memory.
    Only activation is quantized on each forward call.

    Returns (forward_fn, cleanup_fn). Call cleanup_fn() to delete BF16 weight.
    """
    from flashinfer.gemm import gemm_fp8_nt_groupwise
    from flashinfer.testing.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

    w = linear.weight.data
    bias = linear.bias
    N, K = w.shape

    w_fp8, w_sf = per_block_cast_to_fp8(w)
    b_scale = w_sf.T.contiguous()

    def fp8_forward_preq(input: Tensor) -> Tensor:
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])
        M = x.shape[0]

        x_fp8, x_sf = per_token_cast_to_fp8(x)
        a_scale = x_sf.T.contiguous()

        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        gemm_fp8_nt_groupwise(
            x_fp8, w_fp8, a_scale, b_scale,
            out=out, scale_major_mode="MN",
        )

        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    def cleanup():
        linear.weight = None

    return fp8_forward_preq, cleanup


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

def make_quant_forward(
    linear: nn.Linear, quant_mode: str, backend: str = "cudnn", prequantize: bool = False,
) -> tuple:
    """Create a quantized forward function for the given mode.

    When prequantize=True, weights are quantized once and BF16 originals can be
    freed. Returns (forward_fn, cleanup_fn) where cleanup_fn deletes BF16 weight.
    This is irreversible — use only when all experts will be quantized (ratio=1.0).
    """
    if prequantize:
        if quant_mode == "nvfp4":
            return make_nvfp4_forward_prequantized(linear, backend=backend)
        elif quant_mode == "fp8":
            return make_fp8_forward_prequantized(linear)
        elif quant_mode == "a16w4":
            return make_a16w4_forward(linear)
        else:
            raise ValueError(f"Unknown quant_mode: {quant_mode}")

    if quant_mode == "nvfp4":
        return make_nvfp4_forward(linear, backend=backend)
    elif quant_mode == "fp8":
        return make_fp8_forward(linear)
    elif quant_mode == "a16w4":
        return make_a16w4_forward(linear)
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")


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
