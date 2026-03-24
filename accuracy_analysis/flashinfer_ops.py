"""
torch.library.custom_op wrappers for FlashInfer ops.

FlashInfer's register_custom_op is intentionally disabled (returns lambda x: x)
to avoid CPU dispatch overhead. This means torch.compile cannot trace through
FlashInfer ops — Dynamo hits JIT filesystem calls and crashes.

Workaround (following vLLM pattern): wrap each op in our own torch.library.custom_op
with a register_fake shape function so torch.compile treats them as opaque CUDA ops.

See: https://github.com/flashinfer-ai/flashinfer/issues/2733
     https://github.com/vllm-project/vllm/blob/main/vllm/utils/flashinfer.py
"""

import torch
from torch import Tensor
from typing import Tuple


# ============================================================================
# NVFP4: nvfp4_quantize + mm_fp4
# ============================================================================

@torch.library.custom_op(
    "heter_moe::nvfp4_quantize",
    mutates_args=[],
    device_types="cuda",
)
def nvfp4_quantize(
    a: Tensor,
    a_global_sf: Tensor,
    sf_layout_value: int,
    do_shuffle: bool,
    sf_vec_size: int,
) -> Tuple[Tensor, Tensor]:
    from flashinfer import nvfp4_quantize as _nvfp4_quantize, SfLayout

    layout_map = {0: SfLayout.layout_linear, 1: SfLayout.layout_8x4, 2: SfLayout.layout_128x4}
    sf_layout = layout_map.get(sf_layout_value, SfLayout.layout_128x4)

    return _nvfp4_quantize(
        a, a_global_sf,
        sfLayout=sf_layout, do_shuffle=do_shuffle,
        sf_vec_size=sf_vec_size,
    )


@nvfp4_quantize.register_fake
def _nvfp4_quantize_fake(
    a: Tensor, a_global_sf: Tensor,
    sf_layout_value: int, do_shuffle: bool, sf_vec_size: int,
) -> Tuple[Tensor, Tensor]:
    m, k = a.shape
    return (
        a.new_empty([m, k // 2], dtype=torch.uint8),
        a.new_empty([m, k // sf_vec_size], dtype=torch.uint8),
    )


SF_LAYOUT_128x4 = 2


@torch.library.custom_op(
    "heter_moe::mm_fp4",
    mutates_args=[],
    device_types="cuda",
)
def mm_fp4(
    A: Tensor, B: Tensor,
    A_scale: Tensor, B_scale: Tensor,
    alpha: Tensor,
    block_size: int,
    backend: str,
) -> Tensor:
    from flashinfer import mm_fp4 as _mm_fp4
    return _mm_fp4(
        A, B, A_scale, B_scale, alpha,
        out_dtype=torch.bfloat16, block_size=block_size,
        backend=backend, use_nvfp4=True,
    )


@mm_fp4.register_fake
def _mm_fp4_fake(
    A: Tensor, B: Tensor,
    A_scale: Tensor, B_scale: Tensor,
    alpha: Tensor,
    block_size: int, backend: str,
) -> Tensor:
    return torch.empty(A.shape[0], B.shape[1], dtype=torch.bfloat16, device=A.device)


# ============================================================================
# FP8: per_token_cast_to_fp8, per_block_cast_to_fp8, gemm_fp8_nt_groupwise
# ============================================================================

@torch.library.custom_op(
    "heter_moe::per_token_cast_to_fp8",
    mutates_args=[],
    device_types="cuda",
)
def per_token_cast_to_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    from flashinfer.testing.utils import per_token_cast_to_fp8 as _cast
    return _cast(x)


@per_token_cast_to_fp8.register_fake
def _per_token_cast_to_fp8_fake(x: Tensor) -> Tuple[Tensor, Tensor]:
    M, K = x.shape
    return (
        x.new_empty([M, K], dtype=torch.float8_e4m3fn),
        x.new_empty([M, K // 128], dtype=torch.float32),
    )


@torch.library.custom_op(
    "heter_moe::per_block_cast_to_fp8",
    mutates_args=[],
    device_types="cuda",
)
def per_block_cast_to_fp8(w: Tensor) -> Tuple[Tensor, Tensor]:
    from flashinfer.testing.utils import per_block_cast_to_fp8 as _cast
    return _cast(w)


@per_block_cast_to_fp8.register_fake
def _per_block_cast_to_fp8_fake(w: Tensor) -> Tuple[Tensor, Tensor]:
    N, K = w.shape
    return (
        w.new_empty([N, K], dtype=torch.float8_e4m3fn),
        w.new_empty([N // 128, K // 128], dtype=torch.float32),
    )


@torch.library.custom_op(
    "heter_moe::gemm_fp8_nt_groupwise",
    mutates_args=["out"],
    device_types="cuda",
)
def gemm_fp8_nt_groupwise(
    x_fp8: Tensor, w_fp8: Tensor,
    a_scale: Tensor, b_scale: Tensor,
    out: Tensor,
) -> Tensor:
    from flashinfer.gemm import gemm_fp8_nt_groupwise as _gemm
    _gemm(x_fp8, w_fp8, a_scale, b_scale, out=out, scale_major_mode="MN")
    return out


@gemm_fp8_nt_groupwise.register_fake
def _gemm_fp8_nt_groupwise_fake(
    x_fp8: Tensor, w_fp8: Tensor,
    a_scale: Tensor, b_scale: Tensor,
    out: Tensor,
) -> Tensor:
    return out
