"""
Detailed NVFP4 diagnostic: test different transpose/quantize strategies
to find what produces correct mm_fp4 output on cutlass backend.

Usage:
    python test_nvfp4.py
"""

import torch
import torch.nn as nn
from torch import Tensor
from quant import compute_nvfp4_global_scale, NVFP4_BLOCK_SIZE


def cos_sim(a, b):
    return nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def try_mm(label, x_fp4, b, x_sf, b_sf, alpha, ref):
    from flashinfer import mm_fp4
    try:
        out = mm_fp4(
            x_fp4, b, x_sf, b_sf, alpha,
            out_dtype=torch.bfloat16, block_size=NVFP4_BLOCK_SIZE,
            backend="cutlass", use_nvfp4=True,
        )
        cs = cos_sim(ref, out)
        me = (ref - out).abs().max().item()
        print(f"  {label:<50} cos={cs:.4f}  max_err={me:.4f}")
        return cs, out
    except Exception as e:
        print(f"  {label:<50} ERROR: {str(e)[:80]}")
        return None, None


if __name__ == "__main__":
    from flashinfer import nvfp4_quantize, mm_fp4, SfLayout

    M, K, N = 16, 2048, 1024
    linear = nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    ref = linear(x)  # ground truth: x @ W^T

    w = linear.weight.data  # [N, K]
    print(f"Shapes: x={list(x.shape)}, w={list(w.shape)}, ref={list(ref.shape)}")
    print(f"Expected: x[{M},{K}] @ w^T[{K},{N}] = [{M},{N}]")
    print()

    # Quantize x (activation) — same for all tests
    x_global_sf = compute_nvfp4_global_scale(x)
    x_fp4, x_sf = nvfp4_quantize(
        x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
    )
    print(f"x_fp4 shape: {list(x_fp4.shape)}, dtype: {x_fp4.dtype}")
    print(f"x_sf  shape: {list(x_sf.shape)},  dtype: {x_sf.dtype}")
    print()

    # ====================================================================
    # Strategy A: Quantize w [N,K], then .T (current code)
    # ====================================================================
    print("=" * 70)
    print("Strategy A: quantize w[N,K], then transpose packed result")
    print("=" * 70)
    w_global_sf = compute_nvfp4_global_scale(w)
    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
    )
    print(f"  w_fp4 shape: {list(w_fp4.shape)}, dtype: {w_fp4.dtype}")
    print(f"  w_sf  shape: {list(w_sf.shape)},  dtype: {w_sf.dtype}")
    print(f"  w_fp4.T shape: {list(w_fp4.T.shape)}")
    print(f"  w_sf.T  shape: {list(w_sf.T.shape)}")

    alpha_A = torch.tensor(
        1.0 / (x_global_sf.item() * w_global_sf.item()),
        device=x.device, dtype=torch.float32,
    )

    # A1: pass w_fp4.T, w_sf.T (current code)
    try_mm("A1: mm_fp4(x, w_fp4.T, x_sf, w_sf.T)",
           x_fp4, w_fp4.T.contiguous(), x_sf, w_sf.T.contiguous(), alpha_A, ref)

    # A2: pass w_fp4, w_sf (no transpose)
    try_mm("A2: mm_fp4(x, w_fp4, x_sf, w_sf)",
           x_fp4, w_fp4, x_sf, w_sf, alpha_A, ref)

    # A3: transpose only w_fp4, not w_sf
    try_mm("A3: mm_fp4(x, w_fp4.T, x_sf, w_sf)",
           x_fp4, w_fp4.T.contiguous(), x_sf, w_sf, alpha_A, ref)

    # A4: transpose only w_sf, not w_fp4
    try_mm("A4: mm_fp4(x, w_fp4, x_sf, w_sf.T)",
           x_fp4, w_fp4, x_sf, w_sf.T.contiguous(), alpha_A, ref)

    print()

    # ====================================================================
    # Strategy B: Quantize w^T [K,N] directly (no packed transpose)
    # ====================================================================
    print("=" * 70)
    print("Strategy B: quantize w.T[K,N] directly")
    print("=" * 70)
    wt = w.T.contiguous()  # [K, N]
    wt_global_sf = compute_nvfp4_global_scale(wt)
    wt_fp4, wt_sf = nvfp4_quantize(
        wt, wt_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
    )
    print(f"  wt_fp4 shape: {list(wt_fp4.shape)}, dtype: {wt_fp4.dtype}")
    print(f"  wt_sf  shape: {list(wt_sf.shape)},  dtype: {wt_sf.dtype}")

    alpha_B = torch.tensor(
        1.0 / (x_global_sf.item() * wt_global_sf.item()),
        device=x.device, dtype=torch.float32,
    )

    # B1: pass wt_fp4, wt_sf (no transpose)
    try_mm("B1: mm_fp4(x, wt_fp4, x_sf, wt_sf)",
           x_fp4, wt_fp4, x_sf, wt_sf, alpha_B, ref)

    # B2: pass wt_fp4.T, wt_sf.T
    try_mm("B2: mm_fp4(x, wt_fp4.T, x_sf, wt_sf.T)",
           x_fp4, wt_fp4.T.contiguous(), x_sf, wt_sf.T.contiguous(), alpha_B, ref)

    print()

    # ====================================================================
    # Strategy C: Quantize w [N,K] with do_shuffle=True
    # ====================================================================
    print("=" * 70)
    print("Strategy C: quantize w[N,K] with do_shuffle=True")
    print("=" * 70)
    ws_fp4, ws_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=True,
    )
    print(f"  ws_fp4 shape: {list(ws_fp4.shape)}, dtype: {ws_fp4.dtype}")
    print(f"  ws_sf  shape: {list(ws_sf.shape)},  dtype: {ws_sf.dtype}")

    try_mm("C1: mm_fp4(x, ws_fp4.T, x_sf, ws_sf.T) [shuffled]",
           x_fp4, ws_fp4.T.contiguous(), x_sf, ws_sf.T.contiguous(), alpha_A, ref)

    try_mm("C2: mm_fp4(x, ws_fp4, x_sf, ws_sf) [shuffled, no T]",
           x_fp4, ws_fp4, x_sf, ws_sf, alpha_A, ref)

    print()

    # ====================================================================
    # Strategy D: Quantize w.T [K,N] with do_shuffle=True
    # ====================================================================
    print("=" * 70)
    print("Strategy D: quantize w.T[K,N] with do_shuffle=True")
    print("=" * 70)
    wts_fp4, wts_sf = nvfp4_quantize(
        wt, wt_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=True,
    )
    print(f"  wts_fp4 shape: {list(wts_fp4.shape)}, dtype: {wts_fp4.dtype}")
    print(f"  wts_sf  shape: {list(wts_sf.shape)},  dtype: {wts_sf.dtype}")

    try_mm("D1: mm_fp4(x, wts_fp4, x_sf, wts_sf) [wT, shuffled]",
           x_fp4, wts_fp4, x_sf, wts_sf, alpha_B, ref)

    try_mm("D2: mm_fp4(x, wts_fp4.T, x_sf, wts_sf.T) [wT, shuffled, T]",
           x_fp4, wts_fp4.T.contiguous(), x_sf, wts_sf.T.contiguous(), alpha_B, ref)

    print()

    # ====================================================================
    # Print sample values for the best result
    # ====================================================================
    print("=" * 70)
    print("Reference vs BF16 matmul sanity check")
    print("=" * 70)
    manual = x @ w.T
    print(f"  linear(x) vs x@w.T cos_sim: {cos_sim(ref, manual):.6f}")
    print(f"  ref[0,:5]:    {ref[0, :5].tolist()}")
    print(f"  manual[0,:5]: {manual[0, :5].tolist()}")
