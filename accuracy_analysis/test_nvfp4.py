"""
NVFP4 / FP8 accuracy test: compare quantized linear output against BF16 reference.

Uses make_quant_forward to test NVFP4/FP8 accuracy against BF16 reference.

Usage:
    python test_nvfp4.py
"""

import torch
import torch.nn as nn
from quant import make_quant_forward, make_fp8_forward, ExpertMLP


def cos_sim(a, b):
    return nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def test_linear(linear, x):
    """Compare BF16 vs FP8 vs NVFP4 for a single linear."""
    ref = linear(x)

    fp8_fwd, _ = make_fp8_forward(linear)
    fp8_out = fp8_fwd(x)

    nvfp4_fwd, _ = make_quant_forward(linear, "nvfp4", backend="cutlass")
    nvfp4_out = nvfp4_fwd(x)

    return {
        "fp8_cos": cos_sim(ref, fp8_out),
        "fp8_max_err": (ref - fp8_out).abs().max().item(),
        "nv4_cos": cos_sim(ref, nvfp4_out),
        "nv4_max_err": (ref - nvfp4_out).abs().max().item(),
        "nv4_mode": "nvfp4",
    }


def test_expert(M, hidden_dim, inter_dim):
    """Compare BF16 vs FP8 vs NVFP4 for a full ExpertMLP."""
    gate_w = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16, device="cuda")
    up_w = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16, device="cuda")
    down_w = torch.randn(hidden_dim, inter_dim, dtype=torch.bfloat16, device="cuda")
    expert = ExpertMLP(gate_w, up_w, down_w, nn.SiLU()).cuda()
    x = torch.randn(M, hidden_dim, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        ref = expert(x)

    results = {}
    for label, mode in [("fp8", "fp8"), ("nv4", "nvfp4")]:
        orig_fwds = {}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(expert, name)
            orig_fwds[name] = proj.forward
            fwd, _ = make_quant_forward(proj, mode, backend="cutlass")
            proj.forward = fwd

        with torch.no_grad():
            out = expert(x)

        for name, fwd in orig_fwds.items():
            getattr(expert, name).forward = fwd

        results[f"{label}_cos"] = cos_sim(ref, out)
        results[f"{label}_max_err"] = (ref - out).abs().max().item()

    return results


if __name__ == "__main__":
    # ================================================================
    # Test 1: Single linear layer
    # ================================================================
    print("=" * 80)
    print("Single nn.Linear: BF16 vs FP8 vs NVFP4")
    print("=" * 80)
    print(f"{'M':>6} {'K':>6} {'N':>6} | {'FP8 cos':>9} {'FP8 err':>9} | {'NV4 cos':>9} {'NV4 err':>9} {'mode':>8}")
    print("-" * 80)

    shapes = [
        (1, 2048, 768),     # Qwen3-30B-A3B gate/up shape
        (4, 2048, 768),
        (16, 2048, 768),
        (128, 2048, 768),
        (1, 768, 2048),     # Qwen3-30B-A3B down_proj shape
        (1, 2048, 1024),
        (1, 1024, 2048),
        (512, 2048, 768),
    ]

    for M, K, N in shapes:
        linear = nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        r = test_linear(linear, x)
        print(f"{M:>6} {K:>6} {N:>6} | {r['fp8_cos']:>9.4f} {r['fp8_max_err']:>9.4f} | "
              f"{r['nv4_cos']:>9.4f} {r['nv4_max_err']:>9.4f} {r['nv4_mode']:>8}")

    # ================================================================
    # Test 2: Full ExpertMLP
    # ================================================================
    print()
    print("=" * 80)
    print("Full ExpertMLP: BF16 vs FP8 vs NVFP4")
    print("=" * 80)
    print(f"{'M':>6} {'hidden':>8} {'inter':>8} | {'FP8 cos':>9} {'FP8 err':>9} | {'NV4 cos':>9} {'NV4 err':>9}")
    print("-" * 80)

    expert_shapes = [
        (1, 2048, 768),     # Qwen3-30B-A3B dimensions
        (4, 2048, 768),
        (16, 2048, 768),
        (128, 2048, 768),
        (1, 2048, 1024),
    ]

    for M, hidden, inter in expert_shapes:
        r = test_expert(M, hidden, inter)
        print(f"{M:>6} {hidden:>8} {inter:>8} | {r['fp8_cos']:>9.4f} {r['fp8_max_err']:>9.4f} | "
              f"{r['nv4_cos']:>9.4f} {r['nv4_max_err']:>9.4f}")

    # ================================================================
    # Test 3: Verify NVFP4 vs FP8 behavior
    # ================================================================
    print()
    print("=" * 80)
    print("Sanity check: NVFP4 vs FP8 must differ")
    print("=" * 80)

    for M, K, N in [(1, 2048, 1024), (1, 2048, 768), (1, 768, 2048), (128, 2048, 1024)]:
        linear = nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        fp8_fwd, _ = make_quant_forward(linear, "fp8")
        nv4_fwd, _ = make_quant_forward(linear, "nvfp4", backend="cutlass")

        with torch.no_grad():
            fp8_out = fp8_fwd(x)
            nv4_out = nv4_fwd(x)

        same = torch.allclose(fp8_out, nv4_out)
        diff = (fp8_out - nv4_out).abs().max().item()
        status = "FAIL (identical!)" if same else "OK (different)"
        print(f"  [{M:>3}x{K}->{N}] max_diff={diff:.6f}  {status}")

    print()
    print("nvfp4 = direct FP4 GEMM path")
    print("Qwen3-30B-A3B (hidden=2048, inter=768): ALL projections fall back to FP8")
    print("Expected: all cos > 0.90")
