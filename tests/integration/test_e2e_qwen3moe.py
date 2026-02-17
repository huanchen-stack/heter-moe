"""E2E test: Run one layer of Qwen3MoeForCausalLM with HeteroCutlassFusedMoE.

Same pattern as TRT-LLM's test_moe_cudagraph.py but with our custom MoE
swapped in via the library's ``patch_moe_factory()`` integration.

Usage::

    # Functional test (verify it runs):
    python tests/integration/test_e2e_qwen3moe.py

    # Benchmark (compare baseline vs hetero):
    python tests/integration/test_e2e_qwen3moe.py --benchmark

    # Custom params:
    python tests/integration/test_e2e_qwen3moe.py --batch_size 16 --nvfp4_ratio 0.75
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

# Bypass ConfigurableMoE wrapper so create_moe falls through to create_moe_backend.
os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"

import torch

from heter_moe.integrations.trtllm import patch_moe_factory, unpatch_moe_factory


# ---------------------------------------------------------------------------
# Model config creation (same as test_moe_cudagraph.py)
# ---------------------------------------------------------------------------


def create_qwen3moe_config(
    output_dir: str,
    num_experts: int = 128,
    num_experts_per_tok: int = 8,
) -> str:
    """Create a minimal 1-layer Qwen3MoE HuggingFace config for dummy loading."""
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "num_hidden_layers": 1,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(Path(output_dir) / "config.json", "w") as f:
        json.dump(config, f)
    return output_dir


# ---------------------------------------------------------------------------
# Verify the MoE class was actually swapped
# ---------------------------------------------------------------------------


def verify_moe_class(llm, expected_cls_name: str = "HeteroCutlassFusedMoE") -> bool:
    """Walk the model tree and check that the MoE module is our custom class."""
    model = None
    for attr_chain in [
        ["_executor", "_model", "model"],   # PyExecutor path
        ["model"],
    ]:
        obj = llm
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            model = obj
            break

    if model is None:
        print("WARNING: Could not locate model object for MoE class verification.")
        return False

    layers = getattr(model, "layers", [])
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue
        cls_name = type(experts).__name__
        if cls_name == expected_cls_name:
            print(f"  Layer {i}: MoE class = {cls_name} ✓")
            return True
        else:
            print(f"  Layer {i}: MoE class = {cls_name} (expected {expected_cls_name})")
            return False

    print("WARNING: No MoE layers found in model.")
    return False


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------


def run_inference(
    label: str,
    batch_size: int = 8,
    warmup: int = 5,
    runs: int = 10,
    nvfp4_ratio: float = 0.5,
    use_hetero: bool = True,
    num_experts: int = 128,
):
    """Create a 1-layer Qwen3MoE, optionally with HeteroCutlassFusedMoE,
    and run generation."""
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  batch_size={batch_size}  use_hetero={use_hetero}  nvfp4_ratio={nvfp4_ratio}")
    print(f"{'='*70}")

    if use_hetero:
        patch_moe_factory(nvfp4_ratio=nvfp4_ratio)

    model_dir = tempfile.mkdtemp(prefix=f"qwen3moe_{label}_")
    create_qwen3moe_config(model_dir, num_experts=num_experts)

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=1,
        load_format="dummy",
        max_batch_size=batch_size,
        max_seq_len=256,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25),
    )

    if use_hetero:
        verify_moe_class(llm, "HeteroCutlassFusedMoE")

    # Fixed prompts for consistent batch size
    prompts = [[100 + (i % 100) for i in range(128)] for _ in range(batch_size)]
    sp = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=32,
        end_id=-1,
        pad_id=-1,
    )

    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        llm.generate(prompts, sp)
    torch.cuda.synchronize()
    print("Warmup done.")

    # Measure
    print(f"Measuring ({runs} runs)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    total_tokens = 0
    for i in range(runs):
        torch.cuda.synchronize()
        start_event.record()
        outputs = llm.generate(prompts, sp)
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        times.append(elapsed)

        tokens = sum(
            len(o.outputs[0].token_ids)
            for o in outputs
            if o.outputs and o.outputs[0].token_ids
        )
        total_tokens += tokens

        if (i + 1) % max(1, runs // 3) == 0:
            print(f"  {i+1}/{runs}: {elapsed:.2f} ms, {tokens} tokens")

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    avg_tokens = total_tokens / runs
    throughput = avg_tokens / (avg_time / 1000) if avg_time > 0 else 0

    print(f"\n--- {label} ---")
    print(f"  Avg time : {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min / Max: {min(times):.2f} / {max(times):.2f} ms")
    print(f"  Tokens   : {avg_tokens:.0f} / run")
    print(f"  Throughput: {throughput:.1f} tok/s")

    result = {
        "label": label,
        "batch_size": batch_size,
        "use_hetero": use_hetero,
        "nvfp4_ratio": nvfp4_ratio,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "avg_tokens": avg_tokens,
        "throughput_tok_s": throughput,
    }

    del llm
    torch.cuda.empty_cache()

    if use_hetero:
        unpatch_moe_factory()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="E2E test: Qwen3MoE with HeteroCutlassFusedMoE",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--nvfp4_ratio", type=float, default=0.5)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run both baseline and hetero, then compare.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("QWEN3-MOE E2E TEST WITH HeteroCutlassFusedMoE")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM : {torch.cuda.get_device_capability()}")

    results = {}

    if args.benchmark:
        results["baseline"] = run_inference(
            label="baseline",
            batch_size=args.batch_size,
            warmup=args.warmup,
            runs=args.runs,
            use_hetero=False,
            num_experts=args.num_experts,
        )

    results["hetero"] = run_inference(
        label="hetero",
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        nvfp4_ratio=args.nvfp4_ratio,
        use_hetero=True,
        num_experts=args.num_experts,
    )

    if args.benchmark and "baseline" in results and "hetero" in results:
        bl = results["baseline"]
        ht = results["hetero"]
        overhead = ht["avg_time_ms"] / bl["avg_time_ms"] if bl["avg_time_ms"] > 0 else 0

        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"Baseline : {bl['avg_time_ms']:.2f} ms, {bl['throughput_tok_s']:.1f} tok/s")
        print(f"Hetero   : {ht['avg_time_ms']:.2f} ms, {ht['throughput_tok_s']:.1f} tok/s")
        print(f"Overhead : {overhead:.3f}x")
        if overhead < 1.15:
            print("  -> Two-GEMM split has < 15% overhead. Good.")
        else:
            print("  -> Overhead > 15%. Expected for Phase 1 (both groups BF16, no NVFP4 savings).")
        print("=" * 70)

    out_path = Path("hetero_moe_e2e_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
