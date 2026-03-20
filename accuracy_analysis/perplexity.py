import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import json
import argparse
from datetime import datetime

from mixed_precision_model import MultiPrecisionMoEModel, QuantizationCriteria


def compute_perplexity(
    model: MultiPrecisionMoEModel,
    max_length: int = 2048,
    stride: int = None,
    max_tokens: int = None,
):
    """
    Compute perplexity on WikiText-2 test set using sliding window.

    Args:
        max_length: context window size per chunk
        stride: step size between chunks (defaults to max_length // 2)
        max_tokens: cap total tokens for quick testing
    """
    if stride is None:
        stride = max_length // 2

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n\n".join(texts)

    encodings = model.tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.model.device)
    seq_len = input_ids.size(1)

    if max_tokens:
        seq_len = min(seq_len, max_tokens)
        input_ids = input_ids[:, :seq_len]

    print(f"Total tokens: {seq_len}, max_length: {max_length}, stride: {stride}")

    nlls = []
    total_tokens = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing PPL"):
        end_loc = min(begin_loc + max_length, seq_len)

        input_chunk = input_ids[:, begin_loc:end_loc].to(model.model.device)

        if begin_loc == 0:
            target_start = 1
        else:
            target_start = max_length - stride

        with torch.no_grad():
            outputs = model.model(input_chunk)
            logits = outputs.logits

        shift_logits = logits[:, target_start-1:-1, :].contiguous()
        shift_labels = input_chunk[:, target_start:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        nlls.append(loss.sum())
        total_tokens += shift_labels.numel()

        if end_loc >= seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_tokens)

    return ppl.item(), total_tokens


# Available choices:
#   quant_mode: "nvfp4" (W4A4), "fp8" (W8A8), "a16w4" (W4A16 fake quant)
#   ratio:      0.0 - 1.0 (fraction of experts to quantize)
#   criteria:   RANDOM, COLDEST_COUNT, HOTTEST_COUNT, LOWEST_WEIGHT_SUM, HIGHEST_WEIGHT_SUM

CONFIGS = [
    {"name": "baseline",           "quant_mode": None,    "ratio": 0.0, "criteria": QuantizationCriteria.RANDOM},
    {"name": "nvfp4_75pct_coldest","quant_mode": "nvfp4", "ratio": 0.75,"criteria": QuantizationCriteria.COLDEST_COUNT},
    {"name": "nvfp4_75pct_lowest", "quant_mode": "nvfp4", "ratio": 0.75,"criteria": QuantizationCriteria.LOWEST_WEIGHT_SUM},
    {"name": "nvfp4_100pct",       "quant_mode": "nvfp4", "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
    {"name": "fp8_75pct_coldest",  "quant_mode": "fp8",   "ratio": 0.75,"criteria": QuantizationCriteria.COLDEST_COUNT},
    {"name": "fp8_75pct_lowest",   "quant_mode": "fp8",   "ratio": 0.75,"criteria": QuantizationCriteria.LOWEST_WEIGHT_SUM},
    {"name": "fp8_100pct",         "quant_mode": "fp8",   "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
]


def run_experiment(
    model_name: str = "Qwen/Qwen3-30B-A3B",
    mode: str = "memory_bound",
    max_tokens: int = None,
    output_dir: str = "./experiment_results",
    configs: list = None,
    quant_modes: list = None,
    backend: str = "cudnn",
):
    """
    Run perplexity experiment comparing baseline vs quantized.

    Args:
        mode: "memory_bound" (seq_len=512) or "compute_bound" (seq_len=8192)
        max_tokens: cap total tokens (None = use all of WikiText-2)
        configs: list of experiment configs (defaults to CONFIGS)
        quant_modes: which quant modes to pre-create (defaults to modes used in configs)
        backend: FlashInfer backend for nvfp4 ("cudnn" or "cutlass")
    """
    if mode == "memory_bound":
        max_length = 512
    elif mode == "compute_bound":
        max_length = 8192
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'memory_bound' or 'compute_bound'.")

    stride = max_length  # non-overlapping chunks, each processed as batch_size=1

    if configs is None:
        configs = CONFIGS

    # Infer which quant modes are needed from configs
    if quant_modes is None:
        quant_modes = list({c["quant_mode"] for c in configs if c["quant_mode"] is not None})
    if not quant_modes:
        quant_modes = ["nvfp4"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    print(f"Mode: {mode} (max_length={max_length}, stride={stride})")
    model = MultiPrecisionMoEModel(
        model_path=model_name,
        quant_modes=quant_modes,
        backend=backend,
    )

    results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}  [mode={mode}]")
        print(f"{'='*60}")

        model.clear_hooks()

        if config["quant_mode"] is not None and config["ratio"] > 0:
            model.quantize_all_layers(
                quant_mode=config["quant_mode"],
                quantize_ratio=config["ratio"],
                criteria=config["criteria"],
            )

        ppl, num_tokens = compute_perplexity(
            model,
            max_length=max_length,
            stride=stride,
            max_tokens=max_tokens,
        )

        result = {
            "config": config["name"],
            "quant_mode": config["quant_mode"] or "bf16",
            "quantize_ratio": config["ratio"],
            "perplexity": ppl,
            "num_tokens": num_tokens,
            "mode": mode,
            "max_length": max_length,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        print(f"Perplexity: {ppl:.4f}")

    model.clear_hooks()

    # Save results
    results_file = output_dir / f"ppl_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY  [{mode}, max_length={max_length}]")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Perplexity':>12} {'Δ from baseline':>15}")
    print("-" * 55)

    baseline_ppl = results[0]["perplexity"]
    for r in results:
        delta = r["perplexity"] - baseline_ppl
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{r['config']:<25} {r['perplexity']:>12.4f} {delta_str:>15}")

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perplexity experiment with mixed-precision MoE")
    parser.add_argument("--model", type=str, default="./models/Qwen3-30B-A3B",
                        help="Local model path or HuggingFace repo ID")
    parser.add_argument("--mode", type=str, default="memory_bound",
                        choices=["memory_bound", "compute_bound"],
                        help="memory_bound: seq_len=512, compute_bound: seq_len=8192")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Cap total tokens (default: all of WikiText-2)")
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    parser.add_argument("--backend", type=str, default="cudnn",
                        choices=["cudnn", "cutlass"])
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        mode=args.mode,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        backend=args.backend,
    )
