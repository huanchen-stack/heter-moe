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
    checkpoint_path: str = None,
):
    """
    Compute perplexity on WikiText-2 test set using sliding window.

    Args:
        max_length: context window size per chunk
        stride: step size between chunks (defaults to max_length // 2)
        max_tokens: cap total tokens for quick testing
        checkpoint_path: if set, save/resume per-chunk progress to this file
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

    all_begins = list(range(0, seq_len, stride))
    total_chunks = len(all_begins)
    print(f"Total tokens: {seq_len}, max_length: {max_length}, stride: {stride}, chunks: {total_chunks}")

    # ── Resume from checkpoint ─────────────────────────────────────
    nll_sum = 0.0
    total_tokens = 0
    start_chunk = 0

    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        if (ckpt.get("seq_len") == seq_len
                and ckpt.get("max_length") == max_length
                and ckpt.get("stride") == stride):
            nll_sum = ckpt["nll_sum"]
            total_tokens = ckpt["total_tokens"]
            start_chunk = ckpt["completed_chunks"]
            if start_chunk >= total_chunks:
                ppl = torch.exp(torch.tensor(nll_sum / total_tokens)).item()
                print(f"  Already complete (resumed). PPL = {ppl:.4f}")
                return ppl, total_tokens
            print(f"  Resume: continuing from chunk {start_chunk}/{total_chunks} "
                  f"(nll_sum={nll_sum:.2f}, tokens={total_tokens})")

    # ── Main loop ──────────────────────────────────────────────────
    for chunk_idx in tqdm(range(start_chunk, total_chunks), desc="Computing PPL",
                          initial=start_chunk, total=total_chunks):
        begin_loc = all_begins[chunk_idx]
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

        chunk_nll = loss.sum().item()
        chunk_tokens = shift_labels.numel()
        nll_sum += chunk_nll
        total_tokens += chunk_tokens

        running_ppl = torch.exp(torch.tensor(nll_sum / total_tokens)).item()
        print(f"  chunk {chunk_idx+1}/{total_chunks}: "
              f"nll={chunk_nll:.2f}, tokens={chunk_tokens}, running_ppl={running_ppl:.4f}")

        # Checkpoint after each chunk
        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "nll_sum": nll_sum,
                    "total_tokens": total_tokens,
                    "completed_chunks": chunk_idx + 1,
                    "seq_len": seq_len,
                    "max_length": max_length,
                    "stride": stride,
                }, f, indent=2)

        if end_loc >= seq_len:
            break

    ppl = torch.exp(torch.tensor(nll_sum / total_tokens)).item()

    return ppl, total_tokens


# Available choices:
#   quant_mode: "nvfp4" (W4A4), "fp8" (W8A8), "a16w4" (W4A16 fake quant)
#   ratio:      0.0 - 1.0 (fraction of experts to quantize)
#   criteria:   RANDOM, COLDEST_COUNT, HOTTEST_COUNT, LOWEST_WEIGHT_SUM, HIGHEST_WEIGHT_SUM

CONFIGS = [
    {"name": "baseline",           "quant_mode": None,    "ratio": 0.0, "criteria": QuantizationCriteria.RANDOM},
    # {"name": "nvfp4_75pct_coldest","quant_mode": "nvfp4", "ratio": 0.75,"criteria": QuantizationCriteria.COLDEST_COUNT},
    # {"name": "nvfp4_75pct_lowest", "quant_mode": "nvfp4", "ratio": 0.75,"criteria": QuantizationCriteria.LOWEST_WEIGHT_SUM},
    {"name": "nvfp4_100pct",       "quant_mode": "nvfp4", "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
    # {"name": "fp8_75pct_coldest",  "quant_mode": "fp8",   "ratio": 0.75,"criteria": QuantizationCriteria.COLDEST_COUNT},
    # {"name": "fp8_75pct_lowest",   "quant_mode": "fp8",   "ratio": 0.75,"criteria": QuantizationCriteria.LOWEST_WEIGHT_SUM},
    {"name": "fp8_100pct",         "quant_mode": "fp8",   "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
    {"name": "a16w4_100pct",         "quant_mode": "a16w4",   "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
]


def run_experiment(
    model_name: str = "Qwen/Qwen3-30B-A3B",
    mode: str = "memory_bound",
    max_tokens: int = None,
    output_dir: str = "./experiment_results",
    configs: list = None,
    quant_modes: list = None,
    backend: str = "cudnn",
    resume: bool = False,
):
    """
    Run perplexity experiment comparing baseline vs quantized.

    Args:
        mode: "memory_bound" (seq_len=512) or "compute_bound" (seq_len=8192)
        max_tokens: cap total tokens (None = use all of WikiText-2)
        configs: list of experiment configs (defaults to CONFIGS)
        quant_modes: which quant modes to pre-create (defaults to modes used in configs)
        backend: FlashInfer backend for nvfp4 ("cudnn" or "cutlass")
        resume: if True, resume from existing checkpoint in output_dir
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

    # ── Resume: load existing results ──────────────────────────────
    results_file = output_dir / f"ppl_{mode}.json"
    results = []
    completed_configs = set()
    if resume and results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        completed_configs = {r["config"] for r in results}
        print(f"Resume: loaded {len(results)} completed configs from {results_file}")

    print(f"Loading model: {model_name}")
    print(f"Mode: {mode} (max_length={max_length}, stride={stride})")
    model = MultiPrecisionMoEModel(
        model_path=model_name,
        quant_modes=quant_modes,
        backend=backend,
    )

    for config in configs:
        config_name = config["name"]

        if config_name in completed_configs:
            # Check if the per-chunk checkpoint is fully done (or absent)
            chunk_ckpt = output_dir / f"ppl_{mode}_{config_name}_chunks.json"
            if not chunk_ckpt.exists():
                print(f"  Skip (already done): {config_name}")
                continue
            # Checkpoint file exists — might be partial, let it run to finish
            with open(chunk_ckpt) as f:
                ckpt = json.load(f)
            if ckpt.get("completed_chunks", 0) > 0:
                # Remove the old completed result so we re-add with updated value
                results = [r for r in results if r["config"] != config_name]
                completed_configs.discard(config_name)
            else:
                print(f"  Skip (already done): {config_name}")
                continue

        print(f"\n{'='*60}")
        print(f"Running: {config_name}  [mode={mode}]")
        print(f"{'='*60}")

        model.clear_hooks()

        if config["quant_mode"] is not None and config["ratio"] > 0:
            model.quantize_all_layers(
                quant_mode=config["quant_mode"],
                quantize_ratio=config["ratio"],
                criteria=config["criteria"],
            )

        chunk_ckpt_path = str(output_dir / f"ppl_{mode}_{config_name}_chunks.json") if resume else None

        ppl, num_tokens = compute_perplexity(
            model,
            max_length=max_length,
            stride=stride,
            max_tokens=max_tokens,
            checkpoint_path=chunk_ckpt_path,
        )

        result = {
            "config": config_name,
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

        # Save after each config completes
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Clean up chunk checkpoint once config is fully done
        if chunk_ckpt_path and Path(chunk_ckpt_path).exists():
            Path(chunk_ckpt_path).unlink()

    model.clear_hooks()

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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint in output_dir")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        mode=args.mode,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        backend=args.backend,
        resume=args.resume,
    )
