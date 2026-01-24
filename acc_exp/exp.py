import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from model import MultiPrecisionMoEModel, QuantizationCriteria
from quantization import Precision


def compute_perplexity(
    model: MultiPrecisionMoEModel,
    max_length: int = 2048,
    stride: int = 1024,
    max_tokens: int = None,  # Limit for quick testing
):
    """
    Compute perplexity on WikiText-2 test set.
    """
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n\n".join(texts)
    
    # Tokenize
    encodings = model.tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.model.device)
    seq_len = input_ids.size(1)
    
    if max_tokens:
        seq_len = min(seq_len, max_tokens)
        input_ids = input_ids[:, :seq_len]
    
    print(f"Total tokens: {seq_len}")
    
    # Compute perplexity with sliding window
    nlls = []
    total_tokens = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(model.model.device)
        
        # Calculate how many new tokens we're evaluating
        if begin_loc == 0:
            target_start = 1  # Skip first token (no prediction for it)
        else:
            target_start = max_length - stride
        
        with torch.no_grad():
            outputs = model.model(input_chunk)
            logits = outputs.logits  # (1, seq, vocab)
        
        # Shift for next-token prediction
        shift_logits = logits[:, target_start-1:-1, :].contiguous()
        shift_labels = input_chunk[:, target_start:].contiguous()
        
        # Compute cross entropy
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        nlls.append(loss.sum())
        total_tokens += shift_labels.numel()
        
        if end_loc >= seq_len:
            break
    
    # Perplexity
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_tokens)
    
    return ppl.item(), total_tokens


def run_tiny_experiment(
    model_name: str = "allenai/OLMoE-1B-7B-0924",
    max_tokens: int = 10000,  # ~10K tokens for quick test
    output_dir: str = "./experiment_results",
):
    """
    Run a small experiment comparing baseline vs quantized perplexity.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment configurations
    configs = [
        {"name": "baseline",            "precision": None,            "ratio": 0.0, "criteria": QuantizationCriteria.RANDOM},
        {"name": "mxfp8_50pct_random",  "precision": Precision.MXFP8, "ratio": 0.5, "criteria": QuantizationCriteria.RANDOM},
        {"name": "mxfp8_50pct_coldest", "precision": Precision.MXFP8, "ratio": 0.5, "criteria": QuantizationCriteria.COLDEST_COUNT},
        {"name": "mxfp8_50pct_hottest", "precision": Precision.MXFP8, "ratio": 0.5, "criteria": QuantizationCriteria.HOTTEST_COUNT},
        {"name": "mxfp8_50pct_lowest",  "precision": Precision.MXFP8, "ratio": 0.5, "criteria": QuantizationCriteria.LOWEST_WEIGHT_SUM},
        {"name": "mxfp8_50pct_highest", "precision": Precision.MXFP8, "ratio": 0.5, "criteria": QuantizationCriteria.HIGHEST_WEIGHT_SUM},
        {"name": "mxfp8_100pct",        "precision": Precision.MXFP8, "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
        {"name": "mxfp4_50pct_random",  "precision": Precision.MXFP4, "ratio": 0.5, "criteria": QuantizationCriteria.RANDOM},
        {"name": "mxfp4_50pct_coldest", "precision": Precision.MXFP4, "ratio": 0.5, "criteria": QuantizationCriteria.COLDEST_COUNT},
        {"name": "mxfp4_50pct_hottest", "precision": Precision.MXFP4, "ratio": 0.5, "criteria": QuantizationCriteria.HOTTEST_COUNT},
        {"name": "mxfp4_50pct_lowest",  "precision": Precision.MXFP4, "ratio": 0.5, "criteria": QuantizationCriteria.LOWEST_WEIGHT_SUM},
        {"name": "mxfp4_50pct_highest", "precision": Precision.MXFP4, "ratio": 0.5, "criteria": QuantizationCriteria.HIGHEST_WEIGHT_SUM},
        {"name": "mxfp4_100pct",        "precision": Precision.MXFP4, "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
    ]
    
    results = []
    
    # Load model once
    print(f"Loading model: {model_name}")
    model = MultiPrecisionMoEModel(
        model_name=model_name,
        quantization_dtypes=[Precision.MXFP8, Precision.MXFP4],
        decoding_only=False,  # We need to evaluate on full sequences
        force_quantization=True,
        offload=False,
    )
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Running: {config['name']}")
        print(f"{'='*50}")
        
        # Clear any previous hooks
        model.clear_hooks()
        
        # Apply quantization if needed
        if config["precision"] is not None and config["ratio"] > 0:
            model.quantize_all_layers(
                precision=config["precision"],
                quantize_ratio=config["ratio"],
                criteria=config["criteria"],
            )
        
        # Compute perplexity
        ppl, num_tokens = compute_perplexity(
            model,
            max_tokens=max_tokens,
        )
        
        result = {
            "config": config["name"],
            "precision": config["precision"].value if config["precision"] else "fp16/bf16",
            "quantize_ratio": config["ratio"],
            "perplexity": ppl,
            "num_tokens": num_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)
        
        print(f"Perplexity: {ppl:.4f}")
    
    # Save results
    results_file = output_dir / f"tiny_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Config':<20} {'Perplexity':>12} {'Δ from baseline':>15}")
    print("-" * 50)
    
    baseline_ppl = results[0]["perplexity"]
    for r in results:
        delta = r["perplexity"] - baseline_ppl
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{r['config']:<20} {r['perplexity']:>12.4f} {delta_str:>15}")
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_tiny_experiment(
        model_name="allenai/OLMoE-1B-7B-0924",
        max_tokens=10000,  # Start small, increase for real experiment
    )