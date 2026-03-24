"""
Downstream task evaluation for mixed-precision MoE models.

Evaluates on: MMLU-CoT, GSM8k-CoT, HellaSwag, WinoGrande
using lm-evaluation-harness (v0.4+) with a pre-loaded model.

The model is loaded once via MultiPrecisionMoEModel, quantization hooks
are applied per-config, and lm-eval wraps the live HuggingFace model
(zero reloading).

Usage:
    python downstream.py                                        # defaults
    python downstream.py --model ./models/Qwen3-30B-A3B        # custom model
    python downstream.py --tasks hellaswag winogrande           # subset of tasks
    python downstream.py --limit 100                            # quick smoke test
    python downstream.py --resume                               # resume from checkpoint
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

from mixed_precision_model import MultiPrecisionMoEModel, QuantizationCriteria

import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


# ── Task definitions ─────────────────────────────────────────────────────────
# Each task has num_fewshot baked into its YAML config in lm-eval-harness.
# We pass num_fewshot=None to respect those defaults.
#
# Task name                     | fewshot | output        | primary metric
# ------------------------------|---------|---------------|-------------------------
# mmlu_flan_cot_fewshot         | 4       | generate_until| exact_match,get-answer
# gsm8k_cot                    | 8       | generate_until| exact_match,strict-match
# hellaswag                    | 0*      | multiple_choice| acc_norm,none
# winogrande                   | 0*      | multiple_choice| acc,none
#
# * hellaswag and winogrande default to 0-shot in their YAMLs.
#   Pass num_fewshot via CLI to override (e.g., --num_fewshot 5).

TASK_REGISTRY = {
    "mmlu_cot": {
        "lm_eval_task": "mmlu_flan_cot_fewshot",
        "primary_metric": "exact_match,get-answer",
        "display_name": "MMLU-CoT (4-shot)",
    },
    "gsm8k_cot": {
        "lm_eval_task": "gsm8k_cot",
        "primary_metric": "exact_match,strict-match",
        "display_name": "GSM8k-CoT (8-shot)",
    },
    "hellaswag": {
        "lm_eval_task": "hellaswag",
        "primary_metric": "acc_norm,none",
        "display_name": "HellaSwag (0-shot)",
    },
    "winogrande": {
        "lm_eval_task": "winogrande",
        "primary_metric": "acc,none",
        "display_name": "WinoGrande (0-shot)",
    },
}

DEFAULT_TASKS = ["mmlu_cot", "gsm8k_cot", "hellaswag", "winogrande"]


# ── Quantization configs ─────────────────────────────────────────────────────
# Same pattern as perplexity.py CONFIGS.
# Available choices:
#   quant_mode: "nvfp4" (W4A4), "fp8" (W8A8), "a16w4" (W4A16 fake quant), None (bf16)
#   ratio:      0.0 - 1.0 (fraction of experts to quantize)
#   criteria:   RANDOM, COLDEST_COUNT, HOTTEST_COUNT, LOWEST_WEIGHT_SUM, HIGHEST_WEIGHT_SUM

CONFIGS = [
    {"name": "baseline",           "quant_mode": None,    "ratio": 0.0, "criteria": QuantizationCriteria.RANDOM},
    {"name": "nvfp4_100pct",       "quant_mode": "nvfp4", "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
    {"name": "fp8_100pct",         "quant_mode": "fp8",   "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
    # {"name": "nvfp4_75pct_coldest","quant_mode": "nvfp4", "ratio": 0.75,"criteria": QuantizationCriteria.COLDEST_COUNT},
    # {"name": "nvfp4_75pct_lowest", "quant_mode": "nvfp4", "ratio": 0.75,"criteria": QuantizationCriteria.LOWEST_WEIGHT_SUM},
    # {"name": "fp8_75pct_coldest",  "quant_mode": "fp8",   "ratio": 0.75,"criteria": QuantizationCriteria.COLDEST_COUNT},
    # {"name": "a16w4_100pct",       "quant_mode": "a16w4", "ratio": 1.0, "criteria": QuantizationCriteria.RANDOM},
]


def extract_metrics(eval_results: dict, task_key: str) -> dict:
    """Extract all numeric metrics for a task from lm-eval results."""
    task_info = TASK_REGISTRY[task_key]
    lm_task = task_info["lm_eval_task"]
    task_results = eval_results["results"].get(lm_task, {})

    metrics = {}
    for k, v in task_results.items():
        if isinstance(v, (int, float)) and k not in ("alias",):
            metrics[k] = v

    return metrics


def run_single_eval(
    model: MultiPrecisionMoEModel,
    tasks: list[str],
    batch_size: int | str = "auto",
    limit: int | None = None,
    num_fewshot: int | None = None,
) -> dict:
    """Run lm-eval on the current model state (with whatever hooks are active).

    Args:
        model: MultiPrecisionMoEModel (hooks already applied).
        tasks: list of task keys from TASK_REGISTRY.
        batch_size: eval batch size.
        limit: cap number of examples per task (for quick testing).
        num_fewshot: override fewshot count (None = use task default).

    Returns:
        dict mapping task_key -> {primary_metric: float, all_metrics: dict}
    """
    lm_eval_tasks = [TASK_REGISTRY[t]["lm_eval_task"] for t in tasks]

    lm = HFLM(
        pretrained=model.model,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
    )

    eval_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=lm_eval_tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )

    results = {}
    for task_key in tasks:
        task_info = TASK_REGISTRY[task_key]
        all_metrics = extract_metrics(eval_results, task_key)
        primary = all_metrics.get(task_info["primary_metric"])
        results[task_key] = {
            "primary_metric": primary,
            "primary_metric_name": task_info["primary_metric"],
            "all_metrics": all_metrics,
        }

    return results


def run_experiment(
    model_name: str = "./models/Qwen3-30B-A3B",
    tasks: list[str] = None,
    output_dir: str = "./downstream_results",
    configs: list = None,
    quant_modes: list = None,
    backend: str = "cudnn",
    batch_size: int | str = "auto",
    limit: int | None = None,
    num_fewshot: int | None = None,
    resume: bool = False,
):
    """
    Run downstream evaluation comparing baseline vs quantized configs.

    Args:
        model_name: local model path or HuggingFace repo ID.
        tasks: list of task keys (default: all 4 benchmarks).
        output_dir: directory for results JSON files.
        configs: list of quant configs (defaults to CONFIGS).
        quant_modes: which quant modes to pre-create.
        backend: FlashInfer backend for nvfp4.
        batch_size: eval batch size.
        limit: cap examples per task (None = full eval).
        num_fewshot: override fewshot (None = use task YAML default).
        resume: if True, skip already-completed configs.
    """
    if tasks is None:
        tasks = DEFAULT_TASKS
    if configs is None:
        configs = CONFIGS

    if quant_modes is None:
        quant_modes = list({c["quant_mode"] for c in configs if c["quant_mode"] is not None})
    if not quant_modes:
        quant_modes = ["nvfp4"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: load existing results ──────────────────────────────
    task_suffix = "_".join(sorted(tasks))
    results_file = output_dir / f"downstream_{task_suffix}.json"
    results = []
    completed_configs = set()
    if resume and results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        completed_configs = {r["config"] for r in results}
        print(f"Resume: loaded {len(results)} completed configs from {results_file}")

    # ── Load model once ────────────────────────────────────────────
    print(f"Loading model: {model_name}")
    print(f"Tasks: {[TASK_REGISTRY[t]['display_name'] for t in tasks]}")
    print(f"Quant modes to pre-create: {quant_modes}")
    model = MultiPrecisionMoEModel(
        model_path=model_name,
        quant_modes=quant_modes,
        backend=backend,
    )

    # ── Per-config evaluation loop ─────────────────────────────────
    for config in configs:
        config_name = config["name"]

        if config_name in completed_configs:
            print(f"  Skip (already done): {config_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {config_name}")
        print(f"{'='*60}")

        model.clear_hooks()

        if config["quant_mode"] is not None and config["ratio"] > 0:
            model.quantize_all_layers(
                quant_mode=config["quant_mode"],
                quantize_ratio=config["ratio"],
                criteria=config["criteria"],
            )

        task_results = run_single_eval(
            model,
            tasks=tasks,
            batch_size=batch_size,
            limit=limit,
            num_fewshot=num_fewshot,
        )

        result = {
            "config": config_name,
            "quant_mode": config["quant_mode"] or "bf16",
            "quantize_ratio": config["ratio"],
            "tasks": {},
            "timestamp": datetime.now().isoformat(),
        }
        for task_key, task_data in task_results.items():
            result["tasks"][task_key] = {
                "primary_metric": task_data["primary_metric"],
                "primary_metric_name": task_data["primary_metric_name"],
                "display_name": TASK_REGISTRY[task_key]["display_name"],
                "all_metrics": task_data["all_metrics"],
            }

        results.append(result)

        for task_key, task_data in task_results.items():
            display = TASK_REGISTRY[task_key]["display_name"]
            primary = task_data["primary_metric"]
            metric_name = task_data["primary_metric_name"]
            val_str = f"{primary:.4f}" if primary is not None else "N/A"
            print(f"  {display}: {metric_name} = {val_str}")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {results_file}")

    model.clear_hooks()

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY — Downstream Evaluation")
    print(f"{'='*80}")

    task_headers = [TASK_REGISTRY[t]["display_name"] for t in tasks]
    header = f"{'Config':<25}"
    for th in task_headers:
        header += f" {th:>18}"
    print(header)
    print("-" * (25 + 19 * len(tasks)))

    baseline_vals = {}
    if results:
        first = results[0]
        for t in tasks:
            baseline_vals[t] = first["tasks"].get(t, {}).get("primary_metric")

    for r in results:
        row = f"{r['config']:<25}"
        for t in tasks:
            val = r["tasks"].get(t, {}).get("primary_metric")
            if val is not None:
                baseline = baseline_vals.get(t)
                if baseline is not None and r["config"] != results[0]["config"]:
                    delta = val - baseline
                    sign = "+" if delta >= 0 else ""
                    row += f" {val:>10.4f} ({sign}{delta:.4f})"
                else:
                    row += f" {val:>18.4f}"
            else:
                row += f" {'N/A':>18}"
        print(row)

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downstream evaluation with mixed-precision MoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (all 4 benchmarks, baseline + nvfp4 + fp8)
  python downstream.py --model ./models/Qwen3-30B-A3B

  # Quick smoke test (100 examples per task)
  python downstream.py --model ./models/Qwen3-30B-A3B --limit 100

  # Specific tasks only
  python downstream.py --tasks hellaswag winogrande

  # Resume interrupted run
  python downstream.py --resume

  # Override fewshot count for all tasks
  python downstream.py --num_fewshot 5
        """,
    )
    parser.add_argument("--model", type=str, default="./models/Qwen3-30B-A3B",
                        help="Local model path or HuggingFace repo ID")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=list(TASK_REGISTRY.keys()),
                        help="Tasks to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default="./downstream_results")
    parser.add_argument("--backend", type=str, default="cudnn",
                        choices=["cudnn", "cutlass"])
    parser.add_argument("--batch_size", type=str, default="auto",
                        help="Eval batch size (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap examples per task (default: full eval)")
    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="Override fewshot count for all tasks (default: use task YAML default)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint in output_dir")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_experiment(
        model_name=args.model,
        tasks=args.tasks,
        output_dir=args.output_dir,
        backend=args.backend,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        resume=args.resume,
    )
