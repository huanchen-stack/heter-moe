"""
Throughput benchmark mimicking academic paper methodology.
Uses ShareGPT dataset with realistic length distributions.
"""
import json
import time
import random
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer
from tensorrt_llm import LLM, SamplingParams
@dataclass
class BenchmarkRequest:
    prompt: str
    prompt_len: int
    expected_output_len: int
def load_sharegpt_dataset(
    dataset_path: str,
    tokenizer,
    num_requests: int = 1000,
    max_input_len: int = 1024,
    max_output_len: int = 512,
) -> List[BenchmarkRequest]:
    """Load ShareGPT and extract prompts with realistic length distribution."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    requests = []
    for item in data:
        if len(requests) >= num_requests:
            break
        
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue
        
        # First human turn = prompt, first assistant turn = expected output length
        human_turn = next((c for c in conversations if c["from"] == "human"), None)
        assistant_turn = next((c for c in conversations if c["from"] == "gpt"), None)
        
        if not human_turn or not assistant_turn:
            continue
        
        prompt = human_turn["value"]
        prompt_tokens = tokenizer.encode(prompt)
        output_tokens = tokenizer.encode(assistant_turn["value"])
        
        prompt_len = len(prompt_tokens)
        output_len = len(output_tokens)
        
        # Filter by length constraints
        if prompt_len > max_input_len or prompt_len < 4:
            continue
        if output_len > max_output_len or output_len < 4:
            continue
        
        requests.append(BenchmarkRequest(
            prompt=prompt,
            prompt_len=prompt_len,
            expected_output_len=min(output_len, max_output_len),
        ))
    
    random.shuffle(requests)
    return requests[:num_requests]
def create_synthetic_dataset(
    tokenizer,
    num_requests: int = 1000,
    input_len: int = 512,
    output_len: int = 128,
) -> List[BenchmarkRequest]:
    """Create synthetic dataset with fixed lengths (TensorRT-LLM style)."""
    vocab_size = tokenizer.vocab_size
    requests = []
    
    for _ in range(num_requests):
        # Random token IDs from vocabulary
        token_ids = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
        prompt = tokenizer.decode(token_ids)
        
        requests.append(BenchmarkRequest(
            prompt=prompt,
            prompt_len=input_len,
            expected_output_len=output_len,
        ))
    
    return requests
def benchmark_throughput(
    model_name: str,
    requests: List[BenchmarkRequest],
    warmup_requests: int = 10,
) -> dict:
    """Run throughput benchmark."""
    
    llm = LLM(model=model_name)
    
    # Warmup
    warmup_prompts = [r.prompt for r in requests[:warmup_requests]]
    warmup_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=32)
    _ = llm.generate(warmup_prompts, warmup_params)
    
    # Benchmark
    prompts = [r.prompt for r in requests]
    sampling_params = [
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=r.expected_output_len,
            ignore_eos=True,  # Force generation to expected length
        )
        for r in requests
    ]
    
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    total_prompt_tokens = sum(r.prompt_len for r in requests)
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tokens = total_prompt_tokens + total_output_tokens
    
    results = {
        "elapsed_time_sec": elapsed_time,
        "num_requests": len(requests),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "requests_per_sec": len(requests) / elapsed_time,
        "output_tokens_per_sec": total_output_tokens / elapsed_time,  # Primary metric
        "total_tokens_per_sec": total_tokens / elapsed_time,
        "avg_prompt_len": total_prompt_tokens / len(requests),
        "avg_output_len": total_output_tokens / len(requests),
    }
    
    return results
def print_results(results: dict):
    """Print results in academic paper format."""
    print("=" * 60)
    print("THROUGHPUT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Elapsed Time:              {results['elapsed_time_sec']:.2f} sec")
    print(f"Number of Requests:        {results['num_requests']}")
    print(f"Avg Input Length:          {results['avg_prompt_len']:.1f} tokens")
    print(f"Avg Output Length:         {results['avg_output_len']:.1f} tokens")
    print("-" * 60)
    print(f"Request Throughput:        {results['requests_per_sec']:.2f} req/s")
    print(f"Output Token Throughput:   {results['output_tokens_per_sec']:.2f} tokens/s")
    print(f"Total Token Throughput:    {results['total_tokens_per_sec']:.2f} tokens/s")
    print("=" * 60)
if __name__ == "__main__":
    MODEL = "nvidia/Qwen3-30B-A3B-FP4"
    MODEL = "Qwen/Qwen3-30B-A3B"
    MODEL = "Qwen/Qwen3-30B-A3B-FP8"
    NUM_REQUESTS = 500
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # Option 1: ShareGPT dataset (realistic, preferred for papers)
    # Download from: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    requests = load_sharegpt_dataset(
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        tokenizer,
        num_requests=NUM_REQUESTS,
    )
    
    # # Option 2: Synthetic fixed-length (controlled experiments)
    # requests = create_synthetic_dataset(
    #     tokenizer,
    #     num_requests=NUM_REQUESTS,
    #     input_len=512,
    #     output_len=128,
    # )
    
    results = benchmark_throughput(MODEL, requests)
    print_results(results)
    
    # Save JSON for reproducibility
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)