import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json


class ExpertTracker:
    """Track expert assignments during inference"""
    
    def __init__(self, num_layers, num_experts):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.reset()
    
    def reset(self):
        # Each entry: [num_layers, num_experts] counts for one forward pass
        self.forward_pass_counts = []
        self.current_counts = None
    
    def start_forward(self):
        """Call before each forward pass"""
        self.current_counts = torch.zeros(self.num_layers, self.num_experts)
    
    def record(self, layer_idx, expert_indices):
        """
        Record routing decisions.
        expert_indices: [num_tokens, top_k]
        """
        for k in range(expert_indices.shape[1]):
            counts = torch.bincount(
                expert_indices[:, k].cpu(),
                minlength=self.num_experts
            ).float()
            self.current_counts[layer_idx] += counts
    
    def end_forward(self):
        """Call after each forward pass"""
        if self.current_counts is not None:
            self.forward_pass_counts.append(self.current_counts.numpy())
            self.current_counts = None
    
    def get_counts(self):
        """Returns [num_forward_passes, num_layers, num_experts]"""
        return np.array(self.forward_pass_counts)


def patch_mixtral(model, tracker):
    """Patch Mixtral's MoE blocks to track routing"""
    
    for layer_idx, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        original_forward = moe.forward
        
        def make_hook(orig_fn, moe_ref, l_idx):
            def hooked_forward(hidden_states):
                batch, seq_len, dim = hidden_states.shape
                hidden_flat = hidden_states.view(-1, dim)
                
                # Get routing decisions
                router_logits = moe_ref.gate(hidden_flat)
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
                _, selected_experts = torch.topk(routing_weights, moe_ref.top_k, dim=-1)
                
                # Record: selected_experts is [batch*seq, top_k]
                tracker.record(l_idx, selected_experts)
                
                # Original forward
                return orig_fn(hidden_states)
            return hooked_forward
        
        moe.forward = make_hook(original_forward, moe, layer_idx)
    
    print(f"Patched {len(model.model.layers)} layers")


def run_inference_tracking(
    model_name="mistralai/Mixtral-8x7B-v0.1",
    num_forward_passes=200,
    batch_size=8,
    seq_length=512,
):
    """Run inference and track expert assignments"""
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Get model config
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    top_k = model.config.num_experts_per_tok
    
    print(f"Model: {num_layers} layers, {num_experts} experts, top-{top_k} routing")
    print(f"Tokens per forward pass: {batch_size * seq_length}")
    
    # Setup tracker and patch model
    tracker = ExpertTracker(num_layers, num_experts)
    patch_mixtral(model, tracker)
    
    # Load real data
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"]) > 200)
    texts = dataset["text"]
    
    print(f"\nRunning {num_forward_passes} forward passes...")
    
    with torch.no_grad():
        for i in range(num_forward_passes):
            # Sample random texts for this batch
            batch_texts = [
                texts[idx] for idx in 
                np.random.randint(0, len(texts), batch_size)
            ]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_length,
            ).to(model.device)
            
            tracker.start_forward()
            _ = model(**inputs)
            tracker.end_forward()
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_forward_passes}")
    
    counts = tracker.get_counts()
    print(f"\nCollected counts shape: {counts.shape}")
    print(f"  = [{num_forward_passes} forward passes, {num_layers} layers, {num_experts} experts]")
    
    return counts, {"num_layers": num_layers, "num_experts": num_experts, "top_k": top_k}


def analyze_batch_fluctuation(counts, window_size=3):
    """
    Analyze how the 'running batch' (tokens per expert) fluctuates.
    """
    num_passes, num_layers, num_experts = counts.shape
    
    print(f"\n{'='*70}")
    print("RUNNING BATCH FLUCTUATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Forward passes: {num_passes}")
    print(f"Window size: {window_size} consecutive forward passes")
    
    all_fluctuations = []
    
    for layer in range(num_layers):
        for expert in range(num_experts):
            expert_counts = counts[:, layer, expert]
            
            # Sliding window analysis
            for start in range(num_passes - window_size + 1):
                window = expert_counts[start:start + window_size]
                min_val = window.min()
                max_val = window.max()
                
                if min_val > 0:
                    ratio = max_val / min_val
                    all_fluctuations.append({
                        'layer': layer,
                        'expert': expert,
                        'start': start,
                        'ratio': ratio,
                        'min': int(min_val),
                        'max': int(max_val),
                        'window': window.astype(int).tolist()
                    })
    
    # Sort by ratio
    all_fluctuations.sort(key=lambda x: x['ratio'], reverse=True)
    
    # Report
    print(f"\nTop 15 fluctuations (within {window_size} forward passes):")
    print("-" * 70)
    for i, f in enumerate(all_fluctuations[:15]):
        print(f"{i+1:2d}. Layer {f['layer']:2d}, Expert {f['expert']}: "
              f"{f['ratio']:5.1f}x  (batch sizes: {f['window']})")
    
    # Statistics
    ratios = [f['ratio'] for f in all_fluctuations]
    print(f"\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Maximum fluctuation ratio: {max(ratios):.1f}x")
    print(f"Cases with >16x fluctuation: {sum(1 for r in ratios if r > 16)}")
    print(f"Cases with >10x fluctuation: {sum(1 for r in ratios if r > 10)}")
    print(f"Cases with >5x fluctuation:  {sum(1 for r in ratios if r > 5)}")
    print(f"Cases with >2x fluctuation:  {sum(1 for r in ratios if r > 2)}")
    
    # Per-layer stats
    print(f"\nPer-layer max fluctuation:")
    for layer in range(counts.shape[1]):
        layer_flucts = [f['ratio'] for f in all_fluctuations if f['layer'] == layer]
        if layer_flucts:
            print(f"  Layer {layer:2d}: max {max(layer_flucts):.1f}x")
    
    return all_fluctuations


def plot_expert_batches(counts, save_prefix="mixtral_inference"):
    """Visualize running batch sizes per expert"""
    num_passes, num_layers, num_experts = counts.shape
    
    # Plot 1: Running batch over time for selected layers
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    layers_to_plot = [0, num_layers//3, 2*num_layers//3, num_layers-1]
    
    for idx, layer in enumerate(layers_to_plot):
        ax = axes[idx // 2, idx % 2]
        for expert in range(num_experts):
            ax.plot(counts[:, layer, expert], alpha=0.8, linewidth=1.5, 
                   label=f'E{expert}')
        ax.set_xlabel('Forward Pass')
        ax.set_ylabel('Tokens in Expert (Running Batch)')
        ax.set_title(f'Layer {layer}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Running Batch Size Per Expert During Inference', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_timeseries.png', dpi=150)
    print(f"Saved {save_prefix}_timeseries.png")
    
    # Plot 2: Zoomed view showing fluctuation
    plt.figure(figsize=(14, 6))
    layer = 0
    start, end = 20, 50
    
    for expert in range(num_experts):
        plt.plot(range(start, end), counts[start:end, layer, expert],
                'o-', linewidth=2, markersize=6, label=f'Expert {expert}')
    
    plt.xlabel('Forward Pass')
    plt.ylabel('Tokens Routed (Running Batch)')
    plt.title(f'Layer {layer}: Running Batch Fluctuation (passes {start}-{end})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_zoomed.png', dpi=150)
    print(f"Saved {save_prefix}_zoomed.png")
    
    # Plot 3: Distribution of batch sizes per expert
    plt.figure(figsize=(14, 6))
    layer = 0
    
    data = [counts[:, layer, e] for e in range(num_experts)]
    bp = plt.boxplot(data, labels=[f'E{i}' for i in range(num_experts)])
    plt.xlabel('Expert')
    plt.ylabel('Tokens per Forward Pass')
    plt.title(f'Layer {layer}: Distribution of Running Batch Sizes')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_distribution.png', dpi=150)
    print(f"Saved {save_prefix}_distribution.png")


def main():
    # Configuration
    config = {
        "model_name": "mistralai/Mixtral-8x7B-v0.1",
        "num_forward_passes": 10,  # Number of different inputs to test
        "batch_size": 128,            # Sequences per forward pass
        "seq_length": 512,          # Tokens per sequence
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  Total tokens per forward pass: {config['batch_size'] * config['seq_length']}")
    
    # Run inference tracking
    counts, model_info = run_inference_tracking(**config)
    
    # Save raw data
    np.save("expert_batch_counts.npy", counts)
    # with open("config.json", "w") as f:
    #     json.dump({**config, **model_info}, f, indent=2)
    # print("\nSaved expert_batch_counts.npy and config.json")
    
    # # Analyze fluctuation
    # fluctuations = analyze_batch_fluctuation(counts, window_size=3)
    
    # # Save analysis
    # with open("fluctuation_results.json", "w") as f:
    #     json.dump(fluctuations[:200], f, indent=2)
    
    # # Plot
    # plot_expert_batches(counts)
    
    # print("\nDone!")


if __name__ == "__main__":
    main()
