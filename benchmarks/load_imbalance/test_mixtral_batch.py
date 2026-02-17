import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np


class ExpertTracker:
    def __init__(self, num_layers, num_experts):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.reset()
    
    def reset(self):
        self.records = []
        self.current = {}
    
    def start_forward(self):
        self.current = {}
    
    def record(self, layer_idx, expert_indices, batch_size, seq_len):
        """expert_indices: [batch_size * seq_len, top_k]"""
        top_k = expert_indices.shape[1]
        
        # Reshape to [batch, seq, top_k]
        indices_3d = expert_indices.view(batch_size, seq_len, top_k)
        
        # Per-position counts: [seq_len, num_experts]
        per_pos = torch.zeros(seq_len, self.num_experts, device=expert_indices.device)
        for pos in range(seq_len):
            for k in range(top_k):
                per_pos[pos] += torch.bincount(
                    indices_3d[:, pos, k],
                    minlength=self.num_experts
                ).float()
        
        self.current[layer_idx] = per_pos.cpu().numpy()
    
    def end_forward(self):
        if self.current:
            self.records.append(self.current)
        self.current = {}
    
    def get_batch_sizes(self):
        """Returns [num_passes, num_layers, seq_len, num_experts]"""
        num_passes = len(self.records)
        layers = sorted(self.records[0].keys())
        num_layers = len(layers)
        seq_len = self.records[0][0].shape[0]
        
        arr = np.zeros((num_passes, num_layers, seq_len, self.num_experts))
        for i, rec in enumerate(self.records):
            for j, layer in enumerate(layers):
                arr[i, j, :, :] = rec[layer]
        return arr


def patch_mixtral(model, tracker):
    """Patch Mixtral's MoE blocks to track routing"""
    
    for layer_idx, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        original_forward = moe.forward
        
        def make_hook(orig_fn, moe_ref, l_idx):
            def hooked_forward(hidden_states):
                batch, seq_len, dim = hidden_states.shape
                hidden_flat = hidden_states.view(-1, dim)
                
                router_logits = moe_ref.gate(hidden_flat)
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
                _, selected_experts = torch.topk(routing_weights, moe_ref.top_k, dim=-1)
                
                # Pass batch and seq_len info
                tracker.record(l_idx, selected_experts, batch, seq_len)
                
                return orig_fn(hidden_states)
            return hooked_forward
        
        moe.forward = make_hook(original_forward, moe, layer_idx)
    
    print(f"Patched {len(model.model.layers)} layers")


def run_inference_tracking(
    model_name="mistralai/Mixtral-8x7B-v0.1",
    num_forward_passes=20,
    batch_size=128,
    seq_length=512,
):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    top_k = model.config.num_experts_per_tok
    
    print(f"Model: {num_layers} layers, {num_experts} experts, top-{top_k} routing")
    print(f"Batch size: {batch_size}, Seq length: {seq_length}")
    print(f"At each position: {batch_size} tokens routed to experts")
    
    tracker = ExpertTracker(num_layers, num_experts)
    patch_mixtral(model, tracker)
    
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"]) > 200)
    texts = dataset["text"]
    
    print(f"\nRunning {num_forward_passes} forward passes...")
    
    with torch.no_grad():
        for i in range(num_forward_passes):
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
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_forward_passes}")
    
    counts = tracker.get_batch_sizes()
    np.save("expert_batch_sizes.npy", counts)
    
    print(f"\nShape: {counts.shape}")
    print(f"  = [passes, layers, positions, experts]")
    print(f"  = [{num_forward_passes}, {num_layers}, {seq_length}, {num_experts}]")
    
    # Save readable txt
    num_passes, num_layers, seq_len, num_experts = counts.shape
    with open("expert_batch_sizes.txt", 'w') as f:
        f.write(f"# Shape: {counts.shape} (passes, layers, positions, experts)\n")
        f.write(f"# At each position: {batch_size} tokens from batch, top-{top_k} routing\n")
        for p in range(num_passes):
            for l in range(num_layers):
                for pos in range(seq_len):
                    row = counts[p, l, pos, :]
                    f.write(f"pass={p:3d} layer={l:2d} pos={pos:4d}: " + " ".join(f"{int(x):4d}" for x in row) + "\n")
    
    print("Saved: expert_batch_sizes.npy and expert_batch_sizes.txt")


def main():
    config = {
        "model_name": "mistralai/Mixtral-8x7B-v0.1",
        "num_forward_passes": 20,
        "batch_size": 256,
        "seq_length": 512,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    run_inference_tracking(**config)


if __name__ == "__main__":
    main()