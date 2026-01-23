from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch
import re
from enum import Enum

from quantization import create_fake_quantized_module, Precision


class QuantizationCriteria(Enum):
    RANDOM = "random"
    LOWEST_WEIGHT_SUM = "lowest_weight_sum"
    HIGHEST_WEIGHT_SUM = "highest_weight_sum"
    COLDEST_COUNT = "coldest_count"
    HOTTEST_COUNT = "hottest_count"
    CUSTOM = "custom"

def select_experts_to_quantize(
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        quantize_ratio: float,
        criteria: QuantizationCriteria = QuantizationCriteria.RANDOM,
        custom_fn=None
    ):

    unique_experts = topk_indices.unique().tolist()
    num_to_quantize = int(len(unique_experts) * quantize_ratio)
    
    if num_to_quantize == 0:
        return []
    
    if criteria == QuantizationCriteria.RANDOM:
        perm = torch.randperm(len(unique_experts))[:num_to_quantize]
        experts_to_quantize = {unique_experts[i] for i in perm.tolist()}
    
    elif criteria in [QuantizationCriteria.LOWEST_WEIGHT_SUM, QuantizationCriteria.HIGHEST_WEIGHT_SUM]:
        # Sum routing weights for each expert across batch
        weight_sums = torch.zeros(num_experts, device=topk_weights.device)
        flat_indices = topk_indices.flatten()  # (batch * seq_len * top_k,)
        flat_weights = topk_weights.flatten()  # (batch * seq_len * top_k,)
        weight_sums.scatter_add_(0, flat_indices, flat_weights)
        
        # Select experts with lowest weight sum (only from those actually selected)
        unique_tensor = torch.tensor(unique_experts, device=topk_weights.device)
        unique_weight_sums = weight_sums[unique_tensor]
        sorted_indices = torch.argsort(unique_weight_sums,
            descending=(criteria == QuantizationCriteria.HIGHEST_WEIGHT_SUM))  # ascending
        experts_to_quantize = {unique_experts[i] for i in sorted_indices[:num_to_quantize].tolist()}
    
    elif criteria in [QuantizationCriteria.COLDEST_COUNT, QuantizationCriteria.HOTTEST_COUNT]:
        # Count how many times each expert is selected across batch
        counts = torch.zeros(num_experts, device=topk_indices.device, dtype=torch.long)
        flat_indices = topk_indices.flatten()
        ones = torch.ones_like(flat_indices)
        counts.scatter_add_(0, flat_indices, ones)
        
        # Select experts with lowest count (coldest)
        unique_tensor = torch.tensor(unique_experts, device=topk_indices.device)
        unique_counts = counts[unique_tensor]
        sorted_indices = torch.argsort(unique_counts,
            descending=(criteria == QuantizationCriteria.HOTTEST_COUNT))  # ascending
        experts_to_quantize = {unique_experts[i] for i in sorted_indices[:num_to_quantize].tolist()}
    
    elif criteria == QuantizationCriteria.CUSTOM:
        if custom_fn is None:
            raise ValueError("custom_fn required when criteria is CUSTOM")
        experts_to_quantize = custom_fn(topk_indices, topk_weights)
    
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    
    return list(experts_to_quantize)

class MultiPrecisionMoEModel:
    def __init__(
            self, 
            model_name="allenai/OLMoE-1B-7B-0924", 
            cache_dir="/models/",
            quantization_dtypes=[Precision.MXFP8],
            quantized_cache_dir="/models/quantized_experts/",
            force_quantization=True,
            autostore_quantized_experts=False
        ):

        cache_dir = Path(cache_dir) / model_name.replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantized_cache_dir = Path(quantized_cache_dir) / model_name.replace("/", "_")
        quantized_cache_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )

        print("Downloading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        print("Done!")
        
        self.quantization_dtypes = quantization_dtypes

        print("Quantizing experts... to precisions:", 
              [p.value for p in quantization_dtypes])

        self.quantized_experts = None

        if not force_quantization:
            self._load_quantized_experts(quantized_cache_dir)
        if not self.quantized_experts:
            self._quantize_experts()
        if autostore_quantized_experts:
            self._save_quantized_experts(quantized_cache_dir)
        
        self._hook_handles = []

    def _quantize_experts(self):
        self.quantized_experts = {}

        for precision in self.quantization_dtypes:
            self.quantized_experts[precision.value] = {}

            for name, module in self.model.named_modules():
                if not ("experts" in name and "_proj" in name):
                    continue
                
                match = re.search(r'layers\.(\d+)\..*experts\.(\d+)\.(\w+_proj)', name)
                assert match is not None, f"Could not parse: {name}"

                layer_id = int(match.group(1))
                expert_id = int(match.group(2))
                if (layer_id, expert_id) not in self.quantized_experts[precision.value]:
                    self.quantized_experts[precision.value][(layer_id, expert_id)] = {}
                proj_type = str(match.group(3))
                quant_linear = create_fake_quantized_module(
                    module, precision=precision
                )

                self.quantized_experts[precision.value][(layer_id, expert_id)][proj_type] = quant_linear
        print("Quantization of experts complete.")

    def _load_quantized_experts(self, quantized_cache_dir: Path):
        quantized_experts_path = quantized_cache_dir / "quantized_experts.pt"
        if quantized_experts_path.exists():
            self.quantized_experts = torch.load(quantized_experts_path)
            print("Loaded quantized experts from cache.")
        else:
            print("No cached quantized experts found.")
    
    def _save_quantized_experts(self, quantized_cache_dir: Path):
        torch.save(
            self.quantized_experts, 
            quantized_cache_dir / "quantized_experts.pt"
        )
        print("Saved quantized experts to cache.")

    def _forward_quantized_expert(
            self, layer_id: int, expert_id: int, precision: Precision, 
            hidden_states
        ):
        key = (layer_id, expert_id)
        quant_expert = self.quantized_experts[precision.value][key]

        gate_proj, up_proj, down_proj = (
            quant_expert["gate_proj"],
            quant_expert["up_proj"],
            quant_expert["down_proj"]
        )
        gate_out = gate_proj(hidden_states)
        up_out = up_proj(hidden_states)

        assert isinstance(self.model.model.layers[0].mlp.experts[0].act_fn, torch.nn.SiLU), \
            f"Expected SiLU activation, got {type(self.model.model.layers[0].mlp.experts[0].act_fn)}"
        activated = torch.nn.functional.silu(gate_out) * up_out
        
        output = down_proj(activated)
        return output

    def register_layer_override_hooks(
            self, layer_id: int, precision: Precision,
            quantize_ratio: float=0.5,
            criteria: QuantizationCriteria=QuantizationCriteria.RANDOM,
            custom_fn=None,
            experts_to_quantize=None
        ):
        moe_layer = self.model.model.layers[layer_id].mlp
        num_experts_per_tok = self.model.config.num_experts_per_tok
        num_experts = self.model.config.num_experts
        
        def moe_hook(module, input, output):
            # Handle tuple output
            if isinstance(output, tuple):
                hidden_output, rest = output[0], output[1:]
            else:
                hidden_output, rest = output, None
            
            hidden_states = input[0]  # (batch, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Get router decisions
            gate = module.gate
            router_logits = gate(hidden_states)  # (batch, seq_len, num_experts)
            topk_weights, topk_indices = torch.topk(router_logits, num_experts_per_tok, dim=-1)
            topk_weights = torch.softmax(topk_weights, dim=-1)
            
            if not experts_to_quantize:
                experts_to_quantize = select_experts_to_quantize(
                    topk_indices=topk_indices,
                    topk_weights=topk_weights,
                    quantize_ratio=quantize_ratio,
                    criteria=criteria,
                    num_experts=num_experts,
                    custom_fn=custom_fn,
                )
            
            if not experts_to_quantize:
                return output
            
            # Modify output for quantized experts
            modified_output = hidden_output.clone()
            
            for b in range(batch_size):
                for s in range(seq_len):   # USING DOUBLE FOR LOOP FOR FUTURE SELECTION WITHIN SEQ
                    token_topk_indices = topk_indices[b, s, :]  # (top_k,)
                    token_topk_weights = topk_weights[b, s, :]  # (top_k,)
                    
                    for pos in range(num_experts_per_tok):
                        expert_id = token_topk_indices[pos].item()
                        
                        if expert_id not in experts_to_quantize:
                            print(f"Skipping expert {expert_id}: Not in quantization list.")
                            continue
                        
                        expert_weight = token_topk_weights[pos]
                        token_hidden = hidden_states[b:b+1, s:s+1, :]  # (1, 1, hidden_dim)
                        
                        original_expert = module.experts[expert_id]
                        original_out = original_expert(token_hidden)
                        
                        quant_out = self._forward_quantized_expert(
                            layer_id, expert_id, precision, token_hidden
                        )
                        
                        modified_output[b, s, :] = (
                            modified_output[b, s, :]
                            - expert_weight * original_out[0, 0, :]
                            + expert_weight * quant_out[0, 0, :]
                        )
            
            # Return in same format
            if rest is not None:
                return (modified_output,) + rest
            else:
                return modified_output
        
        handle = moe_layer.register_forward_hook(moe_hook)
        self._hook_handles.append(handle)
        print(f"Registered layer override hook for layer {layer_id} with precision {precision.value}.")
        
        return handle

    def register_all_layers(
            self, precision: Precision=Precision.MXFP8, 
            layer_hooks=True,
            expert_hooks=False,
            layer_ids=None, expert_ids=None
        ):
        layer_ids = layer_ids if layer_ids is not None else range(len(self.model.model.layers))
        layer_hooks = not expert_hooks
        if layer_hooks:
            for layer_id in layer_ids:
                self.register_layer_override_hooks(layer_id, precision)
            print("Registered all layer override hooks.")
        if expert_hooks:
            for layer_id in layer_ids:
                num_experts = self.model.config.num_experts
                expert_ids = expert_ids if expert_ids is not None else range(num_experts)
                for expert_id in expert_ids:
                    self.register_expert_override_hook(layer_id, expert_id, precision)
            print("Registered all expert override hooks.")

    def register_expert_override_hook(self, layer_id: int, expert_id: int, precision: Precision):
        key = (layer_id, expert_id)
        assert key in self.quantized_experts[precision.value], \
            f"Quantized expert for layer {layer_id}, expert {expert_id} not found."

        expert_module = self.model.model.layers[layer_id].mlp.experts[expert_id]

        def expert_hook(module, input, output):
            return self._forward_quantized_expert(
                layer_id, expert_id, precision, input[0]
            )

        handle = expert_module.register_forward_hook(expert_hook)
        self._hook_handles.append(handle)
        print(f"Registered quantized expert hook for layer {layer_id}, expert {expert_id}.")
        
        return handle

    def clear_hooks(self):
        if hasattr(self, "_hook_handles") and self._hook_handles:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles = []
            print("Cleared all registered hooks.")
        else:
            print("Default fallback to clear hooks from modules...")
            for _, module in self.model.named_modules():
                if hasattr(module, "_forward_hooks"):
                    module._forward_hooks.clear()
                if hasattr(module, "_backward_hooks"):
                    module._backward_hooks.clear()
                if hasattr(module, "_forward_pre_hooks"):
                    module._forward_pre_hooks.clear()
            print("Done.")

if __name__ == "__main__":
    model = MultiPrecisionMoEModel(
        model_name="allenai/OLMoE-1B-7B-0924",
        quantization_dtypes=[Precision.MXFP8, Precision.MXFP4]
    )
    print("Model and quantized experts ready.")
    sample_layer, sample_expert = 0, 0
    print(f"Sample quantized expert for layer {sample_layer}, expert {sample_expert}:")
    print(model.quantized_experts[Precision.MXFP8.value][(sample_layer, sample_expert)]["down_proj"].weight)
    print(model.quantized_experts[Precision.MXFP4.value][(sample_layer, sample_expert)]["down_proj"].weight)
    print("Done.")

    # TODO:
    #   1. Add more tests
    #   2. Test with Qwen