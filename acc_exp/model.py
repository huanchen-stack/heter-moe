from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch
import re
import logging
from enum import Enum

from quantization import Precision, quantized_module, dequantize_module

logger = logging.getLogger(__name__)


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
    # print(topk_indices)
    # print(topk_weights)

    unique_experts = topk_indices.unique().tolist()
    num_to_quantize = int(len(unique_experts) * quantize_ratio)
    
    if num_to_quantize == 0:
        return []
    
    if criteria == QuantizationCriteria.RANDOM:
        perm = torch.randperm(len(unique_experts))[:num_to_quantize]
        experts_to_quantize = {unique_experts[i] for i in perm.tolist()}
    
    elif criteria in [QuantizationCriteria.LOWEST_WEIGHT_SUM, QuantizationCriteria.HIGHEST_WEIGHT_SUM]:
        # Sum routing weights for each expert across batch
        weight_sums = torch.zeros(num_experts, device=topk_weights.device, dtype=topk_weights.dtype)
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
    
    # print(f"Selected experts to quantize: {experts_to_quantize}")
    return list(experts_to_quantize)

class MultiPrecisionMoEModel:
    def __init__(
            self, 
            model_name="allenai/OLMoE-1B-7B-0924", 
            cache_dir="/models/",
            quantization_dtypes=[Precision.MXFP8],
            quantized_cache_dir="/models/quantized_experts/",
            decoding_only=True,
            fake_quantize=True,
            offload=True,
            force_quantization=True,
            autostore_quantized_experts=False
        ):

        cache_dir = Path(cache_dir) / model_name.replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantized_cache_dir = Path(quantized_cache_dir) / model_name.replace("/", "_")
        quantized_cache_dir.mkdir(parents=True, exist_ok=True)

        self.decoding_only = decoding_only
        self.fake_quantize = fake_quantize
        self.offload = offload

        logger.info("Downloading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )

        logger.info("Downloading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        logger.info("Model loaded.")
        
        self.quantization_dtypes = quantization_dtypes

        logger.info("Quantizing experts to precisions: %s",
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
                quant_linear = quantized_module(
                    module, precision=precision, fake=self.fake_quantize
                )
                quant_linear.to(device="cpu" if self.offload else "cuda", non_blocking=True)

                self.quantized_experts[precision.value][(layer_id, expert_id)][proj_type] = quant_linear
        logger.info("Quantization of experts complete.")

    def _load_quantized_experts(self, quantized_cache_dir: Path):
        quantized_experts_path = quantized_cache_dir / "quantized_experts.pt"
        if quantized_experts_path.exists():
            self.quantized_experts = torch.load(quantized_experts_path, weights_only=False)
            logger.info("Loaded quantized experts from cache.")
        else:
            logger.debug("No cached quantized experts found.")
    
    def _save_quantized_experts(self, quantized_cache_dir: Path):
        torch.save(
            self.quantized_experts, 
            quantized_cache_dir / "quantized_experts.pt"
        )
        logger.info("Saved quantized experts to cache.")

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
        if self.offload:
            gate_proj.to(device=hidden_states.device)
            up_proj.to(device=hidden_states.device)
            down_proj.to(device=hidden_states.device)
        if not self.fake_quantize:
            gate_proj = dequantize_module(gate_proj)
            up_proj = dequantize_module(up_proj)
            down_proj = dequantize_module(down_proj)

        gate_out = gate_proj(hidden_states)
        up_out = up_proj(hidden_states)

        act_fn = self.model.model.layers[0].mlp.experts[0].act_fn
        activated = act_fn(gate_out) * up_out
        
        output = down_proj(activated)

        if self.offload:
            gate_proj.to(device="cpu", non_blocking=True)
            up_proj.to(device="cpu", non_blocking=True)
            down_proj.to(device="cpu", non_blocking=True)

        return output

    def register_layer_override_hooks(
            self, layer_id: int, precision: Precision,
            quantize_ratio: float=0.5,
            criteria: QuantizationCriteria=QuantizationCriteria.RANDOM,
            custom_fn=None,
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

            if self.decoding_only and seq_len == 0:
                return output

            # Get router decisions
            gate = module.gate
            router_logits = gate(hidden_states)  # (batch, seq_len, num_experts)
            topk_weights, topk_indices = torch.topk(router_logits, num_experts_per_tok, dim=-1)
            topk_weights = torch.softmax(topk_weights, dim=-1)

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

            # Flatten for vectorized processing: (batch * seq_len, hidden_dim)
            flat_hidden = hidden_states.view(-1, hidden_dim)
            # (batch * seq_len, num_experts_per_tok)
            flat_topk_indices = topk_indices.view(-1, num_experts_per_tok)
            flat_topk_weights = topk_weights.view(-1, num_experts_per_tok)
            num_tokens = flat_hidden.shape[0]

            # Accumulate corrections: delta = quant_out - original_out
            delta = torch.zeros_like(flat_hidden)  # (batch * seq_len, hidden_dim)

            # Process each quantized expert in a batched manner
            # Loop over ~16 experts instead of ~16K tokens
            for expert_id in experts_to_quantize:
                # Find all (token_idx, topk_pos) where this expert is selected
                mask = (flat_topk_indices == expert_id)  # (num_tokens, num_experts_per_tok)

                if not mask.any():
                    continue

                # Get token indices and their corresponding weights
                token_indices, topk_positions = torch.where(mask)
                weights_for_expert = flat_topk_weights[token_indices, topk_positions]  # (num_matches,)

                # Get unique token indices for batched forward pass
                unique_token_indices, inverse_indices = torch.unique(token_indices, return_inverse=True)
                expert_hidden = flat_hidden[unique_token_indices]  # (num_unique_tokens, hidden_dim)

                # Batched forward through original expert
                original_expert = module.experts[expert_id]
                original_out = original_expert(expert_hidden)  # (num_unique_tokens, hidden_dim)

                # Batched forward through quantized expert
                quant_out = self._forward_quantized_expert(
                    layer_id, expert_id, precision, expert_hidden
                )  # (num_unique_tokens, hidden_dim)

                # Compute delta per unique token
                token_delta = quant_out - original_out  # (num_unique_tokens, hidden_dim)

                # Map back: each match contributes weight * delta to its token
                # inverse_indices maps each match back to its unique token index
                weighted_deltas = weights_for_expert.unsqueeze(1) * token_delta[inverse_indices]  # (num_matches, hidden_dim)

                # Scatter-add weighted deltas back to original token positions
                delta.index_add_(0, token_indices, weighted_deltas)

            # Apply accumulated delta to output
            flat_output = hidden_output.view(-1, hidden_dim)
            modified_flat = flat_output + delta
            modified_output = modified_flat.view(batch_size, seq_len, hidden_dim)

            # Return in same format
            if rest is not None:
                return (modified_output,) + rest
            else:
                return modified_output
        
        handle = moe_layer.register_forward_hook(moe_hook)
        self._hook_handles.append(handle)
        logger.debug("Registered layer override hook for layer %d with precision %s.", layer_id, precision.value)
        
        return handle

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
        logger.debug("Registered quantized expert hook for layer %d, expert %d.", layer_id, expert_id)
        
        return handle

    def clear_hooks(self):
        if hasattr(self, "_hook_handles") and self._hook_handles:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles = []
            logger.debug("Cleared all registered hooks.")
        else:
            logger.debug("Default fallback to clear hooks from modules...")
            for _, module in self.model.named_modules():
                if hasattr(module, "_forward_hooks"):
                    module._forward_hooks.clear()
                if hasattr(module, "_backward_hooks"):
                    module._backward_hooks.clear()
                if hasattr(module, "_forward_pre_hooks"):
                    module._forward_pre_hooks.clear()
            logger.debug("Hooks cleared.")

    def quantize_all_experts(
            self, precision: Precision=Precision.MXFP8,
            layer_ids=None, expert_ids=None
        ):
        layer_ids = layer_ids if layer_ids is not None else range(len(self.model.model.layers))
        for layer_id in layer_ids:
            num_experts = self.model.config.num_experts
            expert_ids = expert_ids if expert_ids is not None else range(num_experts)
            for expert_id in expert_ids:
                self.register_expert_override_hook(layer_id, expert_id, precision)
        logger.debug("Registered all expert override hooks.")

    def quantize_all_layers(
            self, precision: Precision=Precision.MXFP8, quantize_ratio: float=0.5,
            criteria=QuantizationCriteria.RANDOM, layer_ids=None, expert_ids=None
        ):
        layer_ids = layer_ids if layer_ids is not None else range(len(self.model.model.layers))
        for layer_id in layer_ids:
            self.register_layer_override_hooks(
                layer_id, precision=precision, 
                quantize_ratio=quantize_ratio, criteria=criteria)
        logger.debug("Registered all layer override hooks.")

    def generate(self, prompts, max_new_tokens=20):
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    model = MultiPrecisionMoEModel(
        # model_name="Qwen/Qwen3-30B-A3B",
        quantization_dtypes=[
            Precision.MXFP8, 
            Precision.MXFP4,
            Precision.NVFP4
        ],
        offload=False,
        fake_quantize=False,
        decoding_only=True,
        force_quantization=True,
        autostore_quantized_experts=True
    )

    r = []
    model.quantize_all_layers(precision=Precision.MXFP8, quantize_ratio=1.0)
    r += model.generate([
            "The quick brown fox", 
            "Once upon a time in a land far away"
        ], max_new_tokens=20)
    model.clear_hooks()
    model.quantize_all_layers(precision=Precision.MXFP4, quantize_ratio=1.0)
    r += model.generate([
            "The quick brown fox", 
            "Once upon a time in a land far away"
        ], max_new_tokens=20)
    model.clear_hooks()
    model.quantize_all_layers(precision=Precision.NVFP4, quantize_ratio=1.0)
    r += model.generate([
            "The quick brown fox", 
            "Once upon a time in a land far away"
        ], max_new_tokens=20)
    model.clear_hooks()
    r += model.generate([
            "The quick brown fox", 
            "Once upon a time in a land far away"
        ], max_new_tokens=20)
    
    for res in r:
        print("=======")
        print(res)
        print("=======")
        print()
