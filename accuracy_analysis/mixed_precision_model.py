from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch
import torch.nn as nn
import logging
from enum import Enum

from quant import make_quant_forward, unfuse_moe_experts, MODEL_EXPERT_CONFIG

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
    unique_experts = topk_indices.unique().tolist()
    num_to_quantize = int(len(unique_experts) * quantize_ratio)

    if num_to_quantize == 0:
        return []

    if criteria == QuantizationCriteria.RANDOM:
        perm = torch.randperm(len(unique_experts))[:num_to_quantize]
        experts_to_quantize = {unique_experts[i] for i in perm.tolist()}

    elif criteria in [QuantizationCriteria.LOWEST_WEIGHT_SUM, QuantizationCriteria.HIGHEST_WEIGHT_SUM]:
        weight_sums = torch.zeros(num_experts, device=topk_weights.device, dtype=topk_weights.dtype)
        flat_indices = topk_indices.flatten()
        flat_weights = topk_weights.flatten()
        weight_sums.scatter_add_(0, flat_indices, flat_weights)

        unique_tensor = torch.tensor(unique_experts, device=topk_weights.device)
        unique_weight_sums = weight_sums[unique_tensor]
        sorted_indices = torch.argsort(unique_weight_sums,
            descending=(criteria == QuantizationCriteria.HIGHEST_WEIGHT_SUM))
        experts_to_quantize = {unique_experts[i] for i in sorted_indices[:num_to_quantize].tolist()}

    elif criteria in [QuantizationCriteria.COLDEST_COUNT, QuantizationCriteria.HOTTEST_COUNT]:
        counts = torch.zeros(num_experts, device=topk_indices.device, dtype=torch.long)
        flat_indices = topk_indices.flatten()
        ones = torch.ones_like(flat_indices)
        counts.scatter_add_(0, flat_indices, ones)

        unique_tensor = torch.tensor(unique_experts, device=topk_indices.device)
        unique_counts = counts[unique_tensor]
        sorted_indices = torch.argsort(unique_counts,
            descending=(criteria == QuantizationCriteria.HOTTEST_COUNT))
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
            model_name="Qwen/Qwen3-30B-A3B",
            cache_dir="/models/",
            quant_modes=["nvfp4"],
            backend="cudnn",
        ):
        cache_dir = Path(cache_dir) / model_name.replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )

        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        logger.info("Model loaded.")

        self.model_type = self.model.config.model_type
        self.quant_modes = quant_modes
        self.backend = backend

        self._expert_config = MODEL_EXPERT_CONFIG.get(self.model_type)
        if self._expert_config is None:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Supported: {list(MODEL_EXPERT_CONFIG.keys())}"
            )

        # Unfuse experts for models with fused weight tensors (e.g. Qwen3)
        logger.info("Unfusing MoE experts...")
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, "experts"):
                unfuse_moe_experts(layer.mlp)

        # Pre-create quantized forward functions for all experts x all modes
        logger.info("Creating quantized forwards for modes: %s", quant_modes)
        self._quant_forwards = {}
        self._setup_quant_forwards()

        self._hook_handles = []
        self._patched_forwards = []  # [(proj_module, original_forward), ...]

    def _setup_quant_forwards(self):
        block_names = self._expert_config["block_names"]
        labels = self._expert_config["labels"]

        for mode in self.quant_modes:
            self._quant_forwards[mode] = {}
            for layer_id, layer in enumerate(self.model.model.layers):
                if not hasattr(layer.mlp, "experts"):
                    continue
                experts = layer.mlp.experts
                if not isinstance(experts, nn.ModuleList):
                    logger.warning("Layer %d experts not ModuleList, skipping.", layer_id)
                    continue
                for expert_id in range(len(experts)):
                    expert = experts[expert_id]
                    fwds = {}
                    for block_name, label in zip(block_names, labels):
                        proj = getattr(expert, block_name)
                        fwd, _ = make_quant_forward(proj, mode, backend=self.backend)
                        fwds[label] = fwd
                    self._quant_forwards[mode][(layer_id, expert_id)] = fwds
        logger.info("Quantized forward functions ready.")

    def _forward_quantized_expert(self, layer_id, expert_id, quant_mode, hidden_states):
        """Run an expert's MLP using quantized forward functions."""
        fwds = self._quant_forwards[quant_mode][(layer_id, expert_id)]
        expert = self.model.model.layers[layer_id].mlp.experts[expert_id]
        act_fn = expert.act_fn

        gate_out = fwds["gate"](hidden_states)
        up_out = fwds["up"](hidden_states)
        activated = act_fn(gate_out) * up_out
        return fwds["down"](activated)

    def register_layer_override_hooks(
            self, layer_id, quant_mode,
            quantize_ratio=0.5,
            criteria=QuantizationCriteria.RANDOM,
            custom_fn=None,
        ):
        moe_layer = self.model.model.layers[layer_id].mlp
        num_experts_per_tok = self.model.config.num_experts_per_tok
        num_experts = self.model.config.num_experts

        def moe_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_output, rest = output[0], output[1:]
            else:
                hidden_output, rest = output, None

            hidden_states = input[0]
            batch_size, seq_len, hidden_dim = hidden_states.shape

            # Re-derive routing decisions
            gate_output = module.gate(hidden_states.view(-1, hidden_dim))
            if isinstance(gate_output, tuple):
                router_logits = gate_output[0]
            else:
                router_logits = gate_output
            topk_weights, topk_indices = torch.topk(
                router_logits, num_experts_per_tok, dim=-1
            )
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

            # Flatten for vectorized processing
            flat_hidden = hidden_states.view(-1, hidden_dim)
            flat_topk_indices = topk_indices.view(-1, num_experts_per_tok)
            flat_topk_weights = topk_weights.view(-1, num_experts_per_tok)

            # Accumulate corrections: delta = quant_out - original_out
            delta = torch.zeros_like(flat_hidden)

            for expert_id in experts_to_quantize:
                mask = (flat_topk_indices == expert_id)

                if not mask.any():
                    continue

                token_indices, topk_positions = torch.where(mask)
                weights_for_expert = flat_topk_weights[token_indices, topk_positions]

                unique_token_indices, inverse_indices = torch.unique(
                    token_indices, return_inverse=True
                )
                expert_hidden = flat_hidden[unique_token_indices]

                # Original expert forward
                original_expert = module.experts[expert_id]
                original_out = original_expert(expert_hidden)

                # Quantized expert forward (real quantized GEMM)
                quant_out = self._forward_quantized_expert(
                    layer_id, expert_id, quant_mode, expert_hidden
                )

                token_delta = quant_out - original_out
                weighted_deltas = weights_for_expert.unsqueeze(1) * token_delta[inverse_indices]
                delta.index_add_(0, token_indices, weighted_deltas)

            flat_output = hidden_output.view(-1, hidden_dim)
            modified_flat = flat_output + delta
            modified_output = modified_flat.view(batch_size, seq_len, hidden_dim)

            if rest is not None:
                return (modified_output,) + rest
            else:
                return modified_output

        handle = moe_layer.register_forward_hook(moe_hook)
        self._hook_handles.append(handle)
        logger.debug("Registered layer hook for layer %d, mode %s.", layer_id, quant_mode)
        return handle

    def register_expert_override(self, layer_id, expert_id, quant_mode):
        """Monkey-patch an individual expert to use quantized forward."""
        assert (layer_id, expert_id) in self._quant_forwards[quant_mode], \
            f"No quantized forward for layer {layer_id}, expert {expert_id}."

        expert = self.model.model.layers[layer_id].mlp.experts[expert_id]
        block_names = self._expert_config["block_names"]
        labels = self._expert_config["labels"]
        fwds = self._quant_forwards[quant_mode][(layer_id, expert_id)]

        for block_name, label in zip(block_names, labels):
            proj = getattr(expert, block_name)
            self._patched_forwards.append((proj, proj.forward))
            proj.forward = fwds[label]

    def clear_hooks(self):
        """Remove all hooks and restore monkey-patched forwards."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

        for proj, orig_forward in self._patched_forwards:
            proj.forward = orig_forward
        self._patched_forwards = []
        logger.debug("Cleared all hooks and restored forwards.")

    def quantize_all_experts(self, quant_mode="nvfp4", layer_ids=None, expert_ids=None):
        """Statically patch all specified experts with quantized forwards."""
        layers = self.model.model.layers
        layer_ids = layer_ids if layer_ids is not None else range(len(layers))
        for layer_id in layer_ids:
            if not hasattr(layers[layer_id].mlp, "experts"):
                continue
            num_experts = len(layers[layer_id].mlp.experts)
            eids = expert_ids if expert_ids is not None else range(num_experts)
            for expert_id in eids:
                self.register_expert_override(layer_id, expert_id, quant_mode)
        logger.debug("Quantized all experts with mode %s.", quant_mode)

    def quantize_all_layers(
            self, quant_mode="nvfp4", quantize_ratio=0.5,
            criteria=QuantizationCriteria.RANDOM, layer_ids=None,
        ):
        """Register dynamic per-batch expert selection hooks on all layers."""
        layers = self.model.model.layers
        layer_ids = layer_ids if layer_ids is not None else range(len(layers))
        for layer_id in layer_ids:
            if not hasattr(layers[layer_id].mlp, "experts"):
                continue
            self.register_layer_override_hooks(
                layer_id, quant_mode=quant_mode,
                quantize_ratio=quantize_ratio, criteria=criteria,
            )
        logger.debug("Registered layer hooks for all layers, mode %s.", quant_mode)

    def generate(self, prompts, max_new_tokens=20):
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    model = MultiPrecisionMoEModel(
        model_name="Qwen/Qwen3-30B-A3B",
        quant_modes=["nvfp4"],
    )

    r = []
    model.quantize_all_layers(quant_mode="nvfp4", quantize_ratio=1.0)
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
