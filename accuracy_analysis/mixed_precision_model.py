from transformers import AutoModelForCausalLM, AutoTokenizer
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
        perm = torch.randperm(len(unique_experts), device=topk_indices.device)[:num_to_quantize]
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
            model_path="Qwen/Qwen3-30B-A3B",
            quant_modes=["nvfp4"],
            backend="cudnn",
        ):
        """
        Args:
            model_path: local directory (e.g. "./models/Qwen3-30B-A3B") or
                        HuggingFace repo ID (e.g. "Qwen/Qwen3-30B-A3B").
            quant_modes: list of quant modes to pre-create ("nvfp4", "fp8", "a16w4").
            backend: FlashInfer backend for nvfp4 ("cudnn" or "cutlass").
        """
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
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
        """Monkey-patch selected experts per-batch via pre/post hooks.

        Pre-hook: run gate → select experts → patch their linear forwards.
        Model forward: runs normally, patched experts use quantized GEMM.
        Post-hook: restore original forwards.

        Each expert runs exactly once (no delta recomputation).
        """
        moe_layer = self.model.model.layers[layer_id].mlp
        num_experts_per_tok = self.model.config.num_experts_per_tok
        num_experts = self.model.config.num_experts
        block_names = self._expert_config["block_names"]
        labels = self._expert_config["labels"]

        # Per-call state shared between pre-hook and post-hook
        patched_state = []  # [(proj, original_forward), ...]

        def pre_hook(module, input):
            hidden_states = input[0]
            hidden_dim = hidden_states.shape[-1]

            # Run gate to get routing decisions
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
                return

            # Patch selected experts' linear forwards
            for expert_id in experts_to_quantize:
                fwds = self._quant_forwards[quant_mode][(layer_id, expert_id)]
                expert = module.experts[expert_id]
                for block_name, label in zip(block_names, labels):
                    proj = getattr(expert, block_name)
                    patched_state.append((proj, proj.forward))
                    proj.forward = fwds[label]

        def post_hook(module, input, output):
            # Restore all patched forwards
            for proj, orig_fwd in patched_state:
                proj.forward = orig_fwd
            patched_state.clear()
            return output

        pre_handle = moe_layer.register_forward_pre_hook(pre_hook)
        post_handle = moe_layer.register_forward_hook(post_hook)
        self._hook_handles.extend([pre_handle, post_handle])
        logger.debug("Registered layer hooks for layer %d, mode %s.", layer_id, quant_mode)

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

    def quantize_all_experts_prequantized(self, quant_mode="nvfp4", layer_ids=None):
        """Pre-quantize all expert weights and free BF16 originals.

        This is irreversible — original weights are deleted to save GPU memory.
        Use only when quantize_ratio=1.0 (all experts quantized).
        clear_hooks() will NOT restore original weights after this call.
        """
        block_names = self._expert_config["block_names"]
        labels = self._expert_config["labels"]
        layers = self.model.model.layers
        layer_ids = layer_ids if layer_ids is not None else range(len(layers))
        cleanups = []

        for layer_id in layer_ids:
            if not hasattr(layers[layer_id].mlp, "experts"):
                continue
            experts = layers[layer_id].mlp.experts
            if not isinstance(experts, nn.ModuleList):
                continue
            for expert_id in range(len(experts)):
                expert = experts[expert_id]
                for block_name, label in zip(block_names, labels):
                    proj = getattr(expert, block_name)
                    fwd, cleanup = make_quant_forward(
                        proj, quant_mode, backend=self.backend, prequantize=True,
                    )
                    self._patched_forwards.append((proj, proj.forward))
                    proj.forward = fwd
                    if cleanup is not None:
                        cleanups.append(cleanup)

        for cleanup in cleanups:
            cleanup()

        torch.cuda.empty_cache()
        logger.info(
            "Pre-quantized all experts with mode %s. BF16 weights freed.", quant_mode,
        )

    def quantize_all_layers(
            self, quant_mode="nvfp4", quantize_ratio=0.5,
            criteria=QuantizationCriteria.RANDOM, layer_ids=None,
            prequantize: bool = False,
        ):
        """Register dynamic per-batch expert selection hooks on all layers.

        When quantize_ratio=1.0 and prequantize=True, skips hook-based dynamic
        selection and uses pre-quantized static forwards instead. This frees
        BF16 weights and is irreversible.
        """
        if quantize_ratio >= 1.0 and prequantize:
            logger.info("ratio=1.0 + prequantize: using static pre-quantized path")
            self.quantize_all_experts_prequantized(
                quant_mode=quant_mode, layer_ids=layer_ids,
            )
            return

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
        model_path="./models/Qwen3-30B-A3B",
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
