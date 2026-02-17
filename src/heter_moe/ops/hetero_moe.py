"""Heterogeneous Precision MoE: two grouped GEMMs instead of one.

The ONLY change vs ``CutlassFusedMoE``: ``run_moe()`` calls
``torch.ops.trtllm.fused_moe()`` TWICE (once per precision group)
instead of ONCE, then sums the results.

Everything else — weights, routing, quantization, communication —
is inherited from ``CutlassFusedMoE`` unchanged.
"""

from typing import List, Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import (
    CutlassFusedMoE,
)
from tensorrt_llm._torch.modules.fused_moe.routing import BaseMoeRoutingMethod

from heter_moe.policy.dispatch_plan import DispatchPlan


def create_subset_weights(
    full_weights: torch.Tensor,
    expert_ids: List[int],
) -> torch.Tensor:
    """Create a contiguous subset of expert weights.

    Args:
        full_weights: Full weight tensor ``[num_experts, ...]``.
        expert_ids: Expert IDs to include in the subset.

    Returns:
        Contiguous weight tensor ``[len(expert_ids), ...]``.
    """
    return full_weights[expert_ids].contiguous()


class HeteroCutlassFusedMoE(CutlassFusedMoE):
    """``CutlassFusedMoE`` with two grouped GEMMs: one for NVFP4 experts,
    one for BF16 experts.

    Inherits ALL behaviour from ``CutlassFusedMoE``.  The only override
    is ``run_moe()``, which:

    1. Builds per-group remap tables (cached after first call).
    2. Subsets weights per group (cached after first call).
    3. Masks routing: keeps ``[N, top_k]`` shape, zeros scales for
       non-group experts, maps expert IDs to local indices.
    4. Calls ``torch.ops.trtllm.fused_moe()`` twice.
    5. Sums outputs.

    Args:
        dispatch_plan: A ``DispatchPlan`` produced by
            ``PrecisionScheduler.schedule()``.
        routing_method: TRT-LLM routing method instance.
        num_experts: Total number of experts.
        hidden_size: Hidden dimension.
        intermediate_size: FFN intermediate dimension.
        model_config: TRT-LLM model configuration.
        **kwargs: Forwarded to ``CutlassFusedMoE.__init__``.
    """

    def __init__(
        self,
        *,
        dispatch_plan: DispatchPlan,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        model_config: ModelConfig = ModelConfig(),
        **kwargs,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            model_config=model_config,
            **kwargs,
        )

        self.dispatch_plan = dispatch_plan

        # Lazy-init caches (populated on first run_moe call)
        self._group_cache_built = False
        self._nvfp4_remap: Optional[torch.Tensor] = None
        self._bf16_remap: Optional[torch.Tensor] = None
        self._nvfp4_w3_w1: Optional[torch.Tensor] = None
        self._nvfp4_w2: Optional[torch.Tensor] = None
        self._bf16_w3_w1: Optional[torch.Tensor] = None
        self._bf16_w2: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _build_group_cache(self, device: torch.device) -> None:
        """Build remap tables and subset weights.  Called once."""
        plan = self.dispatch_plan
        num_nvfp4 = plan.num_nvfp4
        num_bf16 = plan.num_bf16

        # Remap: global expert id -> local index within group.
        # Out-of-group entries map to num_group_experts (sentinel).
        nvfp4_remap = torch.full(
            (self.num_experts,), num_nvfp4,
            device=device, dtype=torch.int32,
        )
        nvfp4_ids = torch.tensor(
            plan.nvfp4_expert_ids, device=device, dtype=torch.long,
        )
        nvfp4_remap[nvfp4_ids] = torch.arange(
            num_nvfp4, device=device, dtype=torch.int32,
        )
        self._nvfp4_remap = nvfp4_remap

        bf16_remap = torch.full(
            (self.num_experts,), num_bf16,
            device=device, dtype=torch.int32,
        )
        bf16_ids = torch.tensor(
            plan.bf16_expert_ids, device=device, dtype=torch.long,
        )
        bf16_remap[bf16_ids] = torch.arange(
            num_bf16, device=device, dtype=torch.int32,
        )
        self._bf16_remap = bf16_remap

        # Subset weights
        weight_dtype = self.w3_w1_weight.dtype
        self._nvfp4_w3_w1 = create_subset_weights(
            self.w3_w1_weight, plan.nvfp4_expert_ids,
        ).view(weight_dtype)
        self._nvfp4_w2 = create_subset_weights(
            self.w2_weight, plan.nvfp4_expert_ids,
        ).view(weight_dtype)
        self._bf16_w3_w1 = create_subset_weights(
            self.w3_w1_weight, plan.bf16_expert_ids,
        ).view(weight_dtype)
        self._bf16_w2 = create_subset_weights(
            self.w2_weight, plan.bf16_expert_ids,
        ).view(weight_dtype)

        self._group_cache_built = True

    # ------------------------------------------------------------------
    # run_moe override — the ONLY functional change
    # ------------------------------------------------------------------

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        is_sf_swizzled: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: Optional[bool] = None,
    ) -> torch.Tensor:
        """Two grouped GEMMs instead of one.

        Keeps ``[N, top_k]`` routing shape.  For each group:

        - Remap expert IDs to local indices (sentinel for non-group).
        - Zero out scales for non-group experts.
        - Call ``torch.ops.trtllm.fused_moe()`` with subset weights.

        Sum the two outputs.
        """
        if not self._group_cache_built:
            self._build_group_cache(x.device)

        plan = self.dispatch_plan
        num_nvfp4 = plan.num_nvfp4
        num_bf16 = plan.num_bf16

        if enable_alltoall is None:
            enable_alltoall = self.enable_alltoall

        # --- NVFP4 group ---
        nvfp4_experts = self._nvfp4_remap[
            token_selected_experts.long()
        ].to(torch.int32)
        nvfp4_valid = nvfp4_experts < num_nvfp4
        nvfp4_scales = torch.where(
            nvfp4_valid,
            token_final_scales,
            torch.zeros_like(token_final_scales),
        )

        out_nvfp4 = torch.ops.trtllm.fused_moe(
            x,
            nvfp4_experts,
            nvfp4_scales,
            self._nvfp4_w3_w1,
            self.w3_w1_bias,
            self._nvfp4_w2,
            self.w2_bias,
            output_dtype,
            quant_scales=self.quant_scales,
            input_sf=x_sf,
            swizzled_input_sf=is_sf_swizzled,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=self.has_deepseek_fp8_block_scales,
            use_w4_group_scaling=self.has_w4afp8 or self.has_w4a16_mxfp4,
            use_int8_woq_per_channel=self.has_int8_woq_per_channel,
            use_mxfp8_act_scaling=self.has_w4a8_mxfp4_mxfp8,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=None,
        )[0]

        # --- BF16 group ---
        bf16_experts = self._bf16_remap[
            token_selected_experts.long()
        ].to(torch.int32)
        bf16_valid = bf16_experts < num_bf16
        bf16_scales = torch.where(
            bf16_valid,
            token_final_scales,
            torch.zeros_like(token_final_scales),
        )

        out_bf16 = torch.ops.trtllm.fused_moe(
            x,
            bf16_experts,
            bf16_scales,
            self._bf16_w3_w1,
            self.w3_w1_bias,
            self._bf16_w2,
            self.w2_bias,
            output_dtype,
            quant_scales=self.quant_scales,
            input_sf=x_sf,
            swizzled_input_sf=is_sf_swizzled,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=self.has_deepseek_fp8_block_scales,
            use_w4_group_scaling=self.has_w4afp8 or self.has_w4a16_mxfp4,
            use_int8_woq_per_channel=self.has_int8_woq_per_channel,
            use_mxfp8_act_scaling=self.has_w4a8_mxfp4_mxfp8,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=None,
        )[0]

        # --- Combine ---
        if moe_output is not None:
            moe_output.copy_(out_nvfp4 + out_bf16)
            return moe_output
        return out_nvfp4 + out_bf16
