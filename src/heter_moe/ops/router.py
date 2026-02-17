"""Heterogeneous Router: wraps TRT-LLM routing to capture signals.

Similar to ``HeteroCutlassFusedMoE`` in ``hetero_moe.py``, this module
wraps an existing TRT-LLM routing method.  The ONLY addition: every
``apply()`` call records the routing decisions as ``RouterSignals``
for downstream consumption by the precision scheduler.

Everything else — actual routing logic, top-k selection, normalization —
is delegated to the wrapped inner method unchanged.
"""

from typing import Optional

import torch

from tensorrt_llm._torch.modules.fused_moe.routing import (
    BaseMoeRoutingMethod,
    RoutingMethodType,
)

from heter_moe.signals.router import RouterSignals


class HeteroRouter(BaseMoeRoutingMethod):
    """Wraps any ``BaseMoeRoutingMethod`` to capture ``RouterSignals``.

    The wrapper delegates all routing logic to *inner* and records
    ``router_logits``, ``expert_indices``, ``expert_weights``, and
    ``expert_load_counts`` after each ``apply()`` call.

    Args:
        inner: The original TRT-LLM routing method to wrap.
    """

    def __init__(self, inner: BaseMoeRoutingMethod):
        super().__init__()
        self._inner = inner
        self._last_signals: Optional[RouterSignals] = None

    # ------------------------------------------------------------------
    # Routing delegation
    # ------------------------------------------------------------------

    def apply(
        self, router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate to inner routing method and capture signals.

        Args:
            router_logits: Raw router output ``[num_tokens, num_experts]``.

        Returns:
            Same ``(token_selected_experts, token_final_scales)`` as
            the inner method.
        """
        token_selected_experts, token_final_scales = self._inner.apply(
            router_logits,
        )

        # Compute per-expert load counts via scatter_add
        num_experts = router_logits.shape[-1]
        expert_load_counts = torch.zeros(
            num_experts,
            device=token_selected_experts.device,
            dtype=torch.int32,
        )
        expert_load_counts.scatter_add_(
            0,
            token_selected_experts.flatten().long(),
            torch.ones(
                token_selected_experts.numel(),
                device=token_selected_experts.device,
                dtype=torch.int32,
            ),
        )

        self._last_signals = RouterSignals(
            router_logits=router_logits.detach(),
            expert_indices=token_selected_experts.detach(),
            expert_weights=(
                token_final_scales.detach()
                if token_final_scales is not None
                else None
            ),
            expert_load_counts=expert_load_counts,
        )

        return token_selected_experts, token_final_scales

    # ------------------------------------------------------------------
    # Interface delegation
    # ------------------------------------------------------------------

    def get_experts_per_token(self) -> int:
        """Delegate to inner routing method."""
        return self._inner.get_experts_per_token()

    @property
    def experts_per_token(self) -> int:
        """Delegate to inner routing method."""
        return self._inner.experts_per_token

    @property
    def routing_method_type(self) -> RoutingMethodType:
        """Delegate to inner routing method."""
        return self._inner.routing_method_type

    # ------------------------------------------------------------------
    # Signal access
    # ------------------------------------------------------------------

    @property
    def last_signals(self) -> Optional[RouterSignals]:
        """Most recent ``RouterSignals``, or ``None`` if not yet called."""
        return self._last_signals

    # ------------------------------------------------------------------
    # Transparent attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Forward unknown attributes to the inner routing method.

        This ensures attributes like ``top_k``, ``output_dtype``, etc.
        remain accessible through the wrapper without explicit
        delegation for every possible routing subclass field.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._inner, name)
