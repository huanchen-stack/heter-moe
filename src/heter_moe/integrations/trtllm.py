"""TensorRT-LLM integration for heterogeneous MoE.

Provides monkey-patch functions to intercept TRT-LLM model construction
and inject signal capture and precision dispatch:

- ``patch_moe_factory()`` — swaps ``CutlassFusedMoE`` with
  ``HeteroCutlassFusedMoE`` for heterogeneous precision execution.
- ``patch_router()`` — wraps routing methods with ``HeteroRouter`` to
  capture ``RouterSignals`` (logits, expert indices, weights, load).
- ``patch_attention()`` — attaches ``HeteroAttention`` hooks to all
  ``Attention`` modules to capture ``AttentionSignals`` (token importance).
- ``unpatch_*()`` — restores originals.

Usage::

    from heter_moe.integrations.trtllm import (
        patch_moe_factory,
        patch_router,
        patch_attention,
    )

    # Before model construction:
    patch_moe_factory(nvfp4_ratio=0.5, seed=42)
    patch_router()

    llm = LLM(model=model_dir, ...)

    # After model construction:
    attn_hooks = patch_attention(llm.model)

    # Query signals after inference:
    for router in get_active_routers():
        print(router.last_signals)
    for hook in attn_hooks:
        print(hook.last_signals)
"""

from typing import List, Optional

from torch import nn

from heter_moe.ops.attention import HeteroAttention
from heter_moe.ops.router import HeteroRouter
from heter_moe.policy.scheduler import PrecisionScheduler
from heter_moe.policy.strategies import RandomStrategy


# Module-level registry for active HeteroRouter instances
_active_routers: List[HeteroRouter] = []


# ------------------------------------------------------------------
# MoE factory patch (heterogeneous precision execution)
# ------------------------------------------------------------------


def patch_moe_factory(
    nvfp4_ratio: float = 0.5,
    seed: int = 42,
    hot_expert_ids: Optional[List[int]] = None,
) -> None:
    """Monkey-patch TRT-LLM's ``create_moe_backend`` to use ``HeteroCutlassFusedMoE``.

    Must be called **before** ``LLM()`` constructs the model.

    Args:
        nvfp4_ratio: Fraction of experts assigned to NVFP4.
        seed: Random seed for the ``RandomStrategy``.
        hot_expert_ids: Expert IDs forced to BF16 regardless of ratio.
    """
    import tensorrt_llm._torch.modules.fused_moe.create_moe.create_moe_backend as factory_module
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import (
        CutlassFusedMoE,
    )

    from heter_moe.ops.hetero_moe import HeteroCutlassFusedMoE

    _original = factory_module

    def _patched(moe_cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        if moe_cls is CutlassFusedMoE:
            num_experts = kwargs.get("num_experts", 0)

            strategy = RandomStrategy(
                nvfp4_ratio=nvfp4_ratio,
                seed=seed,
                hot_expert_ids=hot_expert_ids,
            )
            scheduler = PrecisionScheduler(
                num_experts=num_experts,
                strategy=strategy,
            )
            plan = scheduler.schedule()

            return HeteroCutlassFusedMoE(
                dispatch_plan=plan,
                **kwargs,
            )
        return _original(moe_cls, *args, **kwargs)

    factory_module = _patched


def unpatch_moe_factory() -> None:
    """Restore the original ``create_moe_backend`` factory."""
    import tensorrt_llm._torch.modules.fused_moe.create_moe as factory_module
    from importlib import reload

    reload(factory_module)


# ------------------------------------------------------------------
# Router signal capture patch
# ------------------------------------------------------------------


def patch_router() -> None:
    """Monkey-patch TRT-LLM's ``create_moe`` to wrap routing methods.

    Every ``BaseMoeRoutingMethod`` passed to ``create_moe()`` is wrapped
    with ``HeteroRouter``, which captures ``RouterSignals`` on each
    ``apply()`` call.  Wrapped instances are tracked in a module-level
    registry accessible via ``get_active_routers()``.

    Must be called **before** ``LLM()`` constructs the model.
    """
    import tensorrt_llm._torch.modules.fused_moe.create_moe as factory_module

    _original_create_moe = factory_module.create_moe

    def _patched_create_moe(
        routing_method, *args, **kwargs,
    ):  # type: ignore[no-untyped-def]
        wrapped = HeteroRouter(routing_method)
        _active_routers.append(wrapped)
        return _original_create_moe(wrapped, *args, **kwargs)

    factory_module.create_moe = _patched_create_moe


def unpatch_router() -> None:
    """Restore the original ``create_moe`` factory and clear router registry."""
    import tensorrt_llm._torch.modules.fused_moe.create_moe as factory_module
    from importlib import reload

    _active_routers.clear()
    reload(factory_module)


def get_active_routers() -> List[HeteroRouter]:
    """Return all ``HeteroRouter`` instances created by ``patch_router()``.

    Each instance exposes ``.last_signals`` with the most recent
    ``RouterSignals`` from that MoE layer.
    """
    return list(_active_routers)


# ------------------------------------------------------------------
# Attention signal capture patch
# ------------------------------------------------------------------


def patch_attention(model: nn.Module) -> List[HeteroAttention]:
    """Attach ``HeteroAttention`` signal hooks to all Attention modules.

    Walks the model's module tree and attaches forward hooks to every
    ``Attention`` instance found.  Must be called **after** model
    construction.

    Args:
        model: The constructed TRT-LLM model (e.g., ``llm.model``).

    Returns:
        List of ``HeteroAttention`` hook objects.  Each exposes
        ``.last_signals`` for the most recent ``AttentionSignals``.
        Call ``unpatch_attention()`` to detach all hooks.
    """
    from tensorrt_llm._torch.modules.attention import Attention

    hooks: List[HeteroAttention] = []
    for _name, module in model.named_modules():
        if isinstance(module, Attention):
            layer_index = getattr(module, "layer_idx", len(hooks))
            hook = HeteroAttention(module, layer_index=layer_index)
            hooks.append(hook)
    return hooks


def unpatch_attention(hooks: List[HeteroAttention]) -> None:
    """Remove all ``HeteroAttention`` hooks.

    Args:
        hooks: The list returned by ``patch_attention()``.
    """
    for hook in hooks:
        hook.remove()
    hooks.clear()
