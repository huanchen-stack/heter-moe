"""heter-moe: Heterogeneous Precision Mixture-of-Experts."""

# Ops
from heter_moe.ops.attention import HeteroAttention
from heter_moe.ops.hetero_moe import HeteroCutlassFusedMoE, create_subset_weights
from heter_moe.ops.router import HeteroRouter

# Policy
from heter_moe.policy.dispatch_plan import DispatchPlan, ExpertPrecision
from heter_moe.policy.scheduler import PrecisionScheduler
from heter_moe.policy.strategies import (
    BaseStrategy,
    ConfidenceThresholdStrategy,
    ExpertLoadStrategy,
    RandomStrategy,
)

# Signals
from heter_moe.signals.attention import AttentionSignals
from heter_moe.signals.router import RouterSignals

# Integrations
from heter_moe.integrations.trtllm import (
    get_active_routers,
    patch_attention,
    patch_moe_factory,
    patch_router,
    unpatch_attention,
    unpatch_moe_factory,
    unpatch_router,
)

# Policy — config sweep
from heter_moe.policy.config_sweep import HeteroConfigSweeper

__all__ = [
    # Ops
    "HeteroAttention",
    "HeteroCutlassFusedMoE",
    "HeteroRouter",
    "create_subset_weights",
    # Policy
    "DispatchPlan",
    "ExpertPrecision",
    "PrecisionScheduler",
    "BaseStrategy",
    "ConfidenceThresholdStrategy",
    "ExpertLoadStrategy",
    "RandomStrategy",
    # Signals
    "AttentionSignals",
    "RouterSignals",
    # Integrations
    "get_active_routers",
    "patch_attention",
    "patch_moe_factory",
    "patch_router",
    "unpatch_attention",
    "unpatch_moe_factory",
    "unpatch_router",
    # Config sweep
    "HeteroConfigSweeper",
]
