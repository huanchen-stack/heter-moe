"""Precision scheduler for heterogeneous MoE.

The scheduler is the central coordinator: it accepts signals (attention
and/or router), delegates to a pluggable ``BaseStrategy``, and produces
a ``DispatchPlan`` that the ops layer consumes.

Usage::

    from heter_moe.policy.scheduler import PrecisionScheduler
    from heter_moe.policy.strategies import RandomStrategy

    scheduler = PrecisionScheduler(
        num_experts=128,
        strategy=RandomStrategy(nvfp4_ratio=0.5, seed=42),
    )
    plan = scheduler.schedule()
"""

from typing import Optional

from heter_moe.policy.dispatch_plan import DispatchPlan
from heter_moe.policy.strategies import BaseStrategy
from heter_moe.signals.attention import AttentionSignals
from heter_moe.signals.router import RouterSignals


class PrecisionScheduler:
    """Produces a ``DispatchPlan`` by running a strategy on collected signals.

    Attributes:
        num_experts: Number of experts in the MoE layer.
        strategy: The assignment algorithm to use.
    """

    def __init__(
        self,
        num_experts: int,
        strategy: BaseStrategy,
    ):
        self.num_experts = num_experts
        self.strategy = strategy
        self._current_plan: Optional[DispatchPlan] = None

    def schedule(
        self,
        attention_signals: Optional[AttentionSignals] = None,
        router_signals: Optional[RouterSignals] = None,
    ) -> DispatchPlan:
        """Run the strategy and return a new ``DispatchPlan``.

        Args:
            attention_signals: Optional attention-derived signals.
            router_signals: Optional routing-derived signals.

        Returns:
            The computed ``DispatchPlan``, also stored as
            ``self.current_plan``.
        """
        self._current_plan = self.strategy.assign(
            num_experts=self.num_experts,
            attention_signals=attention_signals,
            router_signals=router_signals,
        )
        return self._current_plan

    @property
    def current_plan(self) -> Optional[DispatchPlan]:
        """The most recently computed ``DispatchPlan``, or ``None``."""
        return self._current_plan
