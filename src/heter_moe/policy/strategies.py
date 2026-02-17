"""Precision assignment strategies for heterogeneous MoE.

All strategies inherit from ``BaseStrategy`` and implement ``assign()``,
which takes the number of experts plus optional signal inputs and returns
a ``DispatchPlan``.

Currently functional:
    - ``RandomStrategy`` — static random assignment (used in benchmarks).

Planned (stubs):
    - ``ConfidenceThresholdStrategy`` — use router confidence scores.
    - ``ExpertLoadStrategy`` — use cumulative expert activation counts.
"""

import abc
import random
from typing import List, Optional

from heter_moe.policy.dispatch_plan import DispatchPlan
from heter_moe.signals.attention import AttentionSignals
from heter_moe.signals.router import RouterSignals


class BaseStrategy(abc.ABC):
    """Abstract base for precision assignment strategies.

    Subclasses must implement ``assign()`` which inspects optional
    attention/router signals and produces a ``DispatchPlan``.
    """

    @abc.abstractmethod
    def assign(
        self,
        num_experts: int,
        attention_signals: Optional[AttentionSignals] = None,
        router_signals: Optional[RouterSignals] = None,
    ) -> DispatchPlan:
        """Determine precision assignment for each expert.

        Args:
            num_experts: Total number of experts in the MoE layer.
            attention_signals: Optional attention-based signals.
            router_signals: Optional routing-based signals.

        Returns:
            A ``DispatchPlan`` mapping experts to precision groups.
        """
        ...


class RandomStrategy(BaseStrategy):
    """Randomly assign a fraction of experts to NVFP4.

    Experts listed in *hot_expert_ids* are always kept at BF16.
    Among the remaining experts, *nvfp4_ratio* are randomly selected
    for NVFP4.  The assignment is deterministic when *seed* is set.

    Args:
        nvfp4_ratio: Fraction of non-hot experts to assign to NVFP4.
        seed: Random seed for reproducibility.
        hot_expert_ids: Expert IDs forced to BF16 regardless of ratio.
    """

    def __init__(
        self,
        nvfp4_ratio: float = 0.5,
        seed: Optional[int] = None,
        hot_expert_ids: Optional[List[int]] = None,
    ):
        self.nvfp4_ratio = nvfp4_ratio
        self.seed = seed
        self.hot_expert_ids = set(hot_expert_ids) if hot_expert_ids else set()

    def assign(
        self,
        num_experts: int,
        attention_signals: Optional[AttentionSignals] = None,
        router_signals: Optional[RouterSignals] = None,
    ) -> DispatchPlan:
        if self.seed is not None:
            random.seed(self.seed)

        all_experts = list(range(num_experts))
        remaining = [e for e in all_experts if e not in self.hot_expert_ids]

        num_nvfp4 = int(len(remaining) * self.nvfp4_ratio)
        random.shuffle(remaining)

        nvfp4_ids = sorted(remaining[:num_nvfp4])
        bf16_ids = sorted(remaining[num_nvfp4:] + list(self.hot_expert_ids))

        return DispatchPlan(nvfp4_expert_ids=nvfp4_ids, bf16_expert_ids=bf16_ids)


class ConfidenceThresholdStrategy(BaseStrategy):
    """Assign precision based on router confidence scores.

    High-confidence expert selections → BF16 (these experts contribute
    more to the output).  Low-confidence selections → NVFP4 (less
    impact on quality, benefit from speed).

    Args:
        confidence_threshold: Experts with mean routing weight above
            this threshold are assigned BF16.
        hot_expert_ids: Expert IDs forced to BF16.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        hot_expert_ids: Optional[List[int]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.hot_expert_ids = set(hot_expert_ids) if hot_expert_ids else set()

    def assign(
        self,
        num_experts: int,
        attention_signals: Optional[AttentionSignals] = None,
        router_signals: Optional[RouterSignals] = None,
    ) -> DispatchPlan:
        raise NotImplementedError(
            "ConfidenceThresholdStrategy requires router_signals with "
            "expert_weights.  Not yet implemented."
        )


class ExpertLoadStrategy(BaseStrategy):
    """Assign precision based on expert activation frequency.

    Hot experts (high cumulative activation count) → BF16 for accuracy.
    Cold experts (low activation count) → NVFP4 for speed.

    Args:
        nvfp4_ratio: Fraction of experts to place in the NVFP4 group
            (those with the *lowest* activation counts).
        hot_expert_ids: Expert IDs forced to BF16.
    """

    def __init__(
        self,
        nvfp4_ratio: float = 0.5,
        hot_expert_ids: Optional[List[int]] = None,
    ):
        self.nvfp4_ratio = nvfp4_ratio
        self.hot_expert_ids = set(hot_expert_ids) if hot_expert_ids else set()

    def assign(
        self,
        num_experts: int,
        attention_signals: Optional[AttentionSignals] = None,
        router_signals: Optional[RouterSignals] = None,
    ) -> DispatchPlan:
        raise NotImplementedError(
            "ExpertLoadStrategy requires router_signals with "
            "expert_load_counts.  Not yet implemented."
        )
