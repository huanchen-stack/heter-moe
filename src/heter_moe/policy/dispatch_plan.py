"""Dispatch plan and precision types for heterogeneous MoE."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ExpertPrecision(Enum):
    """Precision modes for MoE experts.

    NVFP4: 4-bit NV floating point (Blackwell SM100/103/120).
           Lower precision, higher throughput — suited for cold experts.
    BF16:  BFloat16, full precision — suited for hot experts where
           accuracy matters most.
    """

    NVFP4 = "nvfp4"
    BF16 = "bf16"


@dataclass
class DispatchPlan:
    """Output of the precision scheduler — which experts use which precision.

    Produced by ``PrecisionScheduler.schedule()`` and consumed by
    ``HeteroCutlassFusedMoE`` to split execution into two grouped GEMMs.

    Attributes:
        nvfp4_expert_ids: Sorted expert IDs assigned to NVFP4 (cold experts).
        bf16_expert_ids: Sorted expert IDs assigned to BF16 (hot experts).
        expert_to_precision: Mapping from every expert ID to its precision.
            Auto-populated from the two ID lists if not provided.
    """

    nvfp4_expert_ids: List[int]
    bf16_expert_ids: List[int]
    expert_to_precision: Dict[int, ExpertPrecision] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.expert_to_precision:
            for eid in self.nvfp4_expert_ids:
                self.expert_to_precision[eid] = ExpertPrecision.NVFP4
            for eid in self.bf16_expert_ids:
                self.expert_to_precision[eid] = ExpertPrecision.BF16

    @property
    def num_nvfp4(self) -> int:
        """Number of experts in the NVFP4 group."""
        return len(self.nvfp4_expert_ids)

    @property
    def num_bf16(self) -> int:
        """Number of experts in the BF16 group."""
        return len(self.bf16_expert_ids)

    def get_precision(self, expert_id: int) -> ExpertPrecision:
        """Get the assigned precision for *expert_id*.

        Defaults to BF16 if the expert is not found in either group.
        """
        return self.expert_to_precision.get(expert_id, ExpertPrecision.BF16)
