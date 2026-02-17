"""Router signal collection for precision decisions."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RouterSignals:
    """Captured routing decisions and expert load statistics.

    These signals let the scheduler observe which experts are selected,
    how confidently the router picks them, and cumulative load, enabling
    strategies like confidence-threshold or expert-load-based precision
    assignment.

    Attributes:
        router_logits: Raw router output scores.
            Shape: [num_tokens, num_experts].
        expert_indices: Selected expert IDs per token.
            Shape: [num_tokens, top_k].
        expert_weights: Router-assigned weights per selected expert.
            Shape: [num_tokens, top_k].
        expert_load_counts: Cumulative activation count per expert
            across recent batches.  Shape: [num_experts].
    """

    router_logits: Optional[torch.Tensor] = None
    expert_indices: Optional[torch.Tensor] = None
    expert_weights: Optional[torch.Tensor] = None
    expert_load_counts: Optional[torch.Tensor] = None
