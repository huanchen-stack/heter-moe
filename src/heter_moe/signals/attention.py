"""Attention signal collection for precision decisions."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AttentionSignals:
    """Captured attention scores for token importance estimation.

    These signals allow the scheduler to gauge how "important" each
    token is based on how much attention it receives, which can
    influence whether its routed experts should run at higher precision.

    Attributes:
        attention_scores: Raw attention weights.
            Shape: [num_heads, seq_len, seq_len] or [num_layers, num_heads, seq_len, seq_len].
        token_importance: Derived per-token importance scores.
            Shape: [seq_len].  Typically aggregated from attention_scores
            (e.g., mean received-attention across heads).
        layer_index: Which transformer layer these signals originate from.
    """

    attention_scores: Optional[torch.Tensor] = None
    token_importance: Optional[torch.Tensor] = None
    layer_index: Optional[int] = None
