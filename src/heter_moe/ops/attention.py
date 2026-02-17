"""Heterogeneous Attention: captures signals from TRT-LLM Attention.

Wraps a TRT-LLM ``Attention`` module's ``forward()`` to capture
output-derived signals as ``AttentionSignals``.  Since TRT-LLM uses
fused CUDA kernels for attention computation, raw attention weights
are not directly accessible.  Instead, we derive ``token_importance``
from the L2-norm of the attention output per token — a practical proxy
for how much signal each token carries through the layer.

Uses forward override (monkey-patch) instead of ``register_forward_hook``
to avoid graph breaks under ``torch.compile``.  TRT-LLM's own
compile-safe pattern uses forward overrides and custom ops — hooks are
only used in ``debug_mode()`` which is never active during compiled
inference.

Signal capture is skipped during ``torch.compile`` tracing and CUDA
graph capture.  ``torch.compiler.is_compiling()`` is a Dynamo-recognized
guard that enables dead-code elimination of the capture branch.

Usage::

    from heter_moe.ops.attention import HeteroAttention

    wrapper = HeteroAttention(model.layers[0].self_attn, layer_index=0)

    # After model.forward():
    signals = wrapper.last_signals

    wrapper.remove()  # Restore original forward
"""

from typing import Optional

import torch
from torch import nn

from heter_moe.signals.attention import AttentionSignals


class HeteroAttention:
    """Wraps a TRT-LLM ``Attention`` module to capture ``AttentionSignals``.

    Monkey-patches ``forward()`` with a thin wrapper that calls the
    original forward and then derives token importance from the output.
    This avoids ``register_forward_hook`` which can break
    ``torch.compile`` graph tracing.

    ``Attention.forward()`` always returns bf16 ``torch.Tensor`` —
    the ``o_proj`` output projection converts any internal quantized
    representation (e.g. ``Fp4QuantizedTensor``) before returning.

    Args:
        attention_module: The TRT-LLM ``Attention`` instance to wrap.
        layer_index: Transformer layer index for signal tagging.
    """

    def __init__(
        self,
        attention_module: nn.Module,
        layer_index: int,
    ):
        self._module = attention_module
        self._layer_index = layer_index
        self._last_signals: Optional[AttentionSignals] = None

        # Save original forward for restore and delegation
        self._original_forward = attention_module.forward

        # Build and install the wrapper closure
        hetero_attn = self
        original_forward = self._original_forward

        def _wrapped_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
            output = original_forward(*args, **kwargs)

            # Skip under torch.compile and CUDA graph capture.
            # torch.compiler.is_compiling() is a Dynamo-recognized guard
            # that enables dead-code elimination of this branch during
            # graph tracing.  During CUDA graph replay the wrapper does
            # not execute at all (only recorded kernels replay).
            if (
                not torch.compiler.is_compiling()
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Attention.forward() returns bf16 after o_proj.
                # Token importance = L2 norm per token: [num_tokens]
                token_importance = torch.norm(
                    output.float(), dim=-1,
                ).detach()

                hetero_attn._last_signals = AttentionSignals(
                    token_importance=token_importance,
                    layer_index=hetero_attn._layer_index,
                )

            return output

        attention_module.forward = _wrapped_forward

    # ------------------------------------------------------------------
    # Signal access
    # ------------------------------------------------------------------

    @property
    def last_signals(self) -> Optional[AttentionSignals]:
        """Most recent ``AttentionSignals``, or ``None``."""
        return self._last_signals

    @property
    def layer_index(self) -> int:
        """Transformer layer index this hook is attached to."""
        return self._layer_index

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Restore the original ``forward()`` on the wrapped module."""
        self._module.forward = self._original_forward
