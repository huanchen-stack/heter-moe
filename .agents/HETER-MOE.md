# HETER-MOE.md — Technical Reference

## Problem Statement

Standard MoE models use uniform precision for all experts. However:
- **Cold experts** (rarely activated): Memory-bound, benefit from 4-bit quantization
- **Hot experts** (frequently activated): Compute-bound, need FP16/BF16 for accuracy

**Heterogeneous MoE** assigns different precisions to different experts based on usage patterns.

## Performance Results (RTX 5070 / SM120)

### NVFP4 vs BF16 Speedup

| Batch Size | BF16 (ms) | NVFP4 (ms) | Speedup |
|------------|-----------|------------|---------|
| 1          | 267.67    | 258.27     | **1.04x** |
| 4          | 307.01    | 277.74     | **1.11x** |
| 8          | 352.86    | 293.89     | **1.20x** |
| 32         | 135.58    | 101.16     | **1.34x** |
| 64         | 165.45    | 121.97     | **1.36x** |

### Heterogeneous Estimates (Batch Size 8)

| NVFP4 % | Est. Time (ms) | Speedup |
|---------|----------------|---------|
| 0%      | 352.86         | 1.00x   |
| 25%     | 338.11         | 1.04x   |
| 50%     | 323.37         | 1.09x   |
| 75%     | 308.63         | 1.14x   |
| 100%    | 293.89         | 1.20x   |

**Key insight**: At 50% NVFP4 (keeping hot experts in BF16 for accuracy), achieve ~1.09x speedup.

## Core Components

### HeteroCutlassFusedMoE (`ops/hetero_moe.py`)

Inherits from TRT-LLM's `CutlassFusedMoE`. The ONLY override is `run_moe()`:

```python
class HeteroCutlassFusedMoE(CutlassFusedMoE):
    """Two grouped GEMMs instead of one."""

    def __init__(self, *, dispatch_plan: DispatchPlan, **kwargs):
        super().__init__(**kwargs)
        self.dispatch_plan = dispatch_plan
        # Lazy-init caches for remap tables + subset weights

    def run_moe(self, x, token_selected_experts, token_final_scales, ...):
        # 1. Build remap tables: global expert ID → local index (cached)
        # 2. For each group (NVFP4, BF16):
        #    - Remap expert IDs (sentinel for non-group)
        #    - Zero scales for non-group experts
        #    - Call torch.ops.trtllm.fused_moe() with subset weights
        # 3. Sum outputs: out_nvfp4 + out_bf16
```

Key details:
- Keeps `[N, top_k]` routing shape — does NOT flatten/split tokens
- Remap tables and subset weights are cached after first call
- Out-of-group expert IDs map to sentinel value (num_group_experts)
- Scales zeroed for non-group experts → kernel ignores them

### HeteroRouter (`ops/router.py`)

Wraps any TRT-LLM `BaseMoeRoutingMethod` via composition:

```python
class HeteroRouter(BaseMoeRoutingMethod):
    """Wraps routing method to capture RouterSignals."""

    def __init__(self, inner: BaseMoeRoutingMethod):
        self._inner = inner

    def apply(self, router_logits):
        # Delegate to inner method
        experts, scales = self._inner.apply(router_logits)
        # Capture signals (runs in eager mode inside moe_custom_op)
        self._last_signals = RouterSignals(
            router_logits=router_logits.detach(),
            expert_indices=experts.detach(),
            expert_weights=scales.detach(),
            expert_load_counts=scatter_add_counts(experts),
        )
        return experts, scales

    def __getattr__(self, name):
        # Forward unknown attrs to inner (top_k, output_dtype, etc.)
        return getattr(self._inner, name)
```

Safe under torch.compile: `apply()` runs inside `moe_custom_op()` which executes in eager mode.

### HeteroAttention (`ops/attention.py`)

Forward monkey-patch on TRT-LLM `Attention` modules:

```python
class HeteroAttention:
    """Captures AttentionSignals via forward override."""

    def __init__(self, attention_module, layer_index):
        original_forward = attention_module.forward

        def _wrapped_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)

            # Dynamo eliminates this branch during graph tracing
            if (not torch.compiler.is_compiling()
                    and not torch.cuda.is_current_stream_capturing()):
                token_importance = torch.norm(output.float(), dim=-1).detach()
                self._last_signals = AttentionSignals(
                    token_importance=token_importance,
                    layer_index=layer_index,
                )
            return output

        attention_module.forward = _wrapped_forward

    def remove(self):
        """Restore original forward."""
```

Key design choices:
- Forward override instead of `register_forward_hook` — hooks break torch.compile
- `torch.compiler.is_compiling()` is a Dynamo-recognized guard → dead-code elimination
- `torch.cuda.is_current_stream_capturing()` prevents ops from being baked into CUDA graphs
- `Attention.forward()` always returns bf16 after `o_proj` — no special handling needed

### PrecisionScheduler (`policy/scheduler.py`)

Central coordinator: signals + strategy → `DispatchPlan`:

```python
scheduler = PrecisionScheduler(
    num_experts=128,
    strategy=RandomStrategy(nvfp4_ratio=0.5, seed=42),
)
plan = scheduler.schedule(
    attention_signals=attn_signals,  # Optional
    router_signals=router_signals,   # Optional
)
# plan.nvfp4_expert_ids = [3, 7, 12, ...]
# plan.bf16_expert_ids = [0, 1, 2, 4, 5, 6, ...]
```

### Strategy Hierarchy (`policy/strategies.py`)

```python
class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    def assign(self, num_experts, attention_signals=None, router_signals=None) -> DispatchPlan: ...

class RandomStrategy(BaseStrategy):        # ✅ Functional
class ConfidenceThresholdStrategy(BaseStrategy):  # 🔲 Stub
class ExpertLoadStrategy(BaseStrategy):           # 🔲 Stub
```

### DispatchPlan (`policy/dispatch_plan.py`)

```python
class ExpertPrecision(Enum):
    NVFP4 = "nvfp4"
    BF16 = "bf16"

@dataclass
class DispatchPlan:
    nvfp4_expert_ids: List[int]    # Sorted
    bf16_expert_ids: List[int]     # Sorted
    expert_to_precision: Dict[int, ExpertPrecision]  # Auto-populated
```

## TRT-LLM Integration (`integrations/trtllm.py`)

Three independent patch functions:

```python
# Before LLM() construction:
patch_router()                    # Wraps create_moe to inject HeteroRouter
patch_moe_factory(nvfp4_ratio=0.5, seed=42)  # Swaps CutlassFusedMoE → HeteroCutlassFusedMoE

llm = LLM(model=model_dir, ...)

# After LLM() construction:
attn_hooks = patch_attention(llm.model)  # Wraps Attention.forward per layer

# During inference — signals captured automatically:
output = llm.generate(prompts)

# Query signals:
for router in get_active_routers():
    print(router.last_signals.expert_load_counts)
for hook in attn_hooks:
    print(hook.last_signals.token_importance)

# Cleanup:
unpatch_router()
unpatch_moe_factory()
unpatch_attention(attn_hooks)
```

### Patch Targets

| Patch | Target | When |
|-------|--------|------|
| `patch_router()` | `create_moe.create_moe()` | Before `LLM()` |
| `patch_moe_factory()` | `create_moe.create_moe_backend()` | Before `LLM()` |
| `patch_attention(model)` | Each `Attention.forward` in model tree | After `LLM()` |

### Known Limitation: ConfigurableMoE Path

When `ENABLE_CONFIGURABLE_MOE=1` (TRT-LLM default), `create_moe()` returns `ConfigurableMoE` directly, bypassing `create_moe_backend()`. The `patch_moe_factory()` patch targets `create_moe_backend` and may not fire in this code path.

## NVFP4 Quantization

TensorRT-LLM provides NVFP4 quantization via:

```python
w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(
    weight_bf16,           # [out_features, in_features] BF16
    global_scale,          # float: 448*6 / max_abs_val
    scaling_vector_size,   # int: block size (default 16)
    use_ue8m0=False,
    swizzle=False,
)
# w_fp4: [out_features, in_features // 2] uint8 (packed 4-bit)
# w_sf: [out_features, in_features // scaling_vector_size] FP8 scales
```

Reference: `tensorrt_llm/tests/unittest/_torch/modules/moe/quantize_utils.py`

## torch.compile / CUDA Graph Compatibility

### Component Safety

| Component | torch.compile | CUDA Graph | Notes |
|-----------|:---:|:---:|-------|
| `HeteroRouter.apply()` | ✅ | ✅ | Inside `moe_custom_op` → eager mode |
| `HeteroAttention` | ✅ | ⚠️ | Forward override + `is_compiling()` guard. Signals stale during graph replay. |
| `HeteroCutlassFusedMoE.run_moe()` | ✅ | ✅ | Inside `moe_custom_op` → eager mode |

### TRT-LLM Compile Flow

```
torch.compile(model, backend=..., fullgraph=...)
  └─ Dynamo traces model graph
      ├─ MoE.forward() → is_torch_compiling() → moe_custom_op()
      │   └─ @torch.library.custom_op → body runs in EAGER mode
      │       ├─ routing_method.apply()  ← HeteroRouter runs here (safe)
      │       └─ run_moe()              ← HeteroCutlassFusedMoE runs here (safe)
      └─ Attention.forward_impl() → is_torch_compiling() → attn_custom_op_inplace()
          └─ Our _wrapped_forward: torch.compiler.is_compiling() → True → skip signals
```

### CUDA Graph Behavior

- `CUDAGraphRunner`: 2 warmup iterations → `torch.cuda.graph()` capture → replay
- During replay: only recorded CUDA kernels execute; Python wrappers don't re-run
- Signals from capture iteration persist → stale for dynamic policies
- Acceptable for `RandomStrategy` (static assignment doesn't need fresh signals)

## KV Cache Considerations

RTX 5070 has limited VRAM. Configure KV cache:

```python
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

kv_cache_config = KvCacheConfig(
    free_gpu_memory_fraction=0.15,  # Leave room for dual weights
)
```

## Known Limitations

1. **ConfigurableMoE bypass** — `patch_moe_factory` may not fire when `ENABLE_CONFIGURABLE_MOE=1` (default)
2. **CUDA graph signal staleness** — Signals don't refresh during graph replay
3. **Strategy stubs** — `ConfidenceThresholdStrategy` and `ExpertLoadStrategy` raise `NotImplementedError`
4. **Static expert assignment** — No dynamic hot/cold detection yet
5. **Single-GPU only** — No tensor/expert parallelism support yet
6. **No unit tests yet** — Only integration test scaffold exists

## References

- TRT-LLM MoE: `tensorrt_llm/_torch/modules/fused_moe/`
- TRT-LLM Routing: `tensorrt_llm/_torch/modules/fused_moe/routing.py`
- TRT-LLM Attention: `tensorrt_llm/_torch/modules/attention.py`
- TRT-LLM Compile: `tensorrt_llm/_torch/auto_deploy/compile/compiler.py`
- NVFP4 quantization: `tests/unittest/_torch/modules/moe/quantize_utils.py`
- CUTLASS GroupGEMM: `cpp/tensorrt_llm/cutlass_extensions/`
