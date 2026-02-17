# AGENT.md

Heterogeneous Precision MoE: A standalone library for mixed-precision Mixture-of-Experts inference.
Enables NVFP4 (4-bit) for cold experts and BF16 (16-bit) for hot experts within the same model.

## Project Context

This project originated from exploration in TensorRT-LLM and is now a standalone library.
Primary target: Integration with TensorRT-LLM's PyTorch backend, but designed to be framework-agnostic.

## Rules

**CRITICAL:**
- Preserve `benchmark/` directory - contains previous exploration tests
- Follow TensorRT-LLM conventions when implementing adapters
- All CUDA operations must support CUDA graphs
- NVFP4 quantization must match TensorRT-LLM's `torch.ops.trtllm.fp4_quantize` format

**Code Style:**
- Type hints required on all public functions
- Docstrings: Google style
- Imports: Relative within package, absolute for externals

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Attention  │────▶│   Router    │────▶│    Scheduler    │────▶│  HeteroMoE  │
│  (signals/) │     │ (signals/)  │     │   (policy/)     │     │   (ops/)    │
└─────────────┘     └─────────────┘     └─────────────────┘     └─────────────┘
       │                   │                     ▲
       │                   │                     │
       └───────────────────┴─────────────────────┘
              Online Signal Collection → Precision Decision → Execution
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `signals/` | Dataclasses for attention/routing signals (`AttentionSignals`, `RouterSignals`) |
| `policy/` | Scheduler creates `DispatchPlan` from signals; strategies implement assignment algorithms |
| `ops/` | `HeteroCutlassFusedMoE` (2x GEMM), `HeteroRouter` (signal capture), `HeteroAttention` (signal capture) |
| `integrations/` | TRT-LLM monkey-patches: `patch_moe_factory`, `patch_router`, `patch_attention` |

### Signal Flow (TRT-LLM Integration)

```
Before LLM():                            After LLM():
  patch_router()      ─┐                   patch_attention(model)
  patch_moe_factory()  ├─ intercept          │
                       │  model build        │  walk model tree,
                       │                     │  wrap Attention.forward()
During inference:
  HeteroRouter.apply()                 ─── captures ──▶ RouterSignals
  HeteroAttention._wrapped_forward()   ─── captures ──▶ AttentionSignals
  PrecisionScheduler.schedule()        ───────────────▶ DispatchPlan
  HeteroCutlassFusedMoE.run_moe()     ───────────────▶ 2x fused_moe output
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Split-Execute-Combine** | Two grouped GEMMs (NVFP4 + BF16) using existing TRT-LLM kernels |
| **Expert ID Remapping** | Each precision group gets contiguous indices [0..N] for kernel compatibility |
| **Forward Override (not hooks)** | `register_forward_hook` breaks `torch.compile`; forward monkey-patch is Dynamo-safe |
| **`torch.compiler.is_compiling()` guard** | Dynamo eliminates signal-capture branch during graph tracing |
| **`torch.cuda.is_current_stream_capturing()` guard** | Prevents signal ops from being baked into CUDA graphs |
| **Composition-based Router wrapping** | `HeteroRouter` delegates to inner routing method — no inheritance chain issues |
| **Factory-level patches** | Intercept `create_moe` / `create_moe_backend` to inject at construction time |

## Directory Structure

```
heter-moe/
├── src/heter_moe/              # Main package
│   ├── __init__.py               # All public exports
│   ├── signals/                  # Signal dataclasses
│   │   ├── attention.py            # AttentionSignals (token_importance, layer_index)
│   │   └── router.py              # RouterSignals (router_logits, expert_indices, weights, load_counts)
│   ├── policy/                   # Precision decision logic
│   │   ├── dispatch_plan.py        # DispatchPlan + ExpertPrecision enum
│   │   ├── scheduler.py            # PrecisionScheduler (signals + strategy → plan)
│   │   ├── strategies.py           # BaseStrategy ABC, RandomStrategy, stubs (Confidence, ExpertLoad)
│   │   └── config_sweep.py         # HeteroConfigSweeper for parameter search
│   ├── ops/                      # Signal capture wrappers + execution
│   │   ├── attention.py            # HeteroAttention — forward monkey-patch, compile-safe
│   │   ├── router.py              # HeteroRouter — wraps BaseMoeRoutingMethod, captures RouterSignals
│   │   └── hetero_moe.py          # HeteroCutlassFusedMoE — 2x fused_moe GEMM
│   └── integrations/             # Framework adapters
│       └── trtllm.py              # patch/unpatch: moe_factory, router, attention
├── benchmarks/                   # Performance tests & exploration history
│   └── heter-moe/
├── tests/
│   ├── unit/
│   └── integration/
│       └── test_e2e_qwen3moe.py
├── cpp/                          # Future C++ kernels (empty)
└── pyproject.toml
```

## Common Commands

| Task | Command |
|------|---------|
| Run unit tests | `pytest tests/` |
| Benchmark accuracy | `python benchmarks/accuracy/benchmark_accuracy.py` |
| Benchmark throughput | `python benchmarks/throughput/benchmark_throughput.py` |

## Key Files

| File | Purpose |
|------|---------|
| `signals/attention.py` | `AttentionSignals` dataclass (token_importance, layer_index) |
| `signals/router.py` | `RouterSignals` dataclass (router_logits, expert_indices, expert_weights, expert_load_counts) |
| `policy/dispatch_plan.py` | `DispatchPlan` + `ExpertPrecision` enum — scheduler output |
| `policy/scheduler.py` | `PrecisionScheduler`: signals + strategy → `DispatchPlan` |
| `policy/strategies.py` | `BaseStrategy` ABC, `RandomStrategy` (functional), stubs for Confidence/Load |
| `policy/config_sweep.py` | `HeteroConfigSweeper` for parameter search |
| `ops/hetero_moe.py` | `HeteroCutlassFusedMoE` — takes `DispatchPlan`, runs 2x `fused_moe` |
| `ops/router.py` | `HeteroRouter` — wraps `BaseMoeRoutingMethod`, captures `RouterSignals` |
| `ops/attention.py` | `HeteroAttention` — forward monkey-patch on `Attention`, captures `AttentionSignals` |
| `integrations/trtllm.py` | `patch_moe_factory()`, `patch_router()`, `patch_attention()` + unpatchers |

## Dependencies

- PyTorch >= 2.0
- TensorRT-LLM (for production kernels, NVFP4 quantization ops, and base classes)
- CUDA >= 12.0 (SM120+ for NVFP4 support)

## Development Phases

1. **Phase 1** (Complete): Framework with split-execute-combine, control plane
2. **Phase 2** (Complete): Benchmarking proving NVFP4 speedup (1.20-1.36x)
3. **Phase 3** (Complete): Library restructure, signal capture hooks, TRT-LLM integration patches, torch.compile/CUDA graph compatibility
4. **Phase 4** (Future): C++ implementation, dynamic routing, end-to-end testing

## torch.compile / CUDA Graph Compatibility

### Key findings:

| Component | Status | Reason |
|-----------|--------|--------|
| `HeteroRouter.apply()` | ✅ Safe | Runs inside `moe_custom_op()` (`@torch.library.custom_op`) — eager mode |
| `HeteroAttention._wrapped_forward()` | ✅ Safe | Forward override with `torch.compiler.is_compiling()` guard — Dynamo eliminates capture branch |
| `HeteroCutlassFusedMoE.run_moe()` | ✅ Safe | Runs inside `moe_custom_op()` — eager mode |
| CUDA graph replay | ⚠️ Signals stale | Wrappers don't re-execute during replay; acceptable for current random policy |

### TRT-LLM compile flow:
- `model_engine.py:291`: `torch.compile(self.model.model, backend=..., fullgraph=...)`
- `interface.py:731`: `MoE.forward()` → when `is_torch_compiling()`, uses `moe_custom_op()` (custom op, eager inside)
- `attention.py:525`: `Attention.forward_impl()` → when `is_torch_compiling()`, uses `attn_custom_op_inplace()`
