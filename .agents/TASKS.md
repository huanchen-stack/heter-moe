# TASKS.md — Implementation Roadmap

## Status Overview

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Split-execute-combine framework | ✅ Complete |
| Phase 2 | Benchmarking validation | ✅ Complete |
| Phase 3 | Library restructure + TRT-LLM integration | ✅ Complete |
| Phase 4 | End-to-end testing, C++ kernels, dynamic routing | 📋 Planned |

---

## Phase 3: Library Restructure + TRT-LLM Integration (✅ Complete)

### 3.1 Repository Structure Migration ✅
- [x] Restructured from flat layout to `signals/`, `policy/`, `ops/`, `integrations/`
- [x] Deleted Phase 1 reference code (`precision_router.py`, old `hetero_moe.py`)
- [x] Updated all `__init__.py` exports

### 3.2 Signal Dataclasses (signals/) ✅
- [x] `AttentionSignals` — token_importance (L2 norm), layer_index
- [x] `RouterSignals` — router_logits, expert_indices, expert_weights, expert_load_counts

### 3.3 Policy Module (policy/) ✅
- [x] `DispatchPlan` + `ExpertPrecision` enum — maps expert IDs to precision groups
- [x] `PrecisionScheduler` — signals + strategy → `DispatchPlan`
- [x] `RandomStrategy` — functional, static random assignment with hot_expert_ids support
- [x] `ConfidenceThresholdStrategy` — stub (raises `NotImplementedError`)
- [x] `ExpertLoadStrategy` — stub (raises `NotImplementedError`)
- [x] `HeteroConfigSweeper` — parameter search utility

### 3.4 Ops Module (ops/) ✅
- [x] `HeteroCutlassFusedMoE` — inherits `CutlassFusedMoE`, overrides `run_moe()` with 2x `fused_moe` calls
- [x] `HeteroRouter` — wraps `BaseMoeRoutingMethod` via composition, captures `RouterSignals`
- [x] `HeteroAttention` — forward monkey-patch on TRT-LLM `Attention`, captures `AttentionSignals`

### 3.5 TRT-LLM Integration Patches (integrations/) ✅
- [x] `patch_moe_factory()` — patches `create_moe_backend` to swap `CutlassFusedMoE` → `HeteroCutlassFusedMoE`
- [x] `patch_router()` — patches `create_moe` to wrap routing_method with `HeteroRouter`
- [x] `patch_attention(model)` — walks model tree, wraps each `Attention.forward()` with `HeteroAttention`
- [x] Corresponding `unpatch_*()` functions and `get_active_routers()` registry

### 3.6 torch.compile / CUDA Graph Compatibility Analysis ✅
- [x] `HeteroRouter.apply()` — safe (runs inside `moe_custom_op`, eager mode)
- [x] `HeteroAttention` — changed from `register_forward_hook` to forward override (hooks break torch.compile)
- [x] `torch.compiler.is_compiling()` + `torch.cuda.is_current_stream_capturing()` guards
- [x] Confirmed attention output is always bf16 (after `o_proj`) — no `Fp4QuantizedTensor` handling needed
- [x] CUDA graph replay: signals go stale (acceptable for current random policy)

---

## Phase 4: End-to-End Testing + Production (Planned)

### 4.1 Immediate Next Steps
- [ ] **Implement `ConfidenceThresholdStrategy`** — use `router_signals.expert_weights` to assign high-confidence → BF16
- [ ] **Implement `ExpertLoadStrategy`** — use `router_signals.expert_load_counts` to assign hot → BF16
- [ ] **End-to-end integration test** — load actual Qwen3-MoE model, run with patches, validate output
- [ ] **Accuracy benchmark** — perplexity comparison: pure BF16 vs heterogeneous at various NVFP4 ratios
- [ ] **Fix `patch_moe_factory` code path** — when `ENABLE_CONFIGURABLE_MOE=1` (default), `create_moe()` returns `ConfigurableMoE` directly, bypassing `create_moe_backend`. Current patch may not fire.

### 4.2 Unit Tests
- [ ] `test_signals.py` — `AttentionSignals`, `RouterSignals` dataclass construction
- [ ] `test_scheduler.py` — `PrecisionScheduler.schedule()` with various strategies
- [ ] `test_strategies.py` — `RandomStrategy` determinism, hot_expert_ids, edge cases
- [ ] `test_hetero_router.py` — `HeteroRouter.apply()` delegation + signal capture
- [ ] `test_hetero_attention.py` — `HeteroAttention` forward override + `remove()` restore
- [ ] `test_hetero_moe.py` — `HeteroCutlassFusedMoE.run_moe()` 2x GEMM + combine

### 4.3 Integration Tests
- [ ] `test_trtllm_patches.py` — `patch_router`, `patch_moe_factory`, `patch_attention` + unpatch
- [ ] `test_e2e_qwen3moe.py` — full model inference with heterogeneous precision

### 4.4 C++ Implementation (Future)
- [ ] Port token splitting / expert remapping to C++
- [ ] Implement `hetero_moe_forward` as `torch.library.custom_op`
- [ ] CUDA graph compatible implementation

### 4.5 Dynamic Expert Assignment (Future)
- [ ] Track per-expert activation counts across batches
- [ ] Implement threshold-based hot/cold detection
- [ ] Online weight conversion (BF16 → NVFP4)

### 4.6 Kernel-Level Heterogeneous Execution (Future)
- [ ] Modify GroupGEMM for mixed-precision experts in single kernel call
- [ ] Memory layout optimization

### 4.7 Distributed Support (Future)
- [ ] Expert parallelism with heterogeneous precision
- [ ] Tensor parallelism compatibility

---

## Completed Tasks

### Phase 1: Framework (✅ Complete)
- [x] `ExpertPrecision` enum, `PrecisionAssignment` dataclass
- [x] `ExpertPrecisionRouter` with `split_by_precision()` + `combine_outputs()`
- [x] `hetero_moe_forward()` orchestration
- [x] Reference `run_group_moe()` (matmul-based)
- *Note: Phase 1 reference code was deleted during Phase 3 restructure*

### Phase 2: Validation (✅ Complete)
- [x] Pure BF16, pure NVFP4, and heterogeneous benchmarks
- [x] Results: 1.20-1.36x speedup confirmed (RTX 5070 / SM120)
- [x] CUDA graph support validated

### Phase 3: Library Restructure (✅ Complete)
- [x] Full library restructure to `signals/` → `policy/` → `ops/` → `integrations/`
- [x] Signal capture via `HeteroRouter` and `HeteroAttention` wrappers
- [x] TRT-LLM integration via factory-level monkey-patches
- [x] torch.compile and CUDA graph compatibility analysis and implementation

---

## Known Limitations

1. **`patch_moe_factory` code path**: When `ENABLE_CONFIGURABLE_MOE=1` (TRT-LLM default), `create_moe()` returns `ConfigurableMoE` directly, bypassing `create_moe_backend`. The factory patch may not fire.
2. **CUDA graph signal staleness**: During graph replay, Python wrappers don't re-execute — signals from capture iteration persist. Acceptable for static/random policy, problematic for dynamic strategies.
3. **`ConfidenceThresholdStrategy` and `ExpertLoadStrategy`**: Currently stubs raising `NotImplementedError`.

## Open Questions

1. **Hot expert detection heuristic**: What threshold defines "hot"? Top 10% by activation count?
2. **Assignment update frequency**: How often to re-evaluate hot/cold? Per-batch? Per-epoch?
3. **Accuracy-speed tradeoff**: What's acceptable perplexity increase for 1.2x speedup?
4. **Memory budget**: Can we afford dual weights, or need on-the-fly quantization?
5. **ConfigurableMoE path**: Should we patch `ConfigurableMoE.forward_impl` directly instead of `create_moe_backend`?
