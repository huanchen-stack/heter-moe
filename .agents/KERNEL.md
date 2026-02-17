# KERNEL.md — Kernel Optimization Strategy for Heterogeneous MoE

**Scope**: NVFP4 × BF16 heterogeneous precision only.

## Current State

Two sequential `torch.ops.trtllm.fused_moe()` calls in `HeteroCutlassFusedMoE.run_moe()`:
1. **NVFP4 group** — cold experts, very few tokens per expert (1-8)
2. **BF16 group** — hot experts, more tokens per expert (16-64)
3. **Sum** the two outputs

Python-level overhead: remap tables, mask/zero scales, weight subsetting (cached after first call).

The `cpp/` directory is empty — Phase 4 target.

---

## How TRT-LLM Autotuner Works (Key Background)

### Tactic Selection
Each `fused_moe()` call goes through the autotuner which selects a `CutlassGemmConfig` (tile shape + cluster shape + stages + split-k + swap_ab):

```
MoERunner.tuning_config = TuningConfig(
    dynamic_tensor_specs = DynamicTensorSpec(
        input_idx=0, dim_idx=0,                           # tune on num_tokens dimension
        gen_tuning_buckets = [1, 2, 4, 8, 16, ..., 8192], # power-of-2 buckets
        map_to_tuning_buckets = last_positive_power_of_2   # runtime: 5 tokens → bucket 4
    ),
    tune_max_num_tokens = 8192,
)
```

- **ONE tactic per bucket** — autotuner profiles all available tile configs, picks the fastest
- **GEMM1 and GEMM2 tuned separately** — `trtllm::fused_moe::gemm1` and `::gemm2`
- **Tactic = index into mGemm{1,2}Profiles** — a vector of `CutlassGemmConfig` structs
- **swap_ab is a tactic option** — autotuner profiles both swap_ab=true and swap_ab=false; configs are doubled by appending swapped variants (see `getTmaWarpSpecializedConfigs()`)

### Available Tile Configs by Architecture (NVFP4-relevant)

| Architecture | Tile Configs (CTA M×N×K) | Notes |
|---|---|---|
| **SM103** (NVFP4×NVFP4 only) | **128×128×128, 128×256×128** | ⚠️ Only 2 tiles. No small-N/small-M FP4 configs |
| **SM100** (NVFP4 on Blackwell) | 128×64×128, 128×128×128, 128×256×128 | 3 tiles. SM103 dispatch does NOT fall back to these for FP4 |
| **SM90** (BF16×BF16 Hopper) | 128×16×128, 128×32×128, 128×64×128, 128×128×128, 128×256×128, 256×128×128 | Rich small-N selection |
| **SM100** (BF16×BF16 Blackwell) | 64×32×128, 64×64×128, 64×128×128, 128×16×128, 128×32×128, 128×64×128, 128×128×128, 128×256×128 | Rich selection |
| **SM120** (BF16×BF16) | 128×128×64, 128×128×128, 128×256×64, 256×128×64 | No small-N |

**Key gap**: SM103 NVFP4 has the least tile variety. BF16 groups have excellent tile coverage.

### swap_ab Mechanism

`swap_ab` transposes the grouped GEMM problem: A↔B swap, output layout transposed.

**What it does for small-M experts**:
- Without swap_ab: A=[M_i, K] activations, B=[K, N] weights. Tile 128×128: M-dimension 1.6% utilized when M_i=2.
- With swap_ab: A=[N, K] weights (large), B=[K, M_i] activations (small). Tile 128×128: N-dimension 1.6% utilized when M_i=2.
- **Same tile waste** in terms of compute — but **TMA access patterns differ**:
  - swap_ab=false: TMA loads small activation tensor (M_i×K_chunk) as A — poor prefetch utilization
  - swap_ab=true: TMA loads large weight tensor (N×K_chunk) as A — better prefetch utilization, better instruction scheduling
- The autotuner already profiles both and picks the faster one per bucket.

### min_latency_mode
A separate code path (`run_moe_min_latency`) optimized for low-batch inference. Returns additional metadata (active expert counts, expert-to-token scores). Currently used in `TRTLLMGenFusedMoE` backend.

### Key Files
| File | Role |
|---|---|
| `TensorRT-LLM/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` | MoERunner, fused_moe op, tactic profiling |
| `TensorRT-LLM/tensorrt_llm/_torch/autotuner.py` | AutoTuner framework, bucketing, cache |
| `TensorRT-LLM/cpp/tensorrt_llm/thop/moeOp.cpp` | C++ FusedMoeRunner, setRunnerProfiles, profile selection |
| `TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/` | CUTLASS MoE GEMM dispatch per architecture |
| `TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` | runMoe(), per-expert problem shape setup, swap_ab application |
| `TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h` | TMA warp-specialized dispatch, tile shape selection, SM103 constraints |
| `TensorRT-LLM/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h` | CutlassGemmConfig struct (tile, cluster, swap_ab, epilogue) |

---

## Deep Dive: Per-Expert-Group Tile Selection — Three Architectures

### The Goal

Within one heterogeneous MoE call, different experts have vastly different token counts (M).
Cold experts: M=1-8 tokens. Hot experts: M=16-64. A single tile config wastes compute on one end.

**We want different expert groups to use different optimal tile configs** — e.g., swap_ab=true with 128×128 for M≤4 experts, swap_ab=false with 128×256 for M≥16 experts.

Three architectures can achieve this, each at a different level of the stack:

### Current Architecture (Baseline)

```
setTactic(config) → runMoe() → gemm1(config) → dispatchMoeGemmSelectTileShapeTmaWarpSpecialized(config)
                                                    ↓
                                             switch (tile_config_sm100) {
                                                 case CtaShape128x128x128B:
                                                     using TileShape = Shape<_128, _128, K>;  // COMPILE-TIME
                                                     // ALL experts share this ONE tile
                                             }
```

Per-expert problem shapes are runtime (moe_kernels.cu:1290), but tile shape is compile-time.
swap_ab is global per launch. The autotuner profiles 4 configs (2 tiles × 2 swap_ab) on SM103 NVFP4.

---

### Architecture A: C++ Multi-Dispatch Orchestrator ⭐ RECOMMENDED FIRST

**Concept**: Partition experts by M-range at runtime. Call `runMoe()` multiple times, each with a different tactic optimized for that M-range. No kernel modifications needed.

```
                              ┌─ setTactic(config_small) → runMoe(small_experts)
Expert Partitioner ───────────┤
  (by per-expert M)           └─ setTactic(config_large) → runMoe(large_experts)
                              
                              → sum outputs
```

#### Feasibility: ✅ FULLY VALIDATED

**Evidence from codebase**:

1. **`setTactic()` is trivial** (moe_kernels.h:607-612):
   ```cpp
   void setTactic(std::optional<CutlassGemmConfig> gemm1_config,
                  std::optional<CutlassGemmConfig> gemm2_config) override {
       gemm1_config_ = std::move(gemm1_config);
       gemm2_config_ = std::move(gemm2_config);
   }
   ```
   Just stores two optionals. Zero-cost config switch.

2. **`gemm1()` and `gemm2()` are static** (moe_kernels.h:636-670):
   ```cpp
   static void gemm1(MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
                     ..., CutlassGemmConfig config, ...);
   ```
   Stateless — takes config as parameter. Can be called with any config without reinitializing.

3. **Workspace is config-independent** (moe_kernels.cu:2822-2837):
   ```cpp
   size_t getWorkspaceSize(int64_t num_rows, int64_t hidden_size, int64_t inter_size,
                           int num_experts, ...) {
       // NO tile config parameter — workspace depends only on problem dimensions
   }
   ```
   Same workspace buffer serves any tile config. No reallocation between calls.

4. **TMA workspace prep is lightweight** — `prepareTmaWsInputs()` (moe_kernels.cu:4605) caches by `(finalize_fusion, swap_ab)` key. O(1) per call — just pointer setup + scale layout.

5. **Already proven pattern** — benchmarking code calls `runMoe()` multiple times in profiling loops. The API is designed for multi-call usage.

#### Performance Model

```
T_baseline = T_launch + T_gemm(all_experts, config_compromise)

T_arch_A   = T_partition                           # ~2-5μs CPU
           + T_launch_1 + T_gemm(small_experts, config_optimal_small)
           + T_launch_2 + T_gemm(large_experts, config_optimal_large)
           + T_sum                                  # extra output accumulation

Breakeven when:
  T_gemm(all, compromise) - [T_gemm(small, opt) + T_gemm(large, opt)] > T_launch + T_partition + T_sum
```

**Expected win**: 10-30% on NVFP4 group when M variance is high (e.g., some experts M=1, others M=32).
**Expected loss**: When M is uniform or GEMM is sub-100μs (launch overhead ~5-10μs dominates).

#### Implementation Complexity: LOW (2-3 weeks)

- **No CUTLASS modifications** — uses existing `setTactic()` / `runMoe()` API
- **No kernel changes** — pure orchestration layer
- **C++ custom op in `cpp/`** — first contribution to empty directory
- **Python fallback** — can prototype in Python first, then port to C++

#### Implementation Sketch

```cpp
// cpp/hetero_moe/multi_dispatch.h
struct ExpertPartition {
    std::vector<int> expert_ids;
    cutlass_extensions::CutlassGemmConfig config;
};

// Partition experts by M-range, pick optimal config per partition
std::vector<ExpertPartition> partition_experts(
    int const* expert_first_token_offset,
    int num_experts,
    std::vector<CutlassGemmConfig> const& available_configs);

// Execute: for each partition, setTactic + runMoe + accumulate
void multi_dispatch_moe(
    CutlassMoeFCRunner& runner,
    std::vector<ExpertPartition> const& partitions,
    /* ... standard fused_moe args ... */,
    cudaStream_t stream);
```

#### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Launch overhead > tile savings | No benefit for small GEMMs | Profile threshold: skip partitioning when total GEMM < 100μs |
| Workspace sharing between calls | Memory pressure | Already validated: workspace is config-independent, reuse same buffer |
| CUDA graph capture with variable launches | Graph topology changes per batch | Use conditional capture or fall back to non-graph path |
| Output accumulation overhead | Extra kernel for sum | Fuse into epilogue or use in-place accumulation |

---

### Architecture B: Custom CUTLASS GroupScheduler with Per-Group Config Routing

**Concept**: Modify the CUTLASS `MoeProblemVisitor` (scheduler) to carry a per-expert tile config index. When the persistent kernel loop gets a work tile, it reads the config for that expert and dispatches to the correct compiled codepath.

```
Kernel launch (all tile codepaths compiled in)
  ↓
while (problem_visitor.next_tile()) {
    int problem_idx = problem_visitor.problem_index();
    int config_id = tile_config_per_expert[problem_idx];  // NEW: per-expert config lookup
    switch (config_id) {
        case 0: run_mma<TileShape_128x128>(...); break;
        case 1: run_mma<TileShape_128x256>(...); break;
    }
}
```

#### Feasibility: ⚠️ POSSIBLE BUT VERY COMPLEX

**What works**:

1. **`problem_idx` is available per tile** — the persistent kernel loop (moe_cutlass_kernel.h:511-651) already reads `problem_visitor.problem_index()` to get the current expert. Adding `tile_config_per_expert[problem_idx]` is trivial indexing.

2. **SM80 kernel has clear modification point** — `moe_cutlass_kernel.h:583`:
   ```cpp
   Mma mma = CreateMMA();  // Currently one Mma type per kernel
   ```
   Could branch to different `Mma` instantiations based on `config_id`.

3. **CTA-level branching** — different CTAs process different experts. No warp divergence because all warps within a CTA process the same expert's tile.

**What blocks**:

1. **SM90+ uses CUTLASS 3.x `GemmUniversal`** — the TMA warp-specialized path (moe_gemm_tma_ws_launcher.inl:110-300) uses CUTLASS 3.x collective builder, NOT the SM80 `moe_cutlass_kernel.h`. The scheduler is CUTLASS's built-in `GroupProblemShape` + persistent tile scheduler. Per-expert routing would require forking CUTLASS 3.x internals.

2. **Different tile configs = different template instantiations** — `Mma`, `Epilogue`, `SharedStorage` all change with tile shape. A single kernel binary must contain ALL codepaths, increasing:
   - **Register pressure**: Compiler sees all paths, allocates registers for worst case → reduced occupancy
   - **Binary size**: Each tile codepath is ~10-30KB of PTX
   - **Shared memory**: Must statically allocate for the largest tile (128×256×128) even when running the smaller one

3. **TMA descriptors are tile-shape-dependent** — TMA hardware loads data in shape-specific patterns. Different tiles need different TMA descriptor setup, which is done at launch time, not per-CTA. Would need per-expert TMA descriptors (memory and setup overhead).

4. **CUTLASS 3.x collective architecture** — the mainloop and epilogue are tightly coupled through `CollectiveBuilder`. Swapping the tile mid-kernel violates the collective's invariants.

#### Performance Model

```
T_arch_B = T_launch + T_gemm_branching(all_experts)

Where T_gemm_branching < T_gemm_compromise IF:
  - Register pressure increase doesn't kill occupancy
  - Branch overhead is negligible (CTA-level, not warp-level)
  - TMA descriptor overhead is amortized
```

**Best case**: Single launch eliminates Architecture A's multi-launch overhead. Each expert gets optimal tile.
**Worst case**: Register pressure from multiple codepaths reduces occupancy so much that ALL experts run slower.

#### Implementation Complexity: VERY HIGH (2-3 months)

- Requires forking CUTLASS 3.x grouped GEMM kernel
- Must handle TMA descriptor generation per-tile-config
- SM80 path (moe_cutlass_kernel.h) is more feasible to modify than SM90+/SM100+ path
- Testing matrix explodes: `N_tile_configs × N_cluster_shapes × N_expert_counts`

#### When Architecture B Wins Over A

Only when:
- Expert count is very large (64+) and M variance is high
- Multi-launch overhead of Architecture A exceeds 15-20% of GEMM time
- The GEMM is already register-pressure limited (so adding codepaths doesn't hurt)

---

### Architecture C: Persistent Kernel with Runtime Tile Branching (Custom CUDA)

**Concept**: Bypass CUTLASS entirely. Write a custom CUDA persistent kernel that compiles multiple MMA tile codepaths and branches per-CTA based on assigned expert's token count.

```cpp
__global__ void hetero_moe_kernel(/* ... */) {
    // Persistent kernel: each CTA grabs work items
    while (work_available()) {
        int expert_id = grab_next_expert();
        int M_expert = tokens_per_expert[expert_id];
        
        if (M_expert <= 4) {
            run_small_tile_mma(expert_id, ...);   // e.g., 128×64×128
        } else if (M_expert <= 32) {
            run_medium_tile_mma(expert_id, ...);  // e.g., 128×128×128
        } else {
            run_large_tile_mma(expert_id, ...);   // e.g., 128×256×128
        }
    }
}
```

#### Feasibility: ⚠️ POSSIBLE, HIGHEST CEILING, HIGHEST RISK

**What works**:

1. **CTA-level branching has zero warp divergence** — all 128/256 threads in a CTA take the same branch, since each CTA processes one expert at a time.

2. **Precedent: DeepGEMM** — Uses JIT compilation to generate per-tile-config kernels at runtime. Different approach (multiple specialized kernels vs. one branching kernel) but proves the concept of tile-level optimization for MoE.

3. **Precedent: Triton** — Generates per-config kernels via autotuning. Similar to DeepGEMM's approach.

4. **Full control** — no CUTLASS template constraints. Can implement exactly the scheduler and MMA dispatch logic needed.

**What blocks**:

1. **Shared memory allocation** — must use the maximum across all tile codepaths. For 128×256×128 FP4, shared memory per CTA is ~128KB. Smaller tiles waste most of this.

2. **Register pressure** — ALL codepaths compile into the same kernel. Compiler sees all paths and allocates registers for the union. With 3 tile configs: ~30% register increase → ~20% occupancy reduction.

3. **MMA atoms are hardware instructions** — `mma.sync`, `wgmma.mma_async` (SM90+), `tcgen05.mma` (SM100+). These instructions have specific register layouts per tile shape. Runtime switching requires separate register setup per path.

4. **No CUTLASS collective optimizations** — lose TMA pipelining, warp-specialized scheduling, epilogue fusion, and years of NVIDIA performance engineering. Starting from CUTLASS's performance level with a custom kernel is a multi-month effort.

5. **SM100/SM103 GMMA complexity** — Blackwell's `tcgen05` MMA instructions have extremely specific alignment, register layout, and scheduling requirements. Implementing these correctly without CUTLASS is a research project.

#### Performance Model

```
T_arch_C = T_launch + T_gemm_custom(all_experts)

Theoretical best: each expert gets exactly optimal tile → maximum utilization.
Practical: lose 10-30% from missing CUTLASS optimizations (TMA pipelining, warp specialization).
Net: likely WORSE than Architecture A unless the custom kernel is extremely well-optimized.
```

#### Implementation Complexity: EXTREME (3-6 months)

- Custom persistent kernel from scratch
- Must implement TMA descriptor setup, warp-specialized scheduling
- Must handle SM90, SM100, SM103, SM120 separately
- Must match CUTLASS performance on the mainloop MMA (the hard part)
- Testing, debugging CUDA kernels with complex register layouts

#### When Architecture C Wins

Only when:
- Architecture A's multi-launch overhead is the primary bottleneck (many partitions needed)
- Architecture B's register pressure from CUTLASS forking is unacceptable
- Absolute maximum performance is required and months of kernel engineering are available
- The team has deep CUDA kernel development experience (GMMA, TMA, warp specialization)

---

### Architecture Comparison

| Dimension | **A: Multi-Dispatch** | **B: Custom Scheduler** | **C: Custom Kernel** |
|---|---|---|---|
| **Feasibility** | ✅ Fully validated | ⚠️ Possible, complex | ⚠️ Possible, extreme |
| **CUTLASS changes** | None | Fork CUTLASS 3.x grouped GEMM | Bypass CUTLASS |
| **Kernel launches** | N partitions | 1 | 1 |
| **Per-expert optimality** | Per-partition optimal | Per-expert optimal | Per-expert optimal |
| **Register pressure** | None (separate kernels) | Moderate (multiple codepaths) | High (all codepaths) |
| **Implementation time** | 2-3 weeks | 2-3 months | 3-6 months |
| **Risk** | Low | High | Very high |
| **SM103 NVFP4 configs** | 4 (2 tiles × 2 swap_ab) | 4 per-expert | Unlimited (custom tiles) |
| **CUDA graph compat** | Conditional (variable launch count) | Yes (single launch) | Yes (single launch) |
| **Maintenance burden** | Minimal (uses TRT-LLM API) | High (CUTLASS fork) | Very high (custom kernel) |

### Quantifying the Waste (Motivation)

For an NVFP4 expert with M_i=2 tokens and tile 128×128×128:
- **M-dimension utilization**: 2/128 = **1.6%** — 98.4% of tile M-compute is padding
- **N-dimension**: Weight dim (e.g., 7168) → **100%** utilized across tiles
- **swap_ab helps**: converts the problem to (N, M_i, K) — the small dimension becomes N, and TMA loads the large weight tensor as A for better prefetch utilization
- **Tail effects dominate**: CTAs assigned to small experts finish in one tile, but CTA overhead (TMA descriptor setup, smem allocation) is amortized over minimal useful work

With Architecture A, small-M experts get swap_ab=true + 128×128, large-M experts get swap_ab=false + 128×256. Each partition's autotuner selects the optimal tactic for its M-range.

### SM103 NVFP4 Tile Restriction — Investigation Required

SM103 FP4 only has 2 tile configs (128×128, 128×256). SM100 FP4 has 3 (adds 128×64).

The `are_tile_shapes_supported_sm100()` restriction for SM103:
```cpp
if constexpr (Arch::kMinComputeCapability == 103) {
    return is_fp4 && TileM == 128 && (TileN == 128 || TileN == 256);
}
```

**May be conservative** — SM100 supports TileN=64 for FP4 (same Blackwell family). Commented-out code suggests smaller N was explored. If SM103 can support 128×64×128, all three architectures benefit from richer config space (6 configs: 3 tiles × 2 swap_ab).

**Action**: Test SM103 with TileN=64 by relaxing the software check. If GMMA/TMA alignment allows it, submit as TRT-LLM contribution (orthogonal to Architecture A/B/C).

---

## Proposals (Prioritized)

### P0: Per-Group Autotuning Config (Python-only, immediate)

**Problem**: Both groups use `tune_max_num_tokens=8192`. NVFP4 never sees >64 tokens — wasted profiling.

**Proposal**: `tune_max_num_tokens_nvfp4 = 64`, `tune_max_num_tokens_bf16 = 8192`.

**Expected impact**: 5-15% on NVFP4 group. Zero risk.

**Complexity**: 1 day.

---

### P1: min_latency_mode for NVFP4 Group (Python-only, near-term)

**Problem**: Default path is throughput-optimized. `min_latency_mode` skips inactive experts.

**Proposal**: Conditional `min_latency_mode=True` for NVFP4 when total tokens ≤ 64.

**Expected impact**: 10-30% on NVFP4 group.

**Complexity**: 1 week. Need separate MoERunner instances, different return format handling.

**Risk**: SM100+ only constraint? CUDA graph compatibility?

---

### P2: NVFP4 Multi-Launch Expert Partitioning → Architecture A (cpp/, medium-term)

**Problem**: One tactic for all NVFP4 experts despite variable per-expert M.

**Proposal**: Partition by M-range, launch separate `runMoe()` per partition. **This is Architecture A** — see Deep Dive above for full evaluation, implementation sketch, and risk analysis.

**Expected impact**: 10-30% when M variance is high.

**Complexity**: 2-3 weeks. First `cpp/` contribution.

---

### P3: CUDA Stream Overlap (Python + minor C++, near-term)

**Problem**: Two `fused_moe()` calls are sequential.

**Proposal**: NVFP4 on aux stream, BF16 on default, synchronize before sum.

**Expected impact**: 5-15% (conditional on SM availability).

**Complexity**: 2-3 days.

**Risk**: SM contention, CUDA graph multi-stream capture.

---

### P4: C++ Fused Dispatch Kernel (cpp/, medium-term)

**Problem**: Python overhead for remap/mask/cast per call.

**Proposal**: C++ custom op: remap + mask + dual fused_moe + sum.

**Expected impact**: 3-10% + clean abstraction.

**Complexity**: 2-3 weeks.

---

### P5: Token-Count-Aware Tactic Override (Python, medium-term)

**Problem**: Autotuner tunes on total tokens, not per-expert M.

**Proposal**: Use max per-expert token count as `tuner_num_tokens` hint.

**Expected impact**: 5-10% on NVFP4.

**Complexity**: 1 week.

---

### P6: FP4 Small-N Tile Configs for SM103 (CUTLASS contribution, long-term)

**Problem**: SM103 NVFP4 only has 2 tiles (128×128, 128×256). No small-N for tiny M.

**Proposal**: Add 128×32×128 and 128×64×128 for SM103 FP4. Submit as TRT-LLM PR.

**Investigation**: SM103 GMMA constraints, TMA alignment, `MinNDimAlignmentNVFP4 = 128` may block.

**Expected impact**: 2-4× for M=1-8 experts if feasible.

**Complexity**: 3-4 weeks. High risk.

---

### P7: Fused Heterogeneous Grouped GEMM (cpp/, speculative)

**Problem**: Two separate grouped GEMMs = redundant input reads + extra sum kernel.

**Proposal**: Single kernel with per-expert precision descriptor.

**Expected impact**: 10-20%.

**Complexity**: 2-3 months. Very high risk.

---

### P8: Benchmarking Infrastructure (prerequisite)

Per-group latency breakdown, autotuner tactic analysis, SM occupancy profiling.
**Must be built first.**

---

## Strategic Summary

### Per-Expert Tile Selection — Our Contribution

CUTLASS 3.x uses compile-time tile shapes — all experts in one kernel launch share one tile config. This is the constraint we're working around.

**Three architectures evaluated** (see Deep Dive above):

| | Feasibility | Time | Expected Impact |
|---|---|---|---|
| **A: Multi-Dispatch** | ✅ Validated | 2-3 weeks | 10-30% on high-variance M |
| **B: Custom Scheduler** | ⚠️ Complex | 2-3 months | 15-40% theoretical |
| **C: Custom Kernel** | ⚠️ Extreme | 3-6 months | Highest ceiling, highest risk |

**Recommended staging**: A first (immediate, low-risk, validates the concept), then evaluate B if A's multi-launch overhead is a bottleneck, C only if kernel-level optimization becomes the project's focus.

### Optimization Levers Beyond Architecture A/B/C

1. **swap_ab** (already autotuned) — verify it's selected correctly for small-M NVFP4
2. **Per-group autotuning budget** (P0) — focus NVFP4 autotuning on relevant M buckets
3. **min_latency_mode** (P1) — skip inactive experts for NVFP4
4. **FP4 small-tile contribution** (P6) — expands config space from 4 to 6+ combinations on SM103
5. **CUDA stream overlap** (P3) — overlap NVFP4 and BF16 groups

### First cpp/ Contribution: Architecture A

Architecture A is the first code contribution to `cpp/`. It:
- Uses TRT-LLM's existing `setTactic()` / `runMoe()` API — no kernel modifications
- Can be prototyped in Python first, then ported to C++ for production
- Validates the concept of per-expert-group tile selection with real profiling data
- Opens the door to Architecture B/C if more granularity is needed

## Implementation Roadmap

```
Phase 4A — Measurement & Validation (2-3 weeks)
├── P8: Benchmarking infrastructure for per-group profiling
├── P0: Per-group tune_max_num_tokens
├── Verify: swap_ab autotuning correctness for NVFP4 small-M
├── P5: Token-count-aware tactic override
└── Measure: M variance in real workloads (how much do experts differ?)

Phase 4B — Architecture A: Multi-Dispatch Orchestrator (2-3 weeks)
├── Python prototype: partition by M-range, dual runMoe(), validate correctness
├── Profile: measure per-partition GEMM time vs single-launch baseline
├── C++ port: custom op in cpp/ directory (first contribution)
├── P3: CUDA stream overlap (try between partitions AND between NVFP4/BF16)
└── Decide: is multi-launch overhead acceptable, or do we need Architecture B?

Phase 4C — Advanced Kernel Work (gated on Phase 4B results)
├── P6: FP4 small-N tile configs (SM103 feasibility test)
├── P1: min_latency_mode for NVFP4 group
├── Architecture B evaluation (if A's launch overhead > 15% of GEMM time)
└── P7: Fused heterogeneous grouped GEMM (speculative R&D)
```

---

## Open Questions

1. **Does the autotuner work correctly for our two-call pattern?** Each call creates its own MoERunner with different weight shapes. `unique_id` does NOT include num_experts — two groups may collide in cache. **Under investigation.**

2. **swap_ab selection for NVFP4 small-M**: Does the autotuner select swap_ab=true at bucket sizes [1, 2, 4, 8] for NVFP4? If not, is there a bug or is swap_ab=false genuinely faster for FP4?

3. **SM103 GMMA minimum dimensions for FP4**: Is the SM103 tile restriction (TileN ≥ 128) a hardware or software limit? The commented-out code and SM100's support for TileN=64 suggest it might be software.

4. **min_latency_mode constraints**: Is it SM100+ only? Does it work with CUDA graphs?

5. **CUDA graph capture with multiple fused_moe calls**: Workspace sharing between calls on the same stream?

## Gotchas

- **Multi-launch overhead**: More launches + reordering overhead can outweigh tactic efficiency gains. Always **profile end-to-end**, not GEMM-only.
- **Stream overlap SM contention**: If BF16 already saturates SMs, overlapping NVFP4 competes for resources.
- **Always-on min_latency_mode can regress**: Use threshold-based switch.
- **C++ orchestrator TRT-LLM coupling**: TRT-LLM C++ API stability is not guaranteed. Keep interface minimal.
- **SM103 FP4 tile restriction may be hardware**: `are_tile_shapes_supported_sm100()` explicitly restricts SM103 to TileN∈{128, 256}. May be a real GMMA or TMA alignment constraint, not just conservative gating.
