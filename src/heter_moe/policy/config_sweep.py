"""Configuration sweep for heterogeneous MoE kernel parameters.

Sweeps: nvfp4_ratio x batch_size x tune_max_num_tokens x min_latency_mode.
Each configuration is measured by calling ``torch.ops.trtllm.fused_moe()`` per
precision group — the same CUTLASS grouped-GEMM kernel ``CutlassFusedMoE`` uses.

The kernel's internal AutoTuner selects optimal GEMM tactics (swap_ab, tile
sizes, cluster shapes) during warmup.  We measure wall-clock time with CUDA
synchronization to find the best hetero + kernel config.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from heter_moe.ops.hetero_moe import create_subset_weights
from heter_moe.policy.scheduler import PrecisionScheduler
from heter_moe.policy.strategies import RandomStrategy


@dataclass
class KernelConfig:
    """Kernel-level parameters being tuned per precision group."""

    tune_max_num_tokens: int = 8192
    min_latency_mode: bool = False


@dataclass
class SweepConfig:
    """Full heterogeneous MoE configuration being swept."""

    nvfp4_ratio: float
    batch_size: int
    top_k: int
    nvfp4_kernel: KernelConfig = field(default_factory=KernelConfig)
    bf16_kernel: KernelConfig = field(default_factory=KernelConfig)


@dataclass
class SweepResult:
    """Result of sweeping a single configuration."""

    config: SweepConfig
    throughput_tokens_per_sec: float
    latency_ms: float
    nvfp4_time_ms: float
    bf16_time_ms: float


class HeteroConfigSweeper:
    """Profile CUTLASS kernel tactics for the two-group heterogeneous MoE.

    For each candidate configuration:

    1. Create a ``DispatchPlan`` via ``PrecisionScheduler`` + ``RandomStrategy``.
    2. Subset weights per group.
    3. Build mask-based routing (``[N, top_k]`` shape).
    4. Call ``torch.ops.trtllm.fused_moe()`` per group.
    5. Measure wall-clock time with CUDA sync.

    Usage::

        sweeper = HeteroConfigSweeper(
            num_experts=256, hidden_size=7168, intermediate_size=2048,
        )
        best = sweeper.sweep(
            nvfp4_ratios=[0.25, 0.5, 0.75],
            batch_sizes=[1, 4, 8, 16, 32],
        )
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        warmup_runs: int = 10,
        measure_runs: int = 20,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs
        self._results: List[SweepResult] = []

    def sweep(
        self,
        nvfp4_ratios: Optional[List[float]] = None,
        batch_sizes: Optional[List[int]] = None,
        top_k: int = 8,
        tune_max_num_tokens_candidates: Optional[List[int]] = None,
        try_min_latency: bool = True,
        device: str = "cuda",
        seed: int = 42,
    ) -> SweepConfig:
        """Run configuration sweep.

        Returns the ``SweepConfig`` with lowest latency.
        """
        if nvfp4_ratios is None:
            nvfp4_ratios = [0.25, 0.5, 0.75]
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64]
        if tune_max_num_tokens_candidates is None:
            tune_max_num_tokens_candidates = [4096, 8192, 16384]

        self._results = []

        # Build kernel config grid
        kernel_configs: List[KernelConfig] = []
        for tmt in tune_max_num_tokens_candidates:
            kernel_configs.append(KernelConfig(
                tune_max_num_tokens=tmt, min_latency_mode=False,
            ))
            if try_min_latency:
                kernel_configs.append(KernelConfig(
                    tune_max_num_tokens=tmt, min_latency_mode=True,
                ))

        for batch_size in batch_sizes:
            for nvfp4_ratio in nvfp4_ratios:
                for kc in kernel_configs:
                    config = SweepConfig(
                        nvfp4_ratio=nvfp4_ratio,
                        batch_size=batch_size,
                        top_k=top_k,
                        nvfp4_kernel=kc,
                        bf16_kernel=kc,
                    )
                    result = self._measure_config(config, device, seed)
                    self._results.append(result)

                    mode = "minlat" if kc.min_latency_mode else "maxtp"
                    print(
                        f"  bs={batch_size:4d}  nvfp4={nvfp4_ratio:.0%}  "
                        f"tmt={kc.tune_max_num_tokens:5d}  {mode:6s}  "
                        f"-> {result.latency_ms:7.2f} ms  "
                        f"({result.throughput_tokens_per_sec:.0f} tok/s)"
                    )

        best = min(self._results, key=lambda r: r.latency_ms)
        print(
            f"\nBest: bs={best.config.batch_size}  "
            f"nvfp4={best.config.nvfp4_ratio:.0%}  "
            f"tmt={best.config.nvfp4_kernel.tune_max_num_tokens}  "
            f"min_latency={best.config.nvfp4_kernel.min_latency_mode}  "
            f"-> {best.latency_ms:.2f} ms"
        )
        return best.config

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _measure_config(
        self,
        config: SweepConfig,
        device: str,
        seed: int,
    ) -> SweepResult:
        """Measure one configuration using actual CUTLASS kernels."""
        num_tokens = config.batch_size

        # Synthetic inputs
        hidden_states = torch.randn(
            num_tokens, self.hidden_size,
            device=device, dtype=torch.bfloat16,
        )
        token_selected_experts = torch.randint(
            0, self.num_experts, (num_tokens, config.top_k),
            device=device, dtype=torch.int32,
        )
        token_final_scales = torch.softmax(
            torch.randn(num_tokens, config.top_k, device=device), dim=-1,
        ).to(torch.float32)

        # Weights (TRT-LLM convention: w3_w1 = gate+up fused)
        w3_w1_weight = torch.randn(
            self.num_experts, 2 * self.intermediate_size, self.hidden_size,
            device=device, dtype=torch.bfloat16,
        )
        w2_weight = torch.randn(
            self.num_experts, self.hidden_size, self.intermediate_size,
            device=device, dtype=torch.bfloat16,
        )

        # Precision split via scheduler + strategy
        strategy = RandomStrategy(
            nvfp4_ratio=config.nvfp4_ratio,
            seed=seed,
        )
        scheduler = PrecisionScheduler(
            num_experts=self.num_experts,
            strategy=strategy,
        )
        plan = scheduler.schedule()
        num_nvfp4 = plan.num_nvfp4
        num_bf16 = plan.num_bf16

        # Subset weights
        nvfp4_w3_w1 = create_subset_weights(
            w3_w1_weight, plan.nvfp4_expert_ids,
        )
        nvfp4_w2 = create_subset_weights(
            w2_weight, plan.nvfp4_expert_ids,
        )
        bf16_w3_w1 = create_subset_weights(
            w3_w1_weight, plan.bf16_expert_ids,
        )
        bf16_w2 = create_subset_weights(
            w2_weight, plan.bf16_expert_ids,
        )

        # Global -> local remap tables
        nvfp4_remap = torch.full(
            (self.num_experts,), num_nvfp4,
            device=device, dtype=torch.int32,
        )
        nvfp4_ids = torch.tensor(
            plan.nvfp4_expert_ids, device=device, dtype=torch.long,
        )
        nvfp4_remap[nvfp4_ids] = torch.arange(
            num_nvfp4, device=device, dtype=torch.int32,
        )

        bf16_remap = torch.full(
            (self.num_experts,), num_bf16,
            device=device, dtype=torch.int32,
        )
        bf16_ids = torch.tensor(
            plan.bf16_expert_ids, device=device, dtype=torch.long,
        )
        bf16_remap[bf16_ids] = torch.arange(
            num_bf16, device=device, dtype=torch.int32,
        )

        # Mask-based routing
        nvfp4_experts = nvfp4_remap[
            token_selected_experts.long()
        ].to(torch.int32)
        nvfp4_valid = nvfp4_experts < num_nvfp4
        nvfp4_scales = torch.where(
            nvfp4_valid, token_final_scales,
            torch.zeros_like(token_final_scales),
        )

        bf16_experts = bf16_remap[
            token_selected_experts.long()
        ].to(torch.int32)
        bf16_valid = bf16_experts < num_bf16
        bf16_scales = torch.where(
            bf16_valid, token_final_scales,
            torch.zeros_like(token_final_scales),
        )

        def run_group(
            x: torch.Tensor,
            experts: torch.Tensor,
            scales: torch.Tensor,
            w3_w1: torch.Tensor,
            w2: torch.Tensor,
            kc: KernelConfig,
        ) -> torch.Tensor:
            return torch.ops.trtllm.fused_moe(
                x, experts, scales,
                w3_w1, None, w2, None,
                x.dtype,
                quant_scales=[],
                input_sf=None,
                swizzled_input_sf=True,
                swiglu_alpha=None,
                swiglu_beta=None,
                swiglu_limit=None,
                tp_size=1, tp_rank=0,
                ep_size=1, ep_rank=0,
                cluster_size=1, cluster_rank=0,
                enable_alltoall=False,
                use_deepseek_fp8_block_scale=False,
                use_w4_group_scaling=False,
                use_int8_woq_per_channel=False,
                use_mxfp8_act_scaling=False,
                min_latency_mode=kc.min_latency_mode,
                use_fused_finalize=True,
                tune_max_num_tokens=kc.tune_max_num_tokens,
                tuner_num_tokens=None,
                tuner_top_k=None,
                activation_type=0,  # Swiglu
                unpadded_hidden_size=self.hidden_size,
                out_tensor=None,
            )[0]

        # Warmup (lets kernel's internal AutoTuner profile tactics)
        for _ in range(self.warmup_runs):
            if num_nvfp4 > 0:
                run_group(hidden_states, nvfp4_experts, nvfp4_scales,
                          nvfp4_w3_w1, nvfp4_w2, config.nvfp4_kernel)
            if num_bf16 > 0:
                run_group(hidden_states, bf16_experts, bf16_scales,
                          bf16_w3_w1, bf16_w2, config.bf16_kernel)

        # Measure
        torch.cuda.synchronize()
        total_nvfp4_ms = 0.0
        total_bf16_ms = 0.0

        for _ in range(self.measure_runs):
            if num_nvfp4 > 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_group(hidden_states, nvfp4_experts, nvfp4_scales,
                          nvfp4_w3_w1, nvfp4_w2, config.nvfp4_kernel)
                torch.cuda.synchronize()
                total_nvfp4_ms += (time.perf_counter() - t0) * 1000

            if num_bf16 > 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_group(hidden_states, bf16_experts, bf16_scales,
                          bf16_w3_w1, bf16_w2, config.bf16_kernel)
                torch.cuda.synchronize()
                total_bf16_ms += (time.perf_counter() - t0) * 1000

        latency_ms = (total_nvfp4_ms + total_bf16_ms) / self.measure_runs
        throughput = (num_tokens * self.measure_runs) / (
            (total_nvfp4_ms + total_bf16_ms) / 1000
        ) if (total_nvfp4_ms + total_bf16_ms) > 0 else 0.0

        return SweepResult(
            config=config,
            throughput_tokens_per_sec=throughput,
            latency_ms=latency_ms,
            nvfp4_time_ms=total_nvfp4_ms / self.measure_runs,
            bf16_time_ms=total_bf16_ms / self.measure_runs,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> List[SweepResult]:
        """Return all collected sweep results."""
        return list(self._results)

    def save_results(self, path: str) -> None:
        """Save results to JSON."""
        data = [
            {
                "config": {
                    "nvfp4_ratio": r.config.nvfp4_ratio,
                    "batch_size": r.config.batch_size,
                    "top_k": r.config.top_k,
                    "nvfp4_kernel": {
                        "tune_max_num_tokens": r.config.nvfp4_kernel.tune_max_num_tokens,
                        "min_latency_mode": r.config.nvfp4_kernel.min_latency_mode,
                    },
                    "bf16_kernel": {
                        "tune_max_num_tokens": r.config.bf16_kernel.tune_max_num_tokens,
                        "min_latency_mode": r.config.bf16_kernel.min_latency_mode,
                    },
                },
                "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                "latency_ms": r.latency_ms,
                "nvfp4_time_ms": r.nvfp4_time_ms,
                "bf16_time_ms": r.bf16_time_ms,
            }
            for r in self._results
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
