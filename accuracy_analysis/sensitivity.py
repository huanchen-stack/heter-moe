"""
NVFP4 (FP4 E2M1) Quantization Sensitivity Analysis for MoE Models.

Uses FlashInfer's native FP4 tensor core GEMM (mm_fp4) on Blackwell GPUs
for real hardware-matched sensitivity measurement — NOT fake quantization.

Both weights AND activations are quantized to NVFP4 (W4A4).

Loads the full model via from_pretrained (requires sufficient VRAM).

Usage:
    python nf4_sensitivity.py \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --nsamples 128 \\
        --save_path ./calib/nvfp4_sensitivity.json

Output JSON format (same as MxMoE calibration):
    {layer_idx: {expert_idx: [gate_err, up_err, down_err]}}

Dependencies:
    pip install torch transformers flashinfer-python>=0.6.5 datasets tqdm

Requires: Blackwell GPU (SM120/SM100) with CUDA 12.8+ and cuDNN 9.x
"""

import os
import gc
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor

from quant import (
    make_quant_forward,
    MODEL_EXPERT_CONFIG,
    unfuse_moe_experts,
    get_experts,
)
from utils import get_calibration_data, seed_everything


# ============================================================================
# NVFP4 Sensitivity Analysis
# ============================================================================

@torch.no_grad()
def get_model_quant_error(
    model_path: str,
    dataloader: list[Tensor],
    metric: str,  # "layer_out_norm" or "model_out_norm"
    save_path: str,
    max_layers: int = -1,
    max_experts: int = -1,
    backend: str = "cudnn",
    quant_mode: str = "nvfp4",
    joint: bool = False,
    resume: bool = False,
):
    """
    NVFP4 quantization sensitivity analysis.

    For each layer × expert × linear block (gate/up/down):
      1. Compute full-precision output (BF16)
      2. Patch that ONE linear block with NVFP4 (mm_fp4)
      3. Compute NVFP4-quantized output
      4. Error = ‖quantized_output - fp_output‖₂
      5. Unpatch, restore original forward

    Two metrics:
      "layer_out_norm": error at single-layer output
      "model_out_norm": error at full-model last hidden state
    """
    from transformers import AutoModelForCausalLM

    dev = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ── Load full model on GPU ───────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.eval()

    config = model.config
    model_type = config.model_type
    num_layers = config.num_hidden_layers if max_layers < 0 else min(max_layers, config.num_hidden_layers)
    hidden_size = config.hidden_size
    num_experts = getattr(config, "num_experts", getattr(config, "num_local_experts", 1))
    expert_cfg = MODEL_EXPERT_CONFIG[model_type]
    block_names = expert_cfg["block_names"]
    block_labels = expert_cfg["labels"]
    num_samples = len(dataloader)
    layers = model.model.layers

    print(f"Quantization Sensitivity Analysis (metric={metric}, quant={quant_mode})")
    print(f"  Model: {model_path} ({model_type})")
    print(f"  Layers: {num_layers}, Hidden: {hidden_size}, Experts: {num_experts}")
    print(f"  Backend: {backend}, Samples: {num_samples}")
    print(f"  Quant mode: {quant_mode}")

    # ── Unfuse MoE experts (3D fused tensors → individual nn.Module) ──
    # Newer transformers stores expert weights as fused 3D nn.Parameter
    # (Qwen3MoeExperts). We need individual ExpertMLP modules so that
    # monkey-patching a single nn.Linear.forward actually affects the
    # layer computation during sensitivity measurement.
    for layer in layers:
        if hasattr(layer.mlp, "experts"):
            unfuse_moe_experts(layer.mlp)
    print("  Experts unfused: individual modules ready for patching")

    layer_loss: list[list[list[float]]] = [[] for _ in range(num_layers)]
    layer_loss_save = {}

    # ── Resume: load existing results from save_path ─────────────────
    if resume and os.path.exists(save_path):
        with open(save_path) as f:
            layer_loss_save = json.load(f)
        # Convert string keys back to int
        layer_loss_save = {int(k): v for k, v in layer_loss_save.items()}
        n_existing = sum(len(v) for v in layer_loss_save.values())
        print(f"  Resume: loaded {n_existing} existing expert results from {save_path}")

    ################# METRIC 1: LAYER-LOCAL SENSITIVITY (layer_out_norm) #################
    if metric == "layer_out_norm":

        # ── Prepare inputs using model's own embedding ───────────────────
        embed = model.model.embed_tokens
        inps = torch.zeros(num_samples, dataloader[0].shape[-1], hidden_size, dtype=dtype, device=dev)
        for i, batch in enumerate(dataloader):
            inps[i] = embed(batch.to(dev)).squeeze(0)

        seqlen = inps.shape[1]
        rotary_emb = model.model.rotary_emb
        position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)
        pos_emb = rotary_emb(inps[0:1], position_ids)

        attn_mask = None  # SDPA handles causal masking internally
        full_precision_outs: Tensor = torch.zeros_like(inps).to(torch.float64)
        quantized_outs: Tensor = torch.zeros_like(inps).to(torch.float64)

        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            layer = layers[layer_idx]

            # ── Capture per-expert routing distribution ──────────────────
            expert_token_counts = torch.zeros(num_experts, dtype=torch.long, device=dev)

            def _routing_hook(module, input, output,
                              _counts=expert_token_counts):
                # output layout: (router_logits, router_scores, router_indices)
                # router_indices: [num_tokens, top_k]
                indices = output[2].flatten()
                _counts.index_add_(
                    0, indices,
                    torch.ones_like(indices, dtype=torch.long),
                )

            hook_handle = None
            if hasattr(layer.mlp, "gate"):
                hook_handle = layer.mlp.gate.register_forward_hook(_routing_hook)

            with torch.inference_mode():
                # 1. get the output of the full precision layer
                for i in range(num_samples):
                    full_precision_outs[i] = layer(
                        inps[i].unsqueeze(0),
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        position_embeddings=pos_emb,
                    )[0].to(torch.float64)

            if hook_handle is not None:
                hook_handle.remove()

            # Print routing stats
            total_assignments = expert_token_counts.sum().item()
            total_tokens = num_samples * seqlen
            if total_assignments > 0:
                mean_c = expert_token_counts.float().mean().item()
                std_c = expert_token_counts.float().std().item()
                cv = std_c / mean_c if mean_c > 0 else 0
                print(f"  Layer {layer_idx}: {total_tokens} tokens × top-{total_assignments // total_tokens} "
                      f"= {total_assignments} total assignments over {num_samples} samples")
                print(f"    per-expert tokens: min={expert_token_counts.min().item()} "
                      f"max={expert_token_counts.max().item()} "
                      f"mean={mean_c:.1f} std={std_c:.1f} CV={cv:.3f}")

            # 2. get the output of the quantized layer
            experts = get_experts(layer, model_type)
            num_layer_experts = len(experts) if max_experts < 0 else min(max_experts, len(experts))

            for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                # ── Resume: skip if this layer-expert already exists ─────
                if resume and layer_idx in layer_loss_save:
                    existing = layer_loss_save[layer_idx]
                    if str(exp_id) in existing or exp_id in existing:
                        entry = existing.get(str(exp_id), existing.get(exp_id))
                        cached_err = entry["error"] if isinstance(entry, dict) else entry
                        layer_loss[layer_idx].append(cached_err)
                        exp_tokens = expert_token_counts[exp_id].item() if exp_id < len(expert_token_counts) else 0
                        print(f"  L{layer_idx}-E{exp_id} SKIP (resumed) [{exp_tokens} tokens]: {cached_err}")
                        continue

                expert = experts[exp_id]

                if joint:
                    # ── Joint mode: patch ALL linears at once, single error ───
                    orig_forwards = {}
                    for block_name in block_names:
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            continue
                        orig_forwards[block_name] = linear.forward
                        quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                        linear.forward = quant_fwd

                    with torch.inference_mode():
                        for i in range(num_samples):
                            quantized_outs[i] = layer(
                                inps[i].unsqueeze(0),
                                attention_mask=attn_mask,
                                position_ids=position_ids,
                                position_embeddings=pos_emb,
                            )[0].to(torch.float64)

                    joint_err = torch.norm(quantized_outs - full_precision_outs).item()

                    # Restore all
                    for block_name, orig_fwd in orig_forwards.items():
                        getattr(expert, block_name).forward = orig_fwd

                    expert_err = [joint_err]
                else:
                    # ── Per-linear mode: patch one linear at a time ───────────
                    expert_err = []
                    for qlinear_block, block_name in zip(block_labels, block_names):
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            expert_err.append(0.0)
                            continue

                        with torch.inference_mode():
                            orig_forward = linear.forward
                            quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                            linear.forward = quant_fwd

                            for i in range(num_samples):
                                quantized_outs[i] = layer(
                                    inps[i].unsqueeze(0),
                                    attention_mask=attn_mask,
                                    position_ids=position_ids,
                                    position_embeddings=pos_emb,
                                )[0].to(torch.float64)

                            quant_err = torch.norm(quantized_outs - full_precision_outs).item()
                            expert_err.append(quant_err)
                            linear.forward = orig_forward

                layer_loss[layer_idx].append(expert_err)
                label = "joint" if joint else "per-linear"
                exp_tokens = expert_token_counts[exp_id].item() if exp_id < len(expert_token_counts) else 0
                print(f"  L{layer_idx}-E{exp_id} {quant_mode} ({label}) "
                      f"[{exp_tokens} tokens]: {expert_err}")

            # 5. serialization (incremental)
            mean_tokens = expert_token_counts.float().mean().item() if total_assignments > 0 else 1.0
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            layer_save = {}
            for e in range(len(layer_loss[layer_idx])):
                tok = expert_token_counts[e].item() if e < len(expert_token_counts) else 0
                layer_save[e] = {
                    "error": layer_loss[layer_idx][e],
                    "tokens": tok,
                    "token_ratio": round(tok / mean_tokens, 4) if mean_tokens > 0 else 0.0,
                }
            layer_loss_save[layer_idx] = layer_save
            with open(save_path, "w") as f:
                json.dump(layer_loss_save, f)
            print(f"  Layer-{layer_idx} saved ({len(layer_save)} experts)")

            # 6. prepare the inputs for next layer
            inps = full_precision_outs.to(inps.dtype)

    ################# METRIC 2: FULL-MODEL SENSITIVITY (model_out_norm) #################
    elif metric == "model_out_norm":

        inps = torch.vstack([d.to(dev) for d in dataloader])  # [num_samples, seqlen]

        full_precision_outs: Tensor = torch.zeros(
            (num_samples, inps.shape[-1], hidden_size),
            device=dev, dtype=torch.float64,
        )
        quantized_outs: Tensor = torch.zeros_like(full_precision_outs)

        # ── Capture per-layer routing during full-precision pass ────
        per_layer_counts = {li: torch.zeros(num_experts, dtype=torch.long, device=dev)
                            for li in range(num_layers)}

        def _make_model_routing_hook(layer_idx):
            counts = per_layer_counts[layer_idx]
            def hook(module, input, output, _counts=counts):
                indices = output[2].flatten()
                _counts.index_add_(
                    0, indices,
                    torch.ones_like(indices, dtype=torch.long),
                )
            return hook

        hook_handles = []
        for li in range(num_layers):
            if hasattr(layers[li].mlp, "gate"):
                h = layers[li].mlp.gate.register_forward_hook(_make_model_routing_hook(li))
                hook_handles.append(h)

        # 1. get the output of full-precision model (hidden_states of the last layer)
        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc="Full precision output"):
                full_precision_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

        for h in hook_handles:
            h.remove()

        # Print routing stats per layer
        seqlen_model = inps.shape[-1]
        total_tokens_model = num_samples * seqlen_model
        for li in range(num_layers):
            counts = per_layer_counts[li]
            total = counts.sum().item()
            if total > 0:
                mean_c = counts.float().mean().item()
                std_c = counts.float().std().item()
                cv = std_c / mean_c if mean_c > 0 else 0
                print(f"  Layer {li}: {total_tokens_model} tokens × top-{total // total_tokens_model} "
                      f"= {total} total assignments over {num_samples} samples")
                print(f"    per-expert tokens: min={counts.min().item()} "
                      f"max={counts.max().item()} "
                      f"mean={mean_c:.1f} std={std_c:.1f} CV={cv:.3f}")

        # 2. get the model output with quantized layer
        for layer_idx in tqdm(range(num_layers), desc="Layers"):
            experts = get_experts(layers[layer_idx], model_type)
            num_layer_experts = len(experts) if max_experts < 0 else min(max_experts, len(experts))

            for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                # ── Resume: skip if this layer-expert already exists ─────
                if resume and layer_idx in layer_loss_save:
                    existing = layer_loss_save[layer_idx]
                    if str(exp_id) in existing or exp_id in existing:
                        entry = existing.get(str(exp_id), existing.get(exp_id))
                        cached_err = entry["error"] if isinstance(entry, dict) else entry
                        layer_loss[layer_idx].append(cached_err)
                        exp_tokens = per_layer_counts[layer_idx][exp_id].item() if exp_id < len(per_layer_counts[layer_idx]) else 0
                        print(f"  L{layer_idx}-E{exp_id} SKIP (resumed) [{exp_tokens} tokens]: {cached_err}")
                        continue

                expert = experts[exp_id]

                if joint:
                    # ── Joint mode: patch ALL linears at once, single error ───
                    orig_forwards = {}
                    for block_name in block_names:
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            continue
                        orig_forwards[block_name] = linear.forward
                        quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                        linear.forward = quant_fwd

                    with torch.inference_mode():
                        for i in range(num_samples):
                            quantized_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

                    joint_err = torch.norm(quantized_outs - full_precision_outs).item()

                    for block_name, orig_fwd in orig_forwards.items():
                        getattr(expert, block_name).forward = orig_fwd

                    expert_err = [joint_err]
                else:
                    # ── Per-linear mode: patch one linear at a time ───────────
                    expert_err = []
                    for qlinear_block, block_name in zip(block_labels, block_names):
                        linear: nn.Linear = getattr(expert, block_name, None)
                        if linear is None or not hasattr(linear, "weight"):
                            expert_err.append(0.0)
                            continue

                        orig_forward = linear.forward
                        quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                        linear.forward = quant_fwd

                        with torch.inference_mode():
                            for i in range(num_samples):
                                quantized_outs[i] = model.model(inps[i:i+1, :]).last_hidden_state.to(torch.float64)

                            quant_err = torch.norm(quantized_outs.sub_(full_precision_outs)).item()
                            expert_err.append(quant_err)
                            linear.forward = orig_forward

                layer_loss[layer_idx].append(expert_err)
                label = "joint" if joint else "per-linear"
                exp_tokens = per_layer_counts[layer_idx][exp_id].item() if exp_id < len(per_layer_counts[layer_idx]) else 0
                print(f"  L{layer_idx}-E{exp_id} {quant_mode} ({label}) "
                      f"[{exp_tokens} tokens]: {expert_err}")

            # 5. serialization
            counts = per_layer_counts[layer_idx]
            mean_tokens = counts.float().mean().item() if counts.sum().item() > 0 else 1.0
            layer_save = {}
            for e in range(len(layer_loss[layer_idx])):
                tok = counts[e].item() if e < len(counts) else 0
                layer_save[e] = {
                    "error": layer_loss[layer_idx][e],
                    "tokens": tok,
                    "token_ratio": round(tok / mean_tokens, 4) if mean_tokens > 0 else 0.0,
                }
            layer_loss_save[layer_idx] = layer_save
            with open(save_path, "w") as f:
                json.dump(layer_loss_save, f)
            print(f"  Layer-{layer_idx} saved ({len(layer_save)} experts)")

    ################# METRIC 3: LAYERWISE MODEL OUTPUT NORM (layerwise_model_out_norm) #################
    elif metric == "layerwise_model_out_norm":
        # For each layer i: quantize ONLY layer i (all experts), run full model
        # forward, measure ‖quantized_hidden - ref_hidden‖₂ on last hidden state.
        #
        # Loop order: samples (outer) → layers (inner) so we reuse the
        # reference forward across all layers for the same sample.
        # Cost: N_samples × (1 + N_layers) full-model forwards.

        inps = torch.vstack([d.to(dev) for d in dataloader])  # [num_samples, seqlen]

        # ── Pre-compute quantized forwards for all layers ──────────────
        print("Pre-computing quantized forwards for all layers...")
        per_layer_forwards: dict[int, list[tuple]] = {}
        for layer_idx in range(num_layers):
            experts = get_experts(layers[layer_idx], model_type)
            n = len(experts) if max_experts < 0 else min(max_experts, len(experts))
            layer_fwds = []
            for exp_id in range(n):
                expert = experts[exp_id]
                for block_name in block_names:
                    linear: nn.Linear = getattr(expert, block_name, None)
                    if linear is None or not hasattr(linear, "weight"):
                        continue
                    orig_fwd = linear.forward
                    quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                    layer_fwds.append((linear, orig_fwd, quant_fwd))
            per_layer_forwards[layer_idx] = layer_fwds
        print(f"  {sum(len(v) for v in per_layer_forwards.values())} linears across {num_layers} layers ready")

        def patch_layer(li):
            for lin, _, qfwd in per_layer_forwards[li]:
                lin.forward = qfwd

        def unpatch_layer(li):
            for lin, ofwd, _ in per_layer_forwards[li]:
                lin.forward = ofwd

        # ── Determine resume state ─────────────────────────────────────
        # Save format per layer:
        #   {"error": float, "num_samples": N, "completed_samples": M}
        # completed_samples < num_samples means partial; == num_samples means done.
        # Legacy entries without completed_samples are treated as done.
        layers_to_compute = list(range(num_layers))
        start_sample = 0
        norm_accum = {li: 0.0 for li in layers_to_compute}

        if resume:
            done_layers = set()
            partial_start = None
            for li in layers_to_compute:
                entry = layer_loss_save.get(li, layer_loss_save.get(str(li)))
                if entry is None:
                    continue
                if not isinstance(entry, dict):
                    done_layers.add(li)
                    continue
                cs = entry.get("completed_samples")
                ns = entry.get("num_samples")
                if cs is not None and ns == num_samples and cs < num_samples:
                    # Partial — restore accumulator
                    norm_accum[li] = entry["error"]
                    if partial_start is None:
                        partial_start = cs
                    else:
                        assert partial_start == cs, (
                            f"Inconsistent completed_samples: layer {li} has {cs}, expected {partial_start}"
                        )
                else:
                    done_layers.add(li)

            if done_layers:
                layers_to_compute = [li for li in layers_to_compute if li not in done_layers]
                norm_accum = {li: norm_accum[li] for li in layers_to_compute}
                print(f"  Resume: skipping {len(done_layers)} fully-computed layers")
            if partial_start is not None:
                start_sample = partial_start
                print(f"  Resume: continuing from sample {start_sample}/{num_samples}")

        # ── Save helper ────────────────────────────────────────────────
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        def _save_checkpoint(completed_samples):
            for li in layers_to_compute:
                layer_loss_save[li] = {
                    "error": norm_accum[li],
                    "num_samples": num_samples,
                    "completed_samples": completed_samples,
                }
            with open(save_path, "w") as f:
                json.dump(layer_loss_save, f, indent=2)

        # ── Lockstep: 1 ref + N_layers quant forwards per sample ──────
        remaining = num_samples - start_sample
        print(f"  Lockstep BF16 vs {quant_mode} ({remaining} remaining samples × {len(layers_to_compute)} layers)...")
        with torch.inference_mode():
            for i in tqdm(range(start_sample, num_samples), desc=f"BF16 vs {quant_mode}",
                          initial=start_sample, total=num_samples):
                input_ids = inps[i:i+1, :]

                # Reference forward (fully unpatched) → last hidden state
                ref_hidden = model.model(input_ids).last_hidden_state.to(torch.float64)

                for layer_idx in layers_to_compute:
                    patch_layer(layer_idx)

                    q_hidden = model.model(input_ids).last_hidden_state.to(torch.float64)
                    norm_accum[layer_idx] += torch.norm(q_hidden - ref_hidden).item()
                    del q_hidden

                    unpatch_layer(layer_idx)

                del ref_hidden

                # Checkpoint after each sample
                _save_checkpoint(i + 1)

        # ── Finalize and print ─────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  LAYERWISE MODEL OUTPUT NORM ({quant_mode})")
        print(f"{'='*60}")
        for layer_idx in range(num_layers):
            entry = layer_loss_save.get(layer_idx, layer_loss_save.get(str(layer_idx)))
            if entry is None:
                continue
            err = entry["error"] if isinstance(entry, dict) else entry
            print(f"  Layer {layer_idx:3d}: L2_error = {err:.4f}")
        print(f"  Saved to {save_path}")

    ################# METRIC 4: LAYERWISE PERPLEXITY (layerwise_perplexity) #################
    elif metric == "layerwise_perplexity":
        # For each layer i: quantize ONLY layer i (all experts), run full model
        # forward through the LM head, measure perplexity increase vs BF16.
        #
        # Unlike layerwise_model_out_norm (L2 on hidden states), perplexity
        # captures actual impact on predictions — the softmax is very sensitive
        # to last-layer perturbations even when L2 change is small.
        # Expected result: U-shape (early + final layers matter most).
        #
        # Cost: N_samples × (1 + N_layers) full-model forwards (same as
        # layerwise_model_out_norm but through the LM head).

        inps = torch.vstack([d.to(dev) for d in dataloader])  # [num_samples, seqlen]

        # ── Pre-compute quantized forwards for all layers ──────────────
        print("Pre-computing quantized forwards for all layers...")
        per_layer_forwards: dict[int, list[tuple]] = {}
        for layer_idx in range(num_layers):
            experts = get_experts(layers[layer_idx], model_type)
            n = len(experts) if max_experts < 0 else min(max_experts, len(experts))
            layer_fwds = []
            for exp_id in range(n):
                expert = experts[exp_id]
                for block_name in block_names:
                    linear: nn.Linear = getattr(expert, block_name, None)
                    if linear is None or not hasattr(linear, "weight"):
                        continue
                    orig_fwd = linear.forward
                    quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                    layer_fwds.append((linear, orig_fwd, quant_fwd))
            per_layer_forwards[layer_idx] = layer_fwds
        print(f"  {sum(len(v) for v in per_layer_forwards.values())} linears across {num_layers} layers ready")

        def patch_layer(li):
            for lin, _, qfwd in per_layer_forwards[li]:
                lin.forward = qfwd

        def unpatch_layer(li):
            for lin, ofwd, _ in per_layer_forwards[li]:
                lin.forward = ofwd

        # ── Determine resume state ─────────────────────────────────────
        layers_to_compute = list(range(num_layers))
        start_sample = 0
        nll_accum = {li: 0.0 for li in layers_to_compute}
        kl_accum = {li: 0.0 for li in layers_to_compute}
        token_count_accum = 0  # same for all layers (shared ref)
        ref_nll_accum = 0.0

        if resume:
            done_layers = set()
            partial_start = None
            for li in layers_to_compute:
                entry = layer_loss_save.get(li, layer_loss_save.get(str(li)))
                if entry is None:
                    continue
                if not isinstance(entry, dict):
                    done_layers.add(li)
                    continue
                cs = entry.get("completed_samples")
                ns = entry.get("num_samples")
                if cs is not None and ns == num_samples and cs < num_samples:
                    nll_accum[li] = entry.get("nll_sum", 0.0)
                    kl_accum[li] = entry.get("kl_sum", 0.0)
                    if partial_start is None:
                        partial_start = cs
                        token_count_accum = entry.get("token_count", 0)
                        ref_nll_accum = entry.get("ref_nll_sum", 0.0)
                    else:
                        assert partial_start == cs, (
                            f"Inconsistent completed_samples: layer {li} has {cs}, expected {partial_start}"
                        )
                else:
                    done_layers.add(li)

            if done_layers:
                layers_to_compute = [li for li in layers_to_compute if li not in done_layers]
                nll_accum = {li: nll_accum[li] for li in layers_to_compute}
                kl_accum = {li: kl_accum[li] for li in layers_to_compute}
                print(f"  Resume: skipping {len(done_layers)} fully-computed layers")
            if partial_start is not None:
                start_sample = partial_start
                print(f"  Resume: continuing from sample {start_sample}/{num_samples}")

        # ── Save helper ────────────────────────────────────────────────
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        def _save_checkpoint(completed_samples):
            ref_ppl = torch.exp(torch.tensor(ref_nll_accum / max(token_count_accum, 1))).item()
            for li in layers_to_compute:
                q_ppl = torch.exp(torch.tensor(nll_accum[li] / max(token_count_accum, 1))).item()
                layer_loss_save[li] = {
                    "perplexity": q_ppl,
                    "ref_perplexity": ref_ppl,
                    "ppl_increase": q_ppl - ref_ppl,
                    "ppl_increase_pct": (q_ppl / ref_ppl - 1) * 100 if ref_ppl > 0 else 0,
                    "mean_kl": kl_accum[li] / max(token_count_accum, 1),
                    "nll_sum": nll_accum[li],
                    "kl_sum": kl_accum[li],
                    "ref_nll_sum": ref_nll_accum,
                    "token_count": token_count_accum,
                    "num_samples": num_samples,
                    "completed_samples": completed_samples,
                }
            with open(save_path, "w") as f:
                json.dump(layer_loss_save, f, indent=2)

        # ── Lockstep: 1 ref + N_layers quant forwards per sample ──────
        remaining = num_samples - start_sample
        print(f"  Lockstep BF16 vs {quant_mode} ({remaining} remaining samples × {len(layers_to_compute)} layers)...")
        with torch.inference_mode():
            for i in tqdm(range(start_sample, num_samples), desc=f"BF16 vs {quant_mode} (ppl)",
                          initial=start_sample, total=num_samples):
                input_ids = inps[i:i+1, :]
                shift_targets = input_ids[:, 1:]  # [1, seqlen-1]
                n_tokens = shift_targets.numel()

                # Reference forward (fully unpatched) → log-probs
                ref_logits = model(input_ids).logits.float()
                ref_lp = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                del ref_logits

                # Reference NLL
                ref_nll = -ref_lp[:, :-1, :].gather(
                    dim=-1, index=shift_targets.unsqueeze(-1),
                ).squeeze(-1).sum().item()
                ref_nll_accum += ref_nll
                token_count_accum += n_tokens

                for layer_idx in tqdm(layers_to_compute, leave=False, desc=f"Sample {i} layers"):
                    patch_layer(layer_idx)

                    q_logits = model(input_ids).logits.float()
                    q_lp = torch.nn.functional.log_softmax(q_logits, dim=-1)
                    del q_logits

                    # Quantized NLL
                    q_nll = -q_lp[:, :-1, :].gather(
                        dim=-1, index=shift_targets.unsqueeze(-1),
                    ).squeeze(-1).sum().item()
                    nll_accum[layer_idx] += q_nll

                    # KL(P_ref || Q_quant)
                    kl = (ref_lp.exp() * (ref_lp - q_lp)).sum(dim=-1).sum().item()
                    kl_accum[layer_idx] += kl
                    del q_lp

                    unpatch_layer(layer_idx)

                del ref_lp

                # Checkpoint after each sample
                _save_checkpoint(i + 1)

        # ── Finalize and print ─────────────────────────────────────────
        ref_ppl = torch.exp(torch.tensor(ref_nll_accum / max(token_count_accum, 1))).item()
        print(f"\n{'='*60}")
        print(f"  LAYERWISE PERPLEXITY SENSITIVITY ({quant_mode})")
        print(f"  Reference perplexity (BF16): {ref_ppl:.4f}")
        print(f"{'='*60}")
        for layer_idx in range(num_layers):
            entry = layer_loss_save.get(layer_idx, layer_loss_save.get(str(layer_idx)))
            if entry is None:
                continue
            ppl = entry["perplexity"]
            delta = entry["ppl_increase"]
            pct = entry["ppl_increase_pct"]
            kl = entry["mean_kl"]
            print(f"  Layer {layer_idx:3d}: ppl={ppl:.4f}  Δppl={delta:+.4f} ({pct:+.2f}%)  KL={kl:.6f}")
        print(f"  Saved to {save_path}")

    ################# QUANTIZE ALL: patch every expert, single measurement #################
    elif metric == "quantize_all":

        inps = torch.vstack([d.to(dev) for d in dataloader])  # [num_samples, seqlen]

        # ── Pre-compute ALL quant forwards once (no per-sample re-creation) ──
        print("Pre-computing quantized forwards for all experts...")
        all_forwards: list[tuple[nn.Linear, object, object]] = []  # (linear, orig_fwd, quant_fwd)
        patched_count = 0
        for layer_idx in range(num_layers):
            experts = get_experts(layers[layer_idx], model_type)
            n = len(experts) if max_experts < 0 else min(max_experts, len(experts))
            for exp_id in range(n):
                expert = experts[exp_id]
                for block_name in block_names:
                    linear: nn.Linear = getattr(expert, block_name, None)
                    if linear is None or not hasattr(linear, "weight"):
                        continue
                    orig_fwd = linear.forward
                    quant_fwd, _ = make_quant_forward(linear, quant_mode, backend=backend)
                    all_forwards.append((linear, orig_fwd, quant_fwd))
                    patched_count += 1
        print(f"  {patched_count} linears across {num_layers} layers ready")

        def unpatch_all():
            for lin, ofwd, _ in all_forwards:
                lin.forward = ofwd

        def patch_all():
            for lin, _, qfwd in all_forwards:
                lin.forward = qfwd

        # ── Lockstep: ref + quant per sample ─────────────────────────────
        # Two forwards per sample (unavoidable — storing ref log-probs for
        # the full vocab across all samples would be hundreds of GB).
        # Quant closures are pre-computed; patch/unpatch is just pointer swaps.
        ref_nll_sum = 0.0
        ref_n_tokens = 0
        quant_nll_sum = 0.0
        quant_n_tokens = 0
        kl_sum = 0.0
        kl_n_tokens = 0

        print(f"  Lockstep BF16 vs {quant_mode} ({num_samples} samples)...")
        with torch.inference_mode():
            for i in tqdm(range(num_samples), desc=f"BF16 vs {quant_mode}"):
                input_ids = inps[i:i+1, :]
                shift_targets = input_ids[:, 1:]  # [1, seqlen-1]

                # ── Reference forward (unpatched) ────────────────────────
                unpatch_all()
                ref_logits = model(input_ids).logits.float()
                ref_lp = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                del ref_logits

                # Reference NLL
                ref_nll = -ref_lp[:, :-1, :].gather(
                    dim=-1, index=shift_targets.unsqueeze(-1),
                ).squeeze(-1)
                ref_nll_sum += ref_nll.sum().item()
                ref_n_tokens += ref_nll.numel()
                del ref_nll

                # ── Quantized forward (patched) ──────────────────────────
                patch_all()
                q_logits = model(input_ids).logits.float()
                q_lp = torch.nn.functional.log_softmax(q_logits, dim=-1)
                del q_logits

                # Quantized NLL
                q_nll = -q_lp[:, :-1, :].gather(
                    dim=-1, index=shift_targets.unsqueeze(-1),
                ).squeeze(-1)
                quant_nll_sum += q_nll.sum().item()
                quant_n_tokens += q_nll.numel()
                del q_nll

                # ── KL(P_ref || Q_quant) ─────────────────────────────────
                ref_p = ref_lp.exp()
                kl_per_token = (ref_p * (ref_lp - q_lp)).sum(dim=-1)
                kl_sum += kl_per_token.sum().item()
                kl_n_tokens += kl_per_token.numel()
                del ref_lp, ref_p, q_lp, kl_per_token

        unpatch_all()

        ref_ppl = torch.exp(torch.tensor(ref_nll_sum / max(ref_n_tokens, 1))).item()
        quant_ppl = torch.exp(torch.tensor(quant_nll_sum / max(quant_n_tokens, 1))).item()
        mean_kl = kl_sum / max(kl_n_tokens, 1)

        print(f"\n{'='*60}")
        print(f"  FULL MODEL QUANTIZATION RESULTS ({quant_mode})")
        print(f"{'='*60}")
        print(f"  Reference perplexity (BF16):  {ref_ppl:.4f}")
        print(f"  Quantized perplexity ({quant_mode:>5}):  {quant_ppl:.4f}")
        print(f"  Perplexity increase:          {quant_ppl - ref_ppl:+.4f} ({(quant_ppl/ref_ppl - 1)*100:+.2f}%)")
        print(f"  Mean KL divergence:           {mean_kl:.6f}")
        print(f"  Samples: {num_samples}, Tokens: {kl_n_tokens}")
        print(f"{'='*60}")

        # Save results
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        results = {
            "quant_mode": quant_mode,
            "ref_perplexity": ref_ppl,
            "quant_perplexity": quant_ppl,
            "ppl_increase": quant_ppl - ref_ppl,
            "ppl_increase_pct": (quant_ppl / ref_ppl - 1) * 100,
            "mean_kl_divergence": mean_kl,
            "num_samples": num_samples,
            "num_tokens": kl_n_tokens,
            "num_layers": num_layers,
            "num_experts": num_experts,
        }
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {save_path}")

    else:
        raise ValueError(f"Unknown metric: {metric}")

    del model
    gc.collect(); torch.cuda.empty_cache()

    return layer_loss


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantization Sensitivity Analysis for MoE Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization modes:
  nvfp4   W4A4  — real FP4 GEMM via FlashInfer mm_fp4 (Blackwell GPU required)
  fp8     W8A8  — real FP8 GEMM via FlashInfer gemm_fp8_nt_groupwise
  a16w4   W4A16 — fake-quantize weight to FP4 E2M1 (per-block), BF16 activation

Metrics:
  layer_out_norm  — L2 norm of single-layer output difference (fast, per-layer)
  model_out_norm  — L2 norm of full-model hidden state difference
  layerwise_model_out_norm — Quantize one layer at a time, measure L2 error on last hidden state
  layerwise_perplexity     — Quantize one layer at a time, measure perplexity increase + KL
  quantize_all    — Quantize ALL experts, report perplexity + KL (whole-model baseline)

Examples:
  # Full-model baseline: quantize everything, get perplexity
  python nf4_sensitivity.py --model_path /path/to/model --quant_mode nvfp4 --metric quantize_all
  python nf4_sensitivity.py --model_path /path/to/model --quant_mode fp8 --metric quantize_all

  # Layerwise sensitivity: quantize one layer at a time
  python nf4_sensitivity.py --model_path /path/to/model --quant_mode nvfp4 --metric layerwise_model_out_norm

  # Quick test (1 layer, 4 experts)
  python nf4_sensitivity.py --model_path /path/to/model --max_layers 1 --max_experts 4
        """,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--quant_mode", type=str, default="nvfp4",
                        choices=["nvfp4", "fp8", "a16w4"],
                        help="Quantization mode (default: nvfp4)")
    parser.add_argument("--metric", type=str, default="layer_out_norm",
                        choices=["layer_out_norm", "model_out_norm",
                                 "layerwise_model_out_norm", "layerwise_perplexity",
                                 "quantize_all"],
                        help="Sensitivity metric (default: layer_out_norm). "
                             "layerwise_model_out_norm quantizes one layer at a time and measures L2 error. "
                             "layerwise_perplexity quantizes one layer at a time and measures perplexity increase. "
                             "quantize_all patches ALL experts and reports perplexity + KL.")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--seqlen", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save_path", type=str, default="./calib/sensitivity.json",
                        help="Output JSON path")
    parser.add_argument("--max_layers", type=int, default=-1,
                        help="Max layers to process (-1 = all)")
    parser.add_argument("--max_experts", type=int, default=-1,
                        help="Max experts to analyze per layer (-1 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip layer-expert pairs that already exist in the output JSON. "
                             "Useful for resuming interrupted runs.")
    parser.add_argument("--joint", action="store_true",
                        help="Quantize all three linears (gate/up/down) simultaneously "
                             "instead of one at a time. Produces a single error per expert "
                             "and runs ~3x faster.")
    parser.add_argument("--backend", type=str, default="cudnn",
                        choices=["cudnn", "cutlass", "auto"],
                        help="FlashInfer mm_fp4 backend for nvfp4 mode (default: cudnn)")

    args = parser.parse_args()
    seed_everything(args.seed)

    dataloader = get_calibration_data(args.model_path, args.nsamples, args.seed, args.seqlen)

    get_model_quant_error(
        model_path=args.model_path,
        dataloader=dataloader,
        metric=args.metric,
        save_path=args.save_path,
        max_layers=args.max_layers,
        max_experts=args.max_experts,
        backend=args.backend,
        quant_mode=args.quant_mode,
        joint=args.joint,
        resume=args.resume,
    )
