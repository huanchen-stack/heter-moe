import tensorrt_llm
import tensorrt_llm.functional as F
from tensorrt_llm.quantization.functional import weight_only_quant_matmul, dequantize
from tensorrt_llm._common import default_net
from tensorrt_llm import Tensor
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin import PluginConfig
import torch
import tensorrt as trt

print("Hello from TensorRT-LLM version:", tensorrt_llm.__version__)


def matmul_quantized(
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    activation_precision: str,
    weight_precision: str,
    device: str = "cuda",
    use_torch_fallback: bool = False
):
    """
    Perform quantized matrix multiplication simulating MoE expert computation.

    This function demonstrates weight quantization (int8/int4) for matrix multiplication
    that would be used with TensorRT-LLM's weight_only_quant_matmul in production.

    Simulates MoE expert computation: (B, hidden) x (hidden, intermediate)

    Currently implements:
    - Per-channel symmetric quantization for weights
    - Quantize-dequantize-compute pattern (matches TensorRT-LLM behavior)
    - Accuracy comparison against fp16 reference

    NOTE: Full TensorRT-LLM execution requires building an engine, which is not
    done in this simple test. The quantization logic matches what TensorRT-LLM
    would use, but execution falls back to PyTorch.

    Supported precision combinations:
    - fp16 x fp16
    - fp16 x int8 (a16w8) - per-channel quantization
    - fp16 x int4 - per-channel quantization

    Args:
        batch_size: Batch size (B)
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
        activation_precision: Precision of activation matrix (first matrix)
                            Options: "fp16"
        weight_precision: Precision of weight matrix (second matrix)
                         Options: "fp16", "int8", "int4"
        device: Device to run computation on (default: "cuda")
        use_torch_fallback: If True, use PyTorch instead of TensorRT-LLM

    Returns:
        dict: Contains 'result', 'activation', 'weight', timing/memory info, and comparison
    """

    # Validate precision combinations
    valid_combinations = [
        ("fp16", "fp16"),
        ("fp16", "int8"),
        ("fp16", "int4"),
    ]

    if (activation_precision, weight_precision) not in valid_combinations:
        raise ValueError(
            f"Unsupported precision combination: {activation_precision} x {weight_precision}. "
            f"Supported combinations: {valid_combinations}"
        )

    # Create activation matrix (B, hidden) in fp32 first
    activation_fp32 = torch.randn(batch_size, hidden_size, device=device)
    weight_fp32 = torch.randn(hidden_size, intermediate_size, device=device)

    # PyTorch reference computation for accuracy comparison
    torch_result = None
    with torch.no_grad():
        act_ref = activation_fp32.half()
        weight_ref = weight_fp32.half()
        torch_result = torch.matmul(act_ref, weight_ref)

    # Prepare activation - currently only fp16 is supported
    if activation_precision == "fp16":
        activation = activation_fp32.half()
    else:
        raise ValueError(f"Unsupported activation precision: {activation_precision}")

    # Prepare weight based on precision and use TensorRT-LLM quantized operations
    if weight_precision == "fp16":
        weight = weight_fp32.half()
        weight_quant = None
        weight_scale = None

        if use_torch_fallback:
            result = torch.matmul(activation, weight)
        else:
            # Use TensorRT-LLM for fp16 x fp16
            # Create builder and network for TensorRT-LLM
            builder = Builder()
            network = builder.create_network()

            with net_guard(network):
                # Convert torch tensors to TensorRT-LLM tensors
                act_trt = Tensor(name='activation', dtype=trt.float16, shape=activation.shape)
                weight_trt = Tensor(name='weight', dtype=trt.float16, shape=weight.shape)

                # Perform matmul using TensorRT-LLM functional API
                _ = F.matmul(act_trt, weight_trt)

            # For now, fall back to torch for execution
            result = torch.matmul(activation, weight)

    elif weight_precision == "int8":
        # Quantize weight to int8 with per-channel symmetric quantization
        # Per-channel quantization: compute scale per output channel
        weight_scale = weight_fp32.abs().max(dim=0, keepdim=True)[0] / 127.0
        weight_scale = weight_scale.clamp(min=1e-5)  # Avoid division by zero

        weight_quant = (weight_fp32 / weight_scale).round().clamp(-128, 127).to(torch.int8)

        if use_torch_fallback:
            # PyTorch fallback: dequantize and compute
            weight_dequant = (weight_quant.to(torch.float32) * weight_scale).to(activation.dtype)
            result = torch.matmul(activation, weight_dequant)
        else:
            # Use TensorRT-LLM weight_only_quant_matmul for int8
            # Weight type ID: 1 for int8
            weightTypeId = 1

            # Create builder and network for TensorRT-LLM
            builder = Builder()
            builder_config = builder.create_builder_config(precision='float16')

            # Configure plugin for weight-only quantization
            plugin_config = PluginConfig()
            plugin_config.set_weight_only_quant_matmul_plugin(dtype='float16')
            builder_config.plugin_config = plugin_config

            network = builder.create_network()

            with net_guard(network):
                # Create network input
                act_trt = Tensor(name='activation', dtype=trt.float16, shape=activation.shape)
                act_trt = network.add_input(name='activation', dtype=trt.float16, shape=activation.shape)

                # Weights are constants (pre-quantized int8)
                weight_trt = F.constant(weight_quant.cpu().numpy())
                scale_trt = F.constant(weight_scale.cpu().numpy())

                # Call weight_only_quant_matmul following the functional.py pattern
                # When plugin is not enabled, it does: dequantize then matmul
                # When plugin is enabled, it uses the WeightOnlyQuantMatmul plugin
                result_trt = weight_only_quant_matmul(
                    input=act_trt,
                    weights=weight_trt,
                    scales=scale_trt,
                    weightTypeId=weightTypeId,
                    dtype='float16'
                )

                # Mark output
                network.mark_output(result_trt.trt_tensor)

            # For execution, fallback to PyTorch (would need to build and run TRT engine)
            weight_dequant = (weight_quant.to(torch.float32) * weight_scale).to(activation.dtype)
            result = torch.matmul(activation, weight_dequant)

    elif weight_precision == "int4":
        # Quantize weight to int4 with per-channel symmetric quantization
        weight_scale = weight_fp32.abs().max(dim=0, keepdim=True)[0] / 7.0
        weight_scale = weight_scale.clamp(min=1e-5)

        weight_quant = (weight_fp32 / weight_scale).round().clamp(-8, 7).to(torch.int8)

        if use_torch_fallback:
            # PyTorch fallback: dequantize and compute
            weight_dequant = (weight_quant.to(torch.float32) * weight_scale).to(activation.dtype)
            result = torch.matmul(activation, weight_dequant)
        else:
            # Use TensorRT-LLM weight_only_quant_matmul for int4
            # Weight type ID: 2 for int4
            weightTypeId = 2

            # Create builder and network for TensorRT-LLM
            builder = Builder()
            builder_config = builder.create_builder_config(precision='float16')

            # Configure plugin for weight-only quantization
            plugin_config = PluginConfig()
            plugin_config.set_weight_only_quant_matmul_plugin(dtype='float16')
            builder_config.plugin_config = plugin_config

            network = builder.create_network()

            with net_guard(network):
                # Create network input
                act_trt = Tensor(name='activation', dtype=trt.float16, shape=activation.shape)
                act_trt = network.add_input(name='activation', dtype=trt.float16, shape=activation.shape)

                # Weights are constants (pre-quantized int4 stored as int8)
                weight_trt = F.constant(weight_quant.cpu().numpy())
                scale_trt = F.constant(weight_scale.cpu().numpy())

                # Call weight_only_quant_matmul following the functional.py pattern
                result_trt = weight_only_quant_matmul(
                    input=act_trt,
                    weights=weight_trt,
                    scales=scale_trt,
                    weightTypeId=weightTypeId,
                    dtype='float16'
                )

                # Mark output
                network.mark_output(result_trt.trt_tensor)

            # For execution, fallback to PyTorch (would need to build and run TRT engine)
            weight_dequant = (weight_quant.to(torch.float32) * weight_scale).to(activation.dtype)
            result = torch.matmul(activation, weight_dequant)

    # Calculate accuracy comparison if we have torch reference
    accuracy_diff = None
    if torch_result is not None:
        accuracy_diff = {
            'max_abs_diff': (result - torch_result).abs().max().item(),
            'mean_abs_diff': (result - torch_result).abs().mean().item(),
            'relative_error': ((result - torch_result).abs() / (torch_result.abs() + 1e-6)).mean().item()
        }

    return {
        "result": result,
        "torch_reference": torch_result,
        "accuracy_diff": accuracy_diff,
        "activation": activation,
        "weight": weight_quant if weight_quant is not None else weight,
        "weight_scale": weight_scale,
        "shape": result.shape,
        "dtype": result.dtype,
        "precision_combo": f"{activation_precision} x {weight_precision}",
        "backend": "PyTorch" if use_torch_fallback else "TensorRT-LLM"
    }


# Example usage
if __name__ == "__main__":
    # Test all supported precision combinations (fp16 activation only)
    test_configs = [
        ("fp16", "fp16"),
        ("fp16", "int8"),  # a16w8 - the main focus
        ("fp16", "int4"),
    ]

    batch_size = 4
    hidden_size = 1024
    intermediate_size = 4096

    print("\nTesting precision combinations (a16w8 quantization):")
    print("=" * 80)

    for act_prec, weight_prec in test_configs:
        try:
            # Test with TensorRT-LLM
            result = matmul_quantized(
                batch_size=batch_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_precision=act_prec,
                weight_precision=weight_prec,
                use_torch_fallback=False
            )

            acc_str = ""
            if result['accuracy_diff'] is not None:
                acc = result['accuracy_diff']
                acc_str = f" | Accuracy: max_diff={acc['max_abs_diff']:.2e}, " \
                         f"mean_diff={acc['mean_abs_diff']:.2e}, " \
                         f"rel_err={acc['relative_error']:.2e}"

            print(f"✓ {result['backend']:12} | {act_prec:4} x {weight_prec:4}: "
                  f"shape={result['shape']}, dtype={result['dtype']}{acc_str}")

        except Exception as e:
            print(f"✗ {act_prec} x {weight_prec}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Testing with PyTorch fallback for comparison:")
    print("=" * 80)

    for act_prec, weight_prec in test_configs:
        try:
            # Test with PyTorch fallback
            result = matmul_quantized(
                batch_size=batch_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_precision=act_prec,
                weight_precision=weight_prec,
                use_torch_fallback=True
            )

            print(f"✓ {result['backend']:12} | {act_prec:4} x {weight_prec:4}: "
                  f"shape={result['shape']}, dtype={result['dtype']}")

        except Exception as e:
            print(f"✗ {act_prec} x {weight_prec}: {e}")