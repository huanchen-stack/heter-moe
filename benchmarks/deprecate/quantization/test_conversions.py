"""
Example usage of conversion_kernels.cu

Demonstrates all conversion operations:
- FP16 -> FP8
- FP16 -> INT4
- INT4 -> FP8
- INT4 -> FP16
- FP8 -> FP16
"""

import torch

# Assuming the library is compiled and registered as 'conversion_ops'
# Compile with: torch.utils.cpp_extension.load(...)


def test_fp16_to_fp8():
    """Test FP16 -> FP8 quantization"""
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    group_size = 128

    # Create input tensor
    input_fp16 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')

    # Prepare output tensors
    output_fp8 = torch.empty_like(input_fp16, dtype=torch.float8_e4m3fn)
    num_groups = input_fp16.numel() // group_size
    scales = torch.empty(num_groups, dtype=torch.float32, device='cuda')

    # Quantize
    torch.ops.conversion_ops.fp16_to_fp8(input_fp16, output_fp8, scales, group_size)

    print(f"FP16 -> FP8: Input shape {input_fp16.shape}, Output shape {output_fp8.shape}")
    print(f"Scales shape: {scales.shape}")
    print(f"Memory saved: {input_fp16.nbytes / output_fp8.nbytes:.2f}x")


def test_fp16_to_int4():
    """Test FP16 -> INT4 quantization with bitpacking"""
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    group_size = 128

    # Create input tensor
    input_fp16 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')

    # Prepare output tensors (INT4 is packed, so half the size)
    output_int4 = torch.empty(input_fp16.numel() // 2, dtype=torch.uint8, device='cuda')
    num_groups = input_fp16.numel() // group_size
    scales = torch.empty(num_groups, dtype=torch.float32, device='cuda')

    # Quantize
    torch.ops.conversion_ops.fp16_to_int4(input_fp16, output_int4, scales, group_size)

    print(f"\nFP16 -> INT4: Input shape {input_fp16.shape}")
    print(f"Output packed shape: {output_int4.shape}")
    print(f"Memory saved: {input_fp16.nbytes / output_int4.nbytes:.2f}x")


def test_int4_to_fp8():
    """Test INT4 -> FP8 dequantization"""
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    group_size = 128

    # Create INT4 packed input (simulated)
    total_elements = batch_size * seq_len * hidden_dim
    input_int4 = torch.randint(0, 255, (total_elements // 2,), dtype=torch.uint8, device='cuda')

    # Create scales
    num_groups = total_elements // group_size
    scales = torch.rand(num_groups, dtype=torch.float32, device='cuda') * 0.1

    # Prepare output
    output_fp8 = torch.empty(total_elements, dtype=torch.float8_e4m3fn, device='cuda')

    # Dequantize
    torch.ops.conversion_ops.int4_to_fp8(input_int4, output_fp8, scales, group_size)

    print(f"\nINT4 -> FP8: Input packed shape {input_int4.shape}, Output shape {output_fp8.shape}")


def test_int4_to_fp16():
    """Test INT4 -> FP16 dequantization"""
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    group_size = 128

    # Create INT4 packed input (simulated)
    total_elements = batch_size * seq_len * hidden_dim
    input_int4 = torch.randint(0, 255, (total_elements // 2,), dtype=torch.uint8, device='cuda')

    # Create scales
    num_groups = total_elements // group_size
    scales = torch.rand(num_groups, dtype=torch.float32, device='cuda') * 0.1

    # Prepare output
    output_fp16 = torch.empty(total_elements, dtype=torch.float16, device='cuda')

    # Dequantize
    torch.ops.conversion_ops.int4_to_fp16(input_int4, output_fp16, scales, group_size)

    print(f"\nINT4 -> FP16: Input packed shape {input_int4.shape}")
    print(f"Output shape: {output_fp16.shape}")


def test_fp8_to_fp16():
    """Test FP8 -> FP16 dequantization"""
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    group_size = 128

    # Create FP8 input
    total_elements = batch_size * seq_len * hidden_dim
    input_fp8 = torch.randn(total_elements, dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn)

    # Create scales
    num_groups = total_elements // group_size
    scales = torch.rand(num_groups, dtype=torch.float32, device='cuda') * 0.1

    # Prepare output
    output_fp16 = torch.empty_like(input_fp8, dtype=torch.float16)

    # Dequantize
    torch.ops.conversion_ops.fp8_to_fp16(input_fp8, output_fp16, scales, group_size)

    print(f"\nFP8 -> FP16: Input shape {input_fp8.shape}, Output shape {output_fp16.shape}")


def test_round_trip_int4():
    """Test FP16 -> INT4 -> FP16 round trip"""
    group_size = 128
    total_elements = 1024

    # Original data
    original = torch.randn(total_elements, dtype=torch.float16, device='cuda')

    # Quantize to INT4
    packed_int4 = torch.empty(total_elements // 2, dtype=torch.uint8, device='cuda')
    num_groups = total_elements // group_size
    scales = torch.empty(num_groups, dtype=torch.float32, device='cuda')
    torch.ops.conversion_ops.fp16_to_int4(original, packed_int4, scales, group_size)

    # Dequantize back to FP16
    reconstructed = torch.empty_like(original)
    torch.ops.conversion_ops.int4_to_fp16(packed_int4, reconstructed, scales, group_size)

    # Calculate error
    mse = torch.mean((original - reconstructed) ** 2).item()
    max_error = torch.max(torch.abs(original - reconstructed)).item()

    print(f"\nRound-trip INT4 test:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Compression: {original.nbytes / packed_int4.nbytes:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Conversion Kernels")
    print("=" * 60)

    # Note: These will only work after compiling the CUDA extension
    # Uncomment when ready to test:

    # test_fp16_to_fp8()
    # test_fp16_to_int4()
    # test_int4_to_fp8()
    # test_int4_to_fp16()
    # test_fp8_to_fp16()
    # test_round_trip_int4()

    print("\n" + "=" * 60)
    print("To compile and use:")
    print("  1. Create a setup.py or use torch.utils.cpp_extension.load()")
    print("  2. Compile conversion_kernels.cu")
    print("  3. Import and call torch.ops.conversion_ops.* functions")
    print("=" * 60)
