import copy
import torch
import torch.nn as nn
from torchao.quantization import quantize_
from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.inference_workflow import MXFPInferenceConfig
from enum import Enum
import hashlib
import os
from pathlib import Path


class Precision(Enum):
    FP16 = "float16"
    BF16 = "bfloat16"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"

def create_fake_quantized_module(original_module, precision=Precision.MXFP8):
    # TODO: support NVFP4

    original_dtype = original_module.weight.dtype
    assert original_dtype in [torch.float16, torch.bfloat16], \
        f"Original module dtype must be float16 or bfloat16, got {original_dtype}"

    # Create quantization config
    if precision == Precision.MXFP8:
        config = MXFPInferenceConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            gemm_kernel_choice=MXGemmKernelChoice.CUBLAS,
        )
    elif precision == Precision.MXFP4:
        config = MXFPInferenceConfig(
            activation_dtype=torch.float4_e2m1fn_x2,
            weight_dtype=torch.float4_e2m1fn_x2,
            gemm_kernel_choice=MXGemmKernelChoice.CUTLASS,
        )
    else:
        raise ValueError(f"Unknown precision: {precision}. Use Precision.MXFP8 or Precision.MXFP4")

    # Quantize a temporary copy of the module
    temp_module = copy.deepcopy(original_module)

    try:
        quantize_(temp_module, config=config)
        # Dequantize the weight back to original dtype
        dequantized_weight = temp_module.weight.to_dtype(original_dtype)
    except Exception as e:
        raise RuntimeError(f"Error! torchao version mismatch? {e}")

    # Create a new module of the same type with dequantized weights
    result_module = copy.deepcopy(original_module)
    result_module.weight = nn.Parameter(dequantized_weight)

    # Keep the original bias if it exists
    if hasattr(original_module, 'bias') and original_module.bias is not None:
        result_module.bias = nn.Parameter(original_module.bias.clone())

    return result_module

if __name__ == "__main__":
    print("create_fake_quantized_module() defined")
    print("Supported precisions: mxfp8, mxfp4")
    m = nn.Linear(32, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    q_m = create_fake_quantized_module(m, precision=Precision.MXFP8)
    print("Original weight sample:")
    print(m.weight)
    print("Quantized weight sample:")
    print(q_m.weight)