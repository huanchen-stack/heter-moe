import copy
import torch
import torch.nn as nn
from torchao.quantization import quantize_
from torchao.prototype.mx_formats.config import MXGemmKernelChoice
from torchao.prototype.mx_formats.inference_workflow import (
    MXFPInferenceConfig,
    NVFP4InferenceConfig,
    NVFP4MMConfig
)
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

def quantized_module(original_module, precision=Precision.MXFP8, fake=False):
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
    elif precision == Precision.NVFP4:
        config = NVFP4InferenceConfig(
            mm_config=NVFP4MMConfig.DYNAMIC,
            use_dynamic_per_tensor_scale=True,
        )
    else:
        raise ValueError(f"Unknown precision: {precision}. Use Precision.MXFP8 or Precision.MXFP4")

    # Quantize a temporary copy of the module
    temp_module = copy.deepcopy(original_module)

    try:
        quantize_(temp_module, config=config)
        # Dequantize the weight back to original dtype
        if fake:
            dequantized_weight = temp_module.weight.to_dtype(original_dtype)
    except Exception as e:
        raise RuntimeError(f"Error! torchao version mismatch? {e}")

    # Create a new module of the same type with dequantized weights
    result_module = copy.deepcopy(original_module)
    if fake:
        result_module.weight = nn.Parameter(dequantized_weight)
    else:
        result_module.weight = nn.Parameter(temp_module.weight)

    # Keep the original bias if it exists
    if hasattr(original_module, 'bias') and original_module.bias is not None:
        result_module.bias = nn.Parameter(original_module.bias.clone())

    return result_module

def dequantize_module(quantized_module):
    if not hasattr(quantized_module.weight, 'qdata'):
        return quantized_module
    
    dequantized_module = copy.deepcopy(quantized_module)
    dequantized_weight = quantized_module.weight.to_dtype(
        quantized_module.weight._orig_dtype
    )
    dequantized_module.weight = nn.Parameter(dequantized_weight)

    if hasattr(quantized_module, 'bias') and quantized_module.bias is not None:
        dequantized_module.bias = nn.Parameter(quantized_module.bias.clone())
    
    return dequantized_module

if __name__ == "__main__":
    print("create_fake_quantized_module() defined")
    print("Supported precisions: mxfp8, mxfp4")
    
    m = nn.Linear(32, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    print("Original weight sample:")
    print(m.weight)
    print("============================================")
    
    fake = False
    for precision in [Precision.MXFP8, Precision.MXFP4, Precision.NVFP4]:
        print(f"Testing precision: {precision}")
        q_m = quantized_module(m, precision=precision, fake=fake)
        print("Quantized weight sample:")
        # print(q_m.weight)
        print("Dequantized weight sample:")
        d_q_m = dequantize_module(q_m)
        print(d_q_m.weight)
        print()