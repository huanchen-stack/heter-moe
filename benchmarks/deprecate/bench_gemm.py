import torch

HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS = 0, 0, 0
LOAD_IMBALANCE = []
BATCH_SIZE = 1

def quantize(tensor_in, scale, zero_point, dtype_in, dtype_out, tensor_out):
    pass

def dequantize(tensor_in, scale, zero_point, dtype_in, dtype_out, tensor_out):
    pass

class GGEMMFunctions:
    """data class for GGEMM functions"""

def run_expert(x, BC, D, up_buff, down_buff):
    dequantize()
    pass

def bench_moe_w16a16():
    pass

def bench_moe_w8a16():
    pass

def bench_moe_w4a16():
    pass

def bench_moe_w8a8():
    pass

def bench_moe_w4a8():
    pass

def bench_moe(
    weight_bitwidth,
    activation_bitwidth,
    batch_size, 
    load_imbalance,
):
    pass

