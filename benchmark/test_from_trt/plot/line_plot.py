import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Define raw data for RTX6000 only
data_devices = {
    "RTX6000": [
        # batch = 1024
        (1024, 0.085, "fp16", False, 0.0088),
        (1024, 0.084, "int8", False, 0.0044),
        (1024, 0.150, "int8", True,  0.0044),
        (1024, 0.132, "int4", True,  0.0022),

        # batch = 128
        (128,  0.036, "fp16", False, 0.0088),
        (128,  0.035, "int8", False, 0.0044),
        (128,  0.056, "int8", True,  0.0044),
        (128,  0.059, "int4", True,  0.0022),

        # batch = 16
        (16,   0.027, "fp16", False, 0.0088),
        (16,   0.026, "int8", False, 0.0044),
        (16,   0.054, "int8", True,  0.0044),
        (16,   0.057, "int4", True,  0.0022),

        # batch = 1
        (1,    0.017, "fp16", False, 0.0088),
        (1,    0.017, "int8", False, 0.0044),
        (1,    0.021, "int8", True,  0.0044),
        (1,    0.027, "int4", True,  0.0022),

        # batch = 256
        (256,  0.047, "fp16", False, 0.0088),
        (256,  0.047, "int8", False, 0.0044),
        (256,  0.068, "int8", True,  0.0044),
        (256,  0.072, "int4", True,  0.0022),

        # batch = 32
        (32,   0.027, "fp16", False, 0.0088),
        (32,   0.027, "int8", False, 0.0044),
        (32,   0.056, "int8", True,  0.0044),
        (32,   0.057, "int4", True,  0.0022),

        # batch = 512
        (512,  0.064, "fp16", False, 0.0088),
        (512,  0.064, "int8", False, 0.0044),
        (512,  0.099, "int8", True,  0.0044),
        (512,  0.097, "int4", True,  0.0022),

        # batch = 64
        (64,   0.033, "fp16", False, 0.0088),
        (64,   0.031, "int8", False, 0.0044),
        (64,   0.055, "int8", True,  0.0044),
        (64,   0.060, "int4", True,  0.0022),

        # batch = 8
        (8,    0.025, "fp16", False, 0.0088),
        (8,    0.025, "int8", False, 0.0044),
        (8,    0.041, "int8", True,  0.0044),
        (8,    0.044, "int4", True,  0.0022),
    ]
}


def extract_series(raw_data, weight_type, tiling, batch_range):
    batch_sizes = sorted([b for (b,_,_,_,_) in raw_data if b in batch_range])
    values = []
    for b in batch_sizes:
        values.append(
            next(r for (bb,r,w,t,_) in raw_data if bb==b and w==weight_type and t==tiling)
        )
    return np.array(batch_sizes), np.array(values)

# Custom formatter to remove trailing zeros
def format_func(value, tick_number):
    return f'{value:g}'

batch_range = [8, 16, 32, 64, 128, 256, 512]

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

raw_data = data_devices["RTX6000"]
bs_fp16, a16w16 = extract_series(raw_data, "fp16", False, batch_range)
bs_int4, a16w4_tile = extract_series(raw_data, "int4", True, batch_range)

ax.plot(bs_fp16, a16w16, marker='o', linewidth=2, markersize=4,
        label="a16w16 (FP16)", color='#4A90E2')

ax.plot(bs_int4, a16w4_tile, marker='s', linewidth=2, markersize=4,
        label="a16w4 (INT4)", color='#CC0000')

# Set both x and y axes to log scale (base 2)
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)

# Use custom formatter to remove trailing zeros
ax.yaxis.set_major_formatter(FuncFormatter(format_func))
ax.yaxis.set_minor_formatter(FuncFormatter(format_func))

# Add speedup annotations with arrows
for i, bs in enumerate(bs_fp16):
    fp16_val = a16w16[i]
    int4_val = a16w4_tile[i]
    
    # Only annotate where INT4 is faster (speedup > 1)
    if fp16_val > int4_val:
        speedup = fp16_val / int4_val
        
        # Use geometric mean for midpoint on log scale
        mid_y = np.sqrt(fp16_val * int4_val)
        
        # # Add arrow pointing down from FP16 to INT4
        # ax.annotate('', xy=(bs, int4_val), xytext=(bs, fp16_val),
        #            arrowprops=dict(arrowstyle='->', color='darkgreen', 
        #                          lw=1.5, alpha=0.7))
        
        # # Add speedup text
        # ax.text(bs, mid_y, f'  ×{speedup:.1f}', 
        #        fontsize=9, color='darkgreen', fontweight='bold',
        #        ha='left', va='center')

ax.set_title("Qwen 8-Expert Runtime on RTX6000\nhidden=6144, intermediate=22528", 
             fontsize=12)
ax.grid(True, linestyle="--", alpha=0.4, which='both')
ax.set_ylabel("Runtime (ms)", fontsize=11)
ax.set_xlabel("Batch Size", fontsize=11)
ax.set_xticks(batch_range)
ax.set_xticklabels(batch_range)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("qwen_30b_a3b_RTX6000_lineplot.png", dpi=300, bbox_inches='tight')