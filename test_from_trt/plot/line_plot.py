import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Define raw data for A100 only
data_devices = {
    "A100": [
        (128, 0.448, "fp16", False, 0.7734),
        (128, 0.439, "int8", False, 0.3867),
        (128, 0.646, "int8", True,  0.3867),
        (128, 0.661, "int4", True,  0.1934),

        (16,  0.419, "fp16", False, 0.7734),
        (16,  0.418, "int8", False, 0.3867),
        (16,  0.347, "int8", True,  0.3867),
        (16,  0.217, "int4", True,  0.1934),

        (1,   0.409, "fp16", False, 0.7734),
        (1,   0.402, "int8", False, 0.3867),
        (1,   0.314, "int8", True,  0.3867),
        (1,   0.178, "int4", True,  0.1934),

        (256, 0.599, "fp16", False, 0.7734),
        (256, 0.597, "int8", False, 0.3867),
        (256, 1.198, "int8", True,  0.3867),
        (256, 1.256, "int4", True,  0.1934),

        (32,  0.424, "fp16", False, 0.7734),
        (32,  0.421, "int8", False, 0.3867),
        (32,  0.366, "int8", True,  0.3867),
        (32,  0.237, "int4", True,  0.1934),

        (512, 1.109, "fp16", False, 0.7734),
        (512, 1.103, "int8", False, 0.3867),
        (512, 2.305, "int8", True,  0.3867),
        (512, 2.414, "int4", True,  0.1934),

        (64,  0.421, "fp16", False, 0.7734),
        (64,  0.422, "int8", False, 0.3867),
        (64,  0.415, "int8", True,  0.3867),
        (64,  0.359, "int4", True,  0.1934),

        (8,   0.415, "fp16", False, 0.7734),
        (8,   0.416, "int8", False, 0.3867),
        (8,   0.330, "int8", True,  0.3867),
        (8,   0.196, "int4", True,  0.1934),
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

raw_data = data_devices["A100"]
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

ax.set_title("Mixtral-8x22B Per-Expert Runtime on A100\nhidden=6144, intermediate=22528", 
             fontsize=12)
ax.grid(True, linestyle="--", alpha=0.4, which='both')
ax.set_ylabel("Runtime (ms)", fontsize=11)
ax.set_xlabel("Batch Size", fontsize=11)
ax.set_xticks(batch_range)
ax.set_xticklabels(batch_range)
ax.legend(fontsize=10)

plt.tight_layout()
# plt.savefig("mixtral_8x22b_a100_lineplot.pdf", bbox_inches='tight')
plt.savefig("mixtral_8x22b_a100_lineplot.png", dpi=300, bbox_inches='tight')