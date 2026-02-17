import matplotlib.pyplot as plt
import numpy as np

# Define raw data dictionaries for two devices
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
    ],
    "H100": [
        (128, 0.327, "fp16", False, 0.7734),
        (128, 0.327, "int8", False, 0.3867),
        (128, 0.337, "int8", True,  0.3867),
        (128, 0.305, "int4", True,  0.1934),

        (16,  0.308, "fp16", False, 0.7734),
        (16,  0.306, "int8", False, 0.3867),
        (16,  0.268, "int8", True,  0.3867),
        (16,  0.201, "int4", True,  0.1934),

        (1,   0.301, "fp16", False, 0.7734),
        (1,   0.301, "int8", False, 0.3867),
        (1,   0.234, "int8", True,  0.3867),
        (1,   0.129, "int4", True,  0.1934),

        (256, 0.371, "fp16", False, 0.7734),
        (256, 0.353, "int8", False, 0.3867),
        (256, 0.532, "int8", True,  0.3867),
        (256, 0.469, "int4", True,  0.1934),

        (32,  0.310, "fp16", False, 0.7734),
        (32,  0.312, "int8", False, 0.3867),
        (32,  0.275, "int8", True,  0.3867),
        (32,  0.209, "int4", True,  0.1934),

        (512, 0.644, "fp16", False, 0.7734),
        (512, 0.615, "int8", False, 0.3867),
        (512, 0.899, "int8", True,  0.3867),
        (512, 0.824, "int4", True,  0.1934),

        (64,  0.316, "fp16", False, 0.7734),
        (64,  0.310, "int8", False, 0.3867),
        (64,  0.290, "int8", True,  0.3867),
        (64,  0.231, "int4", True,  0.1934),

        (8,   0.306, "fp16", False, 0.7734),
        (8,   0.306, "int8", False, 0.3867),
        (8,   0.266, "int8", True,  0.3867),
        (8,   0.199, "int4", True,  0.1934),
    ]
}

colors = {
    "a16w16": "#4A90E2",
    "a16w8_full": "#FF9999",
    "a16w8_tile": "#FF5555",
    "a16w4_tile": "#CC0000"
}

def extract_series(raw_data, weight_type, tiling):
    batch_sizes = sorted({b for (b,_,_,_,_) in raw_data})
    values = []
    for b in batch_sizes:
        values.append(
            next(r for (bb,r,w,t,_) in raw_data if bb==b and w==weight_type and t==tiling)
        )
    return np.array(batch_sizes), np.array(values)

fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

for ax, (device, raw_data) in zip(axes, data_devices.items()):
    batch_sizes = sorted({b for (b,_,_,_,_) in raw_data})
    batch_sizes_arr = np.arange(len(batch_sizes))
    width = 0.2

    a16w16 = extract_series(raw_data, "fp16", False)[1]
    a16w8_full = extract_series(raw_data, "int8", False)[1]
    a16w8_tile = extract_series(raw_data, "int8", True)[1]
    a16w4_tile = extract_series(raw_data, "int4", True)[1]

    bars1 = ax.bar(batch_sizes_arr - 1.5*width, a16w16, width,
                label="a16w16 FP16 size(weight)=0.773 GB",
                color=colors["a16w16"])

    bars2 = ax.bar(batch_sizes_arr - 0.5*width, a16w8_full, width,
                label="a16w8 full dequant size(weight)=0.387 GB",
                color=colors["a16w8_full"])

    bars3 = ax.bar(batch_sizes_arr + 0.5*width, a16w8_tile, width,
                label="a16w8 tile dequant size(weight)=0.387 GB",
                color=colors["a16w8_tile"])

    bars4 = ax.bar(batch_sizes_arr + 1.5*width, a16w4_tile, width,
                label="a16w4 tile dequant size(weight)=0.193 GB",
                color=colors["a16w4_tile"])

    ax.set_title(f"Device: {device}", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    interesting_batches = [1, 8, 16, 32, 64]
    for b in interesting_batches:
        if b in batch_sizes:
            idx = batch_sizes.index(b)
            base = a16w16[idx]
            q4 = a16w4_tile[idx]
            if q4 < base:
                speedup = base / q4
                ax.text(idx + 1.9*width, q4 + 0.02, f"x{speedup:.1f}",
                        ha="center", color="maroon", fontsize=9.5)

axes[-1].set_xticks(batch_sizes_arr)
axes[-1].set_xticklabels(batch_sizes)
axes[-1].set_xlabel("Batch Size")
axes[0].set_ylabel("Runtime (ms)", fontsize=11)
axes[1].set_ylabel("Runtime (ms)", fontsize=11)

# ------- SHARED LEGEND -------
# Use the handles from the last axes
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.38, 0.843),
           ncol=1, fontsize=11)

# ------- TITLE WITH PROPER SPACING -------
fig.suptitle("Mixtral-8x22B Per-Expert Runtime Comparison\n"
             "hidden=6144, intermediate=22528", fontsize=12, y=0.95)

plt.tight_layout(rect=[0, 0, 0.99, 0.99])
plt.savefig("mixtral_8x22b_a100_h100.png", dpi=300)
