import matplotlib.pyplot as plt
import numpy as np

# Define raw data dictionaries for two devices
data_devices = {
    "A100": [
        (128,0.611,"fp16",False,0.7734),(128,0.609,"int8",False,0.3867),(128,0.706,"int8",True,0.3867),(128,0.679,"int4",True,0.1934),
        (16,0.596,"fp16",False,0.7734),(16,0.568,"int8",False,0.3867),(16,0.499,"int8",True,0.3867),(16,0.386,"int4",True,0.1934),
        (1,0.602,"fp16",False,0.7734),(1,0.554,"int8",False,0.3867),(1,0.457,"int8",True,0.3867),(1,0.327,"int4",True,0.1934),
        (256,0.759,"fp16",False,0.7734),(256,0.773,"int8",False,0.3867),(256,1.122,"int8",True,0.3867),(256,1.149,"int4",True,0.1934),
        (32,0.614,"fp16",False,0.7734),(32,0.594,"int8",False,0.3867),(32,0.511,"int8",True,0.3867),(32,0.401,"int4",True,0.1934),
        (512,1.297,"fp16",False,0.7734),(512,1.286,"int8",False,0.3867),(512,2.016,"int8",True,0.3867),(512,2.064,"int4",True,0.1934),
        (64,0.598,"fp16",False,0.7734),(64,0.574,"int8",False,0.3867),(64,0.593,"int8",True,0.3867),(64,0.514,"int4",True,0.1934),
        (8,0.570,"fp16",False,0.7734),(8,0.556,"int8",False,0.3867),(8,0.490,"int8",True,0.3867),(8,0.353,"int4",True,0.1934),
    ],
    "H100": [
        (128,0.384,"fp16",False,0.7734),(128,0.383,"int8",False,0.3867),(128,0.428,"int8",True,0.3867),(128,0.391,"int4",True,0.1934),
        (16,0.388,"fp16",False,0.7734),(16,0.360,"int8",False,0.3867),(16,0.334,"int8",True,0.3867),(16,0.272,"int4",True,0.1934),
        (1,0.355,"fp16",False,0.7734),(1,0.361,"int8",False,0.3867),(1,0.285,"int8",True,0.3867),(1,0.190,"int4",True,0.1934),
        (256,0.436,"fp16",False,0.7734),(256,0.432,"int8",False,0.3867),(256,0.581,"int8",True,0.3867),(256,0.564,"int4",True,0.1934),
        (32,0.378,"fp16",False,0.7734),(32,0.360,"int8",False,0.3867),(32,0.349,"int8",True,0.3867),(32,0.275,"int4",True,0.1934),
        (512,0.717,"fp16",False,0.7734),(512,0.685,"int8",False,0.3867),(512,0.985,"int8",True,0.3867),(512,0.910,"int4",True,0.1934),
        (64,0.385,"fp16",False,0.7734),(64,0.385,"int8",False,0.3867),(64,0.371,"int8",True,0.3867),(64,0.304,"int4",True,0.1934),
        (8,0.357,"fp16",False,0.7734),(8,0.369,"int8",False,0.3867),(8,0.336,"int8",True,0.3867),(8,0.267,"int4",True,0.1934),
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
                        ha="center", color="maroon", fontsize=9)

axes[-1].set_xticks(batch_sizes_arr)
axes[-1].set_xticklabels(batch_sizes)
axes[-1].set_xlabel("Batch Size")
axes[0].set_ylabel("Runtime (ms)")
axes[1].set_ylabel("Runtime (ms)")

# ------- SHARED LEGEND -------
# Use the handles from the last axes
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.33, 0.84),
           ncol=1, fontsize=9)

# ------- TITLE WITH PROPER SPACING -------
fig.suptitle("Mixtral-8x22B Per-Expert Runtime Comparison\n"
             "hidden=6144, intermediate=22528", fontsize=12, y=0.95)

plt.tight_layout(rect=[0, 0, 0.99, 0.99])
plt.savefig("mixtral_8x22b_a100_h100.png", dpi=300)
