import matplotlib.pyplot as plt
import numpy as np

# mixtral-8x22B expert runtime data
# hidden=6144, intermediate=22528
# Columns: (batch_size, runtime_ms, weight_type, tiling_dequant, weights_size_GB)
# Device: A100
device = "A100"
raw_data = [
    (128,0.611,"fp16",False,0.7734),(128,0.609,"int8",False,0.3867),(128,0.706,"int8",True,0.3867),(128,0.679,"int4",True,0.1934),
    (16,0.596,"fp16",False,0.7734),(16,0.568,"int8",False,0.3867),(16,0.499,"int8",True,0.3867),(16,0.386,"int4",True,0.1934),
    (1,0.602,"fp16",False,0.7734),(1,0.554,"int8",False,0.3867),(1,0.457,"int8",True,0.3867),(1,0.327,"int4",True,0.1934),
    (256,0.759,"fp16",False,0.7734),(256,0.773,"int8",False,0.3867),(256,1.122,"int8",True,0.3867),(256,1.149,"int4",True,0.1934),
    (32,0.614,"fp16",False,0.7734),(32,0.594,"int8",False,0.3867),(32,0.511,"int8",True,0.3867),(32,0.401,"int4",True,0.1934),
    (512,1.297,"fp16",False,0.7734),(512,1.286,"int8",False,0.3867),(512,2.016,"int8",True,0.3867),(512,2.064,"int4",True,0.1934),
    (64,0.598,"fp16",False,0.7734),(64,0.574,"int8",False,0.3867),(64,0.593,"int8",True,0.3867),(64,0.514,"int4",True,0.1934),
    (8,0.570,"fp16",False,0.7734),(8,0.556,"int8",False,0.3867),(8,0.490,"int8",True,0.3867),(8,0.353,"int4",True,0.1934),
]
# mixtral-8x22B expert runtime data
# hidden=6144, intermediate=22528
# Columns: (batch_size, runtime_ms, weight_type, tiling_dequant, weights_size_GB)
# Device: H100
device = "H100"
raw_data = [
    (128,0.384,"fp16",False,0.7734),(128,0.383,"int8",False,0.3867),(128,0.428,"int8",True,0.3867),(128,0.391,"int4",True,0.1934),
    (16,0.388,"fp16",False,0.7734),(16,0.360,"int8",False,0.3867),(16,0.334,"int8",True,0.3867),(16,0.272,"int4",True,0.1934),
    (1,0.355,"fp16",False,0.7734),(1,0.361,"int8",False,0.3867),(1,0.285,"int8",True,0.3867),(1,0.190,"int4",True,0.1934),
    (256,0.436,"fp16",False,0.7734),(256,0.432,"int8",False,0.3867),(256,0.581,"int8",True,0.3867),(256,0.564,"int4",True,0.1934),
    (32,0.378,"fp16",False,0.7734),(32,0.360,"int8",False,0.3867),(32,0.349,"int8",True,0.3867),(32,0.275,"int4",True,0.1934),
    (512,0.717,"fp16",False,0.7734),(512,0.685,"int8",False,0.3867),(512,0.985,"int8",True,0.3867),(512,0.910,"int4",True,0.1934),
    (64,0.385,"fp16",False,0.7734),(64,0.385,"int8",False,0.3867),(64,0.371,"int8",True,0.3867),(64,0.304,"int4",True,0.1934),
    (8,0.357,"fp16",False,0.7734),(8,0.369,"int8",False,0.3867),(8,0.336,"int8",True,0.3867),(8,0.267,"int4",True,0.1934),
]

# # mixtral-8x22B expert runtime data
# # hidden=6144, intermediate=22528
# # Columns: (batch_size, runtime_ms, weight_type, tiling_dequant, weights_size_GB)
# # Device: H200
# raw_data = [
#     (128,0.277,"fp16",False,0.7734),(128,0.218,"int8",False,0.3867),(128,0.259,"int8",True,0.3867),(128,0.260,"int4",True,0.1934),
#     (16,0.223,"fp16",False,0.7734),(16,0.234,"int8",False,0.3867),(16,0.240,"int8",True,0.3867),(16,0.224,"int4",True,0.1934),
#     (1,0.206,"fp16",False,0.7734),(1,0.198,"int8",False,0.3867),(1,0.168,"int8",True,0.3867),(1,0.132,"int4",True,0.1934),
#     (256,0.331,"fp16",False,0.7734),(256,0.322,"int8",False,0.3867),(256,0.415,"int8",True,0.3867),(256,0.376,"int4",True,0.1934),
#     (32,0.275,"fp16",False,0.7734),(32,0.242,"int8",False,0.3867),(32,0.241,"int8",True,0.3867),(32,0.193,"int4",True,0.1934),
#     (512,0.530,"fp16",False,0.7734),(512,0.546,"int8",False,0.3867),(512,0.736,"int8",True,0.3867),(512,0.850,"int4",True,0.1934),
#     (64,0.249,"fp16",False,0.7734),(64,0.246,"int8",False,0.3867),(64,0.223,"int8",True,0.3867),(64,0.256,"int4",True,0.1934),
#     (8,0.236,"fp16",False,0.7734),(8,0.244,"int8",False,0.3867),(8,0.235,"int8",True,0.3867),(8,0.214,"int4",True,0.1934),
# ]



# # mixtral-8x7B expert runtime data
# # hidden=4096, intermediate=14336
# raw_data = [
#     (128,0.394,"fp16",False,0.3281),(128,0.362,"int8",False,0.1641),(128,0.435,"int8",True,0.1641),(128,0.428,"int4",True,0.0820),
#     (16,0.338,"fp16",False,0.3281),(16,0.334,"int8",False,0.1641),(16,0.330,"int8",True,0.1641),(16,0.279,"int4",True,0.0820),
#     (1,0.351,"fp16",False,0.3281),(1,0.339,"int8",False,0.1641),(1,0.325,"int8",True,0.1641),(1,0.259,"int4",True,0.0820),
#     (256,0.483,"fp16",False,0.3281),(256,0.519,"int8",False,0.1641),(256,0.589,"int8",True,0.1641),(256,0.596,"int4",True,0.0820),
#     (32,0.345,"fp16",False,0.3281),(32,0.336,"int8",False,0.1641),(32,0.356,"int8",True,0.1641),(32,0.313,"int4",True,0.0820),
#     (512,0.722,"fp16",False,0.3281),(512,0.753,"int8",False,0.1641),(512,1.000,"int8",True,0.1641),(512,1.006,"int4",True,0.0820),
#     (64,0.357,"fp16",False,0.3281),(64,0.350,"int8",False,0.1641),(64,0.381,"int8",True,0.1641),(64,0.372,"int4",True,0.0820),
#     (8,0.344,"fp16",False,0.3281),(8,0.339,"int8",False,0.1641),(8,0.323,"int8",True,0.1641),(8,0.267,"int4",True,0.0820),
# ]

batch_sizes = sorted({b for (b,_,_,_,_) in raw_data})

# Prepare series
def extract_series(weight_type, tiling):
    vals=[]
    for b in batch_sizes:
        vals.append(next(r for (bb,r,w,t,_) in raw_data if bb==b and w==weight_type and t==tiling))
    return np.array(vals)

a16w16 = extract_series("fp16", False)
a16w8_full = extract_series("int8", False)
a16w8_tile = extract_series("int8", True)
a16w4_tile = extract_series("int4", True)

# Colors (blue for fp16, red variants for quantized)
colors = {
    "a16w16": "#4A90E2",
    "a16w8_full": "#FF9999",
    "a16w8_tile": "#FF5555",
    "a16w4_tile": "#CC0000"
}

# Plot
x = np.arange(len(batch_sizes))
width = 0.2

plt.figure(figsize=(8,4))

plt.bar(x - 1.5*width, a16w16, width, label="a16w16 FP16 size(weight)=0.773 GB", color=colors["a16w16"])
plt.bar(x - 0.5*width, a16w8_full, width, label="a16w8 full dequant size(weight)=0.387 GB", color=colors["a16w8_full"])
plt.bar(x + 0.5*width, a16w8_tile, width, label="a16w8 tile dequant size(weight)=0.387 GB", color=colors["a16w8_tile"])
plt.bar(x + 1.5*width, a16w4_tile, width, label="a16w4 tile dequant size(weight)=0.193 GB", color=colors["a16w4_tile"])

plt.xticks(x, batch_sizes)
plt.xlabel("Batch Size")
plt.ylabel("Runtime (ms)")
plt.title(f"Mixtral-8x22B Per Expert Runtime\nhidden=6144, intermediate=22528   Device: {device}")
# plt.title("Mixtral-8x7B Expert Runtime vs Batch Size\nhidden=4096, intermediate=14336")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()

# Speedup annotations for batch sizes where speedup > 1
interesting_batches = [1,8,16,32,64]
for b in interesting_batches:
    idx = batch_sizes.index(b)
    base = a16w16[idx]
    q4 = a16w4_tile[idx]
    if q4 < base:  # speedup only if faster
        speedup = base / q4
        plt.text(idx + 1.9*width, q4 + 0.02, f"x{speedup:.1f}", ha="center", color="maroon", fontsize=9)

plt.tight_layout()
plt.savefig(f"mixtral_8x22b_{device.lower()}.png")