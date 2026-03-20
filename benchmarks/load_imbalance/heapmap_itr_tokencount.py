import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

data = np.load('all_layers_counts.npy')

selected_pass = 9
selected_layer = 23

iterations = [7, 1007, 2046]

print("Token counts per expert:")
print("=" * 50)

for it in iterations:
    counts = data[selected_pass, selected_layer, it, :]
    print(f"\nIteration {it}:")
    print(f"  Expert tokens: {counts}")
    print(f"  Total: {counts.sum()}")

# Build heatmap: each column is one iteration, each row is one expert
heatmap_data = np.stack([data[selected_pass, selected_layer, it, :] for it in iterations], axis=1)

cm = plt.cm.hot
norm = mcolors.Normalize(vmin=0, vmax=heatmap_data.max())

plt.figure(figsize=(5, 3))
im = plt.imshow(heatmap_data, aspect='auto', cmap=cm, norm=norm, interpolation='none')
plt.grid(False)

plt.xticks(range(len(iterations)), [str(it) for it in iterations], fontsize=10)
plt.yticks(range(heatmap_data.shape[0]), [str(i) for i in range(heatmap_data.shape[0])], fontsize=11)

plt.xlabel("Iteration")
plt.ylabel("Expert")
plt.title("Token Distribution per Iteration\nMixtral-8x7B (Batch 64, Seq 2048)", fontsize=12)
plt.colorbar(im, label="#tokens")
plt.tight_layout()

os.makedirs("plot", exist_ok=True)
out_path = "plot/heatmap_itr_tokencount.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nSaved {out_path}")