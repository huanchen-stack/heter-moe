import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
data = np.load('all_layers_counts.npy')

selected_pass = 9
selected_layer = 23

# Time ranges
time_ranges = [0, 8, 1000, 8, 2040, 8]

heatmap_data_list = []
step_labels = []

for i in range(0, len(time_ranges), 2):
    start = time_ranges[i]
    n = time_ranges[i+1]
    seg = data[selected_pass, selected_layer, start:start+n, :]
    heatmap_data_list.append(seg)
    step_labels.extend(range(start, start+n))

# Concatenate segments
heatmap_data = np.concatenate(heatmap_data_list, axis=0).T

# Colormap
cm = plt.cm.hot

vmin = 0
vmax = heatmap_data.max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Figure
plt.figure(figsize=(7, 3))

im = plt.imshow(
    heatmap_data,
    aspect='auto',
    cmap=cm,
    norm=norm,
    interpolation='none',           # <− KEY: prevents thin lines
)

# No grid (removes thin white/black lines)
plt.grid(False)

# Boundaries for x-axis ticks
tick_positions = []
tick_labels = []

idx = 0
for i in range(0, len(time_ranges), 2):
    start = time_ranges[i]
    num = time_ranges[i+1]

    # Block start tick
    tick_positions.append(idx)
    tick_labels.append(start)

    # Block end tick
    tick_positions.append(idx + num - 1)
    tick_labels.append(start + num - 1)

    idx += num

plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=10)

plt.yticks(range(8), [f'{i}' for i in range(8)], fontsize=11)

plt.xlabel("Time Step")
plt.ylabel("Expert")
plt.title("Token Distribution Heatmap\nMixtral-8x7B (Batch 64, Seq 2048)", fontsize=12)

plt.colorbar(im, label="#tokens")

plt.tight_layout()
plt.savefig("expert_heatmap.png", dpi=300, bbox_inches='tight')
print("Saved expert_heatmap.png")
