import numpy as np
import matplotlib.pyplot as plt

# Load the expert batch counts data
data = np.load('all_layers_counts.npy')

print(f'Data shape: {data.shape}')
print(f'Expected: (passes, layers, decoding_steps, experts)')

# Assuming shape is (passes, layers, decoding_steps, experts)
num_experts = data.shape[3]

# Collect data for each expert across all passes, layers, and decoding steps
expert_data_lists = []

for expert_idx in range(num_experts):
    # Extract data for this expert: all passes, all layers, all decoding steps
    expert_data = data[:, :, :, expert_idx]
    # Flatten into a single list
    expert_list = expert_data.flatten()
    expert_data_lists.append(expert_list)
    print(f'Expert {expert_idx}: {len(expert_list)} data points')

# Create CDF plot
plt.figure(figsize=(6, 4))

# Define colors for the 8 experts
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Plot CDF for each expert
for expert_idx, expert_data in enumerate(expert_data_lists):
    # Sort the data and get unique values with their counts
    sorted_data = np.sort(expert_data)

    # Get unique values and their counts for smoother CDF
    unique_vals, counts = np.unique(sorted_data, return_counts=True)

    # Calculate cumulative counts
    cumulative_counts = np.cumsum(counts)

    # Calculate CDF values (normalize to [0, 1])
    cdf_values = cumulative_counts / len(sorted_data)

    # Add a starting point at (min_value, 0) for better visualization
    x_vals = np.concatenate([[unique_vals[0]], unique_vals])
    y_vals = np.concatenate([[0], cdf_values])

    # Plot with markers at each data point
    plt.plot(x_vals, y_vals,
             label=f'Expert {expert_idx}',
             color=colors[expert_idx],
             linewidth=2,
            #  alpha=0.85,
            #  marker='o',
            #  markersize=2,
             markerfacecolor=colors[expert_idx],
             markeredgewidth=0)

# Formatting
plt.xlabel('#tokens', fontsize=11, )
plt.ylabel('CDF', fontsize=11, )
plt.title('CDF of Token Distribution Across Experts\nMixtral-8x7B (Batch Size=64, Seq Len=2048)',
          fontsize=12)
plt.legend(loc='best', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save the plot
plt.savefig('expert_cdf_plot.png', dpi=300, bbox_inches='tight')
print('\nCDF plot saved as: expert_cdf_plot.png')

# Display the plot
plt.savefig('expert_cdf_plot.png', dpi=300, bbox_inches='tight')