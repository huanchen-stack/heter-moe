import numpy as np

# Load the expert batch counts data
data = np.load('all_layers_counts.npy')

# Print structure information
print('File Structure of all_layers_counts.npy')
print('=' * 50)
print(f'Shape: {data.shape}')
print(f'Dtype: {data.dtype}')
print(f'Number of dimensions: {len(data.shape)}')
print()

# Detailed dimension breakdown
if len(data.shape) >= 4:
    print('4D Tensor Structure:')
    print(f'  Dimension 0 (passes): {data.shape[0]}')
    print(f'  Dimension 1 (layers): {data.shape[1]}')
    print(f'  Dimension 2 (decoding steps): {data.shape[2]}')
    print(f'  Dimension 3 (experts): {data.shape[3]}')
else:
    print('Dimensions:')
    for i, dim in enumerate(data.shape):
        print(f'  Dimension {i}: {dim}')

print()
print('Data Statistics:')
print(f'  Min value: {data.min()}')
print(f'  Max value: {data.max()}')
print(f'  Mean value: {data.mean():.4f}')
print(f'  Total elements: {data.size}')

# Display sample data
print()
print('Sample Data:')
print('=' * 50)
if len(data.shape) >= 4:
    print(f'First pass, first layer, first decoding step (all experts):')
    print(data[0, 0, 0, :])
    print()
    print(f'First pass, first layer, first expert (all decoding steps):')
    print(data[0, 0, :, 0])
else:
    print('First few elements:', data.flat[:20])

# Per-dimension statistics
print()
print('Per-Dimension Analysis:')
print('=' * 50)
if len(data.shape) >= 4:
    # Expert-wise statistics (across all passes, layers, and steps)
    expert_totals = data.sum(axis=(0, 1, 2))
    print(f'Total counts per expert (summed across all passes/layers/steps):')
    print(expert_totals)
    print(f'  Most loaded expert: Expert {expert_totals.argmax()} with {expert_totals.max()} total')
    print(f'  Least loaded expert: Expert {expert_totals.argmin()} with {expert_totals.min()} total')
    print(f'  Load imbalance ratio: {expert_totals.max() / expert_totals.min():.2f}x')
    print()

    # Layer-wise statistics
    layer_totals = data.sum(axis=(0, 2, 3))
    print(f'Total counts per layer:')
    print(layer_totals)
    print()

# Leaf-level analysis (each individual line in the tensor)
print()
print('Leaf-Level Analysis (Sum of Each Line):')
print('=' * 50)

# Flatten and get all individual values (leaves)
all_leaves = data.flatten()
print(f'Total number of leaves: {len(all_leaves)}')
print(f'Non-zero leaves: {np.count_nonzero(all_leaves)}')
print()

# Calculate statistics for individual leaf values
print('Individual Leaf Statistics:')
print(f'  Mean: {np.mean(all_leaves):.4f}')
print(f'  Median: {np.median(all_leaves):.4f}')

# Mode calculation
from scipy import stats
# mode_result = stats.mode(all_leaves, keepdims=True)
# print(f'  Mode: {mode_result.mode[0]} (appears {mode_result.count[0]} times)')

print(f'  Max: {np.max(all_leaves)}')
print(f'  Min: {np.min(all_leaves)}')
print(f'  Standard deviation: {np.std(all_leaves):.4f}')

# If the data is 4D, also analyze sums along the expert dimension (sum each line of experts)
if len(data.shape) >= 4:
    print()
    print('Sum Across Experts (for each pass/layer/step combination):')
    print('=' * 50)

    # Sum across the expert dimension (axis 3)
    expert_sums = data.sum(axis=3)  # Shape: (passes, layers, steps)
    expert_sums_flat = expert_sums.flatten()

    print(f'Total number of (pass, layer, step) combinations: {len(expert_sums_flat)}')
    print()
    print('Statistics for expert sums per decoding step:')
    print(f'  Mean: {np.mean(expert_sums_flat):.4f}')
    print(f'  Median: {np.median(expert_sums_flat):.4f}')

    # mode_result_sums = stats.mode(expert_sums_flat, keepdims=True)
    # print(f'  Mode: {mode_result_sums.mode[0]} (appears {mode_result_sums.count[0]} times)')

    print(f'  Max: {np.max(expert_sums_flat)}')
    print(f'  Min: {np.min(expert_sums_flat)}')
    print(f'  Standard deviation: {np.std(expert_sums_flat):.4f}')
