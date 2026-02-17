import numpy as np

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