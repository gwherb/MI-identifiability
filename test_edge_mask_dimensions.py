#!/usr/bin/env python3
"""
Test script to check edge mask dimensions in circuit discovery.
"""

import torch
import numpy as np
from mi_identifiability.neural_model import MLP
from mi_identifiability.circuit import enumerate_all_valid_circuit, find_circuits

# Create a small test network
layer_sizes = [2, 3, 3, 1]
model = MLP(layer_sizes, device='cpu')

print("Network structure:", layer_sizes)
print()

# Enumerate a few circuits to check their edge mask dimensions
circuits = enumerate_all_valid_circuit(model, min_sparsity=0.0, use_tqdm=False)

print(f"Total circuits enumerated: {len(circuits)}")
print()

# Check the first 5 circuits
for i, circuit in enumerate(circuits[:5]):
    print(f"Circuit {i}:")
    print(f"  Node masks shapes: {[mask.shape for mask in circuit.node_masks]}")
    print(f"  Edge masks shapes: {[mask.shape for mask in circuit.edge_masks]}")

    # Check if edge masks match expected dimensions
    expected_edge_shapes = [
        (layer_sizes[1], layer_sizes[0]),  # Layer 0 -> 1: (3, 2)
        (layer_sizes[2], layer_sizes[1]),  # Layer 1 -> 2: (3, 3)
        (layer_sizes[3], layer_sizes[2])   # Layer 2 -> 3: (1, 3)
    ]

    print(f"  Expected edge shapes: {expected_edge_shapes}")

    # Validate dimensions
    for layer_idx, (actual_shape, expected_shape) in enumerate(zip([m.shape for m in circuit.edge_masks], expected_edge_shapes)):
        if actual_shape != expected_shape:
            print(f"  ⚠️  MISMATCH at layer {layer_idx}: got {actual_shape}, expected {expected_shape}")
        else:
            print(f"  ✓ Layer {layer_idx} edge mask dimensions correct")
    print()

print("\nNow testing with find_circuits (which returns perfect circuits)...")
print()

# Create simple XOR data
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Train the model briefly
model.train_model(x_train, y_train, x_train, y_train, epochs=100, lr=0.01, verbose=False)

# Find perfect circuits
circuits, sparsities, _ = find_circuits(
    model,
    x_train,
    y_train,
    accuracy_threshold=0.99,
    min_sparsity=0.0,
    use_gpu_batching=False,
    use_tqdm=False
)

print(f"Found {len(circuits)} perfect circuits")
print()

# Check dimensions of found circuits
for i, circuit in enumerate(circuits[:3]):
    print(f"Perfect Circuit {i}:")
    print(f"  Node masks shapes: {[mask.shape for mask in circuit.node_masks]}")
    print(f"  Edge masks shapes: {[mask.shape for mask in circuit.edge_masks]}")

    expected_edge_shapes = [
        (layer_sizes[1], layer_sizes[0]),
        (layer_sizes[2], layer_sizes[1]),
        (layer_sizes[3], layer_sizes[2])
    ]

    # Validate dimensions
    all_correct = True
    for layer_idx, (actual_shape, expected_shape) in enumerate(zip([m.shape for m in circuit.edge_masks], expected_edge_shapes)):
        if actual_shape != expected_shape:
            print(f"  ⚠️  MISMATCH at layer {layer_idx}: got {actual_shape}, expected {expected_shape}")
            all_correct = False

    if all_correct:
        print(f"  ✓ All edge mask dimensions correct")
    print()

print("\nTest complete!")
