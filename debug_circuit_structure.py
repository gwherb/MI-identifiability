#!/usr/bin/env python3
"""
Debug circuit structure to understand node_masks and edge_masks indexing.
"""

import torch
import numpy as np
from mi_identifiability.neural_model import MLP
from mi_identifiability.circuit import enumerate_all_valid_circuit

# Create a small test network
layer_sizes = [2, 3, 3, 1]
model = MLP(layer_sizes, device='cpu')

print("Network structure:", layer_sizes)
print(f"Number of layers (including input/output): {len(layer_sizes)}")
print(f"Number of weight layers: {len(layer_sizes) - 1}")
print()

# Get one circuit
circuits = enumerate_all_valid_circuit(model, min_sparsity=0.0, use_tqdm=False)
circuit = circuits[0]

print(f"Circuit node_masks length: {len(circuit.node_masks)}")
print(f"Circuit edge_masks length: {len(circuit.edge_masks)}")
print()

print("Node masks:")
for i, mask in enumerate(circuit.node_masks):
    print(f"  node_masks[{i}]: shape {mask.shape}, values {mask}")

print()
print("Edge masks:")
for i, mask in enumerate(circuit.edge_masks):
    print(f"  edge_masks[{i}]: shape {mask.shape}")
    print(f"    Shape should connect node_masks[{i}] -> node_masks[{i+1}]")
    print(f"    Should be: ({len(circuit.node_masks[i+1])}, {len(circuit.node_masks[i])})")
    print()

print("\nAnalysis:")
print(f"It appears node_masks has {len(circuit.node_masks)} elements")
print(f"For network [2, 3, 3, 1], we should have 4 node_masks (one per layer)")
print()

# Check model structure
print("Model layers:")
print(f"  Input features: {model.layers[0][0].in_features}")
for i, layer in enumerate(model.layers):
    linear = layer[0]
    print(f"  Layer {i}: {linear.in_features} -> {linear.out_features}")
