#!/usr/bin/env python3
"""
Check the actual network structure in the real tracking data.
"""

import json

json_path = "detailed_circuit_tracking/logs/run_20251205-173605/detailed_circuits_k3_seed0_depth2_lr0.001_loss0.01.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# Check first epoch
epoch_data = data[0]
print(f"Epoch: {epoch_data['epoch']}")
print(f"Total circuits: {epoch_data['total_circuits']}")
print()

# Check first circuit
if epoch_data['outputs'] and epoch_data['outputs'][0]['circuits']:
    circuit = epoch_data['outputs'][0]['circuits'][0]

    node_masks = circuit['node_masks']
    edge_masks = circuit['edge_masks']

    print(f"Node masks count: {len(node_masks)}")
    print(f"Node masks shapes: {[len(mask) for mask in node_masks]}")
    print()

    print(f"Edge masks count: {len(edge_masks)}")
    print(f"Edge masks shapes: {[f'{len(mask)}x{len(mask[0]) if mask else 0}' for mask in edge_masks]}")
    print()

    print("Expected edge mask dimensions:")
    for i in range(len(edge_masks)):
        from_layer_size = len(node_masks[i])
        to_layer_size = len(node_masks[i+1])
        actual_shape = f"{len(edge_masks[i])}x{len(edge_masks[i][0]) if edge_masks[i] else 0}"
        expected_shape = f"{to_layer_size}x{from_layer_size}"

        match = "✓" if actual_shape == expected_shape else "❌"
        print(f"  Edge {i}: {actual_shape} (expected {expected_shape}) {match}")
    print()

    # Check later epoch for comparison
    if len(data) > 20:
        epoch_data = data[20]
        print(f"Checking Epoch {epoch_data['epoch']} (index 20)...")
        print(f"Total circuits: {epoch_data['total_circuits']}")

        if epoch_data['outputs'] and epoch_data['outputs'][0]['circuits']:
            circuit = epoch_data['outputs'][0]['circuits'][0]

            node_masks = circuit['node_masks']
            edge_masks = circuit['edge_masks']

            print(f"Node masks shapes: {[len(mask) for mask in node_masks]}")
            print(f"Edge masks shapes: {[f'{len(mask)}x{len(mask[0]) if mask else 0}' for mask in edge_masks]}")
            print()

            print("Expected edge mask dimensions:")
            for i in range(len(edge_masks)):
                from_layer_size = len(node_masks[i])
                to_layer_size = len(node_masks[i+1])
                actual_shape = f"{len(edge_masks[i])}x{len(edge_masks[i][0]) if edge_masks[i] else 0}"
                expected_shape = f"{to_layer_size}x{from_layer_size}"

                match = "✓" if actual_shape == expected_shape else "❌"
                print(f"  Edge {i}: {actual_shape} (expected {expected_shape}) {match}")
