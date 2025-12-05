#!/usr/bin/env python3
"""
Analyze how gradually circuits emerge in detailed tracking runs.
"""

import json
from pathlib import Path

runs = [
    "detailed_circuit_tracking/logs/run_20251205-173536/detailed_circuits_k3_seed0_depth2_lr0.001_loss0.01.json",
    "detailed_circuit_tracking/logs/run_20251205-173605/detailed_circuits_k3_seed0_depth2_lr0.001_loss0.01.json"
]

for run_path in runs:
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(run_path).parent.name}")
    print(f"{'='*80}\n")

    with open(run_path, 'r') as f:
        data = json.load(f)

    # Extract circuit counts over time
    epochs = []
    circuit_counts = []

    for checkpoint in data:
        epoch = checkpoint['epoch']
        total_circuits = checkpoint['total_circuits']
        epochs.append(epoch)
        circuit_counts.append(total_circuits)

    print(f"Total checkpoints: {len(epochs)}")
    print(f"Epoch range: {epochs[0]} to {epochs[-1]}")
    print()

    # Show progression
    print("Circuit emergence progression:")
    print("Epoch  | Circuits | Change")
    print("-------|----------|-------")

    for i, (epoch, count) in enumerate(zip(epochs, circuit_counts)):
        change = count - circuit_counts[i-1] if i > 0 else 0
        change_str = f"+{change}" if change > 0 else str(change)
        print(f"{epoch:6d} | {count:8d} | {change_str:>6}")

    # Calculate jumpiness metric (average absolute change)
    if len(circuit_counts) > 1:
        changes = [abs(circuit_counts[i] - circuit_counts[i-1]) for i in range(1, len(circuit_counts))]
        avg_change = sum(changes) / len(changes)
        max_change = max(changes)

        print()
        print(f"Average absolute change per checkpoint: {avg_change:.2f}")
        print(f"Maximum change in single checkpoint: {max_change}")

        # Gradualness score (lower is more gradual)
        # Penalize large jumps
        gradualness_score = avg_change + (max_change * 0.5)
        print(f"Gradualness score (lower=better): {gradualness_score:.2f}")

        # Check for periods of stability
        stable_periods = sum(1 for c in changes if c == 0)
        print(f"Stable checkpoints (no change): {stable_periods}")
