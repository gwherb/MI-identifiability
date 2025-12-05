#!/usr/bin/env python3
"""
Find runs with gradual circuit emergence for better animations.
"""

import pandas as pd
from pathlib import Path
import glob

# Find all convergence CSV files
csv_files = glob.glob("convergence_100/logs/*/convergence_*.csv")

print(f"Analyzing {len(csv_files)} runs for gradual circuit emergence...\n")

gradual_runs = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    # Skip if no circuits found
    if df['total_circuits'].max() == 0:
        continue

    # Calculate gradualness metrics
    circuit_counts = df['total_circuits'].values

    # Skip if only one checkpoint
    if len(circuit_counts) < 2:
        continue

    changes = [abs(circuit_counts[i] - circuit_counts[i-1]) for i in range(1, len(circuit_counts))]

    if not changes:
        continue

    avg_change = sum(changes) / len(changes)
    max_change = max(changes)
    stable_periods = sum(1 for c in changes if c <= 2)  # Allow small changes

    # Gradualness score (lower is better)
    gradualness_score = avg_change + (max_change * 0.3)

    # Look for runs with:
    # - Low average change
    # - Not too many large jumps
    # - Some non-zero progression

    if avg_change < 5 and max_change < 30 and df['total_circuits'].max() > 5:
        seed = Path(csv_path).stem.split('_seed')[1].split('_')[0]
        gradual_runs.append({
            'path': csv_path,
            'seed': seed,
            'avg_change': avg_change,
            'max_change': max_change,
            'gradualness_score': gradualness_score,
            'checkpoints': len(circuit_counts),
            'max_circuits': df['total_circuits'].max(),
            'stable_periods': stable_periods
        })

# Sort by gradualness score (lower is better)
gradual_runs.sort(key=lambda x: x['gradualness_score'])

print(f"Found {len(gradual_runs)} candidates with gradual emergence\n")

print("Top 10 most gradual runs:")
print("="*100)
print(f"{'Seed':<6} {'Avg Change':<12} {'Max Change':<12} {'Score':<10} {'Checkpoints':<13} {'Max Circuits':<13}")
print("-"*100)

for run in gradual_runs[:10]:
    print(f"{run['seed']:<6} {run['avg_change']:<12.2f} {run['max_change']:<12.0f} "
          f"{run['gradualness_score']:<10.2f} {run['checkpoints']:<13} {run['max_circuits']:<13}")
    print(f"  Path: {run['path']}")
    print()

if gradual_runs:
    print("\nRecommendation:")
    best = gradual_runs[0]
    print(f"  Best candidate: seed {best['seed']}")
    print(f"  To create detailed tracking for this seed, run:")
    print(f"")
    print(f"  python main_detailed_tracking.py \\")
    print(f"      --seed {best['seed']} \\")
    print(f"      --n-experiments 1 \\")
    print(f"      --size 3 --depth 2 \\")
    print(f"      --target-logic-gates XOR \\")
    print(f"      --track-detailed-circuits \\")
    print(f"      --convergence-frequency 1 \\")
    print(f"      --device cuda:0")
