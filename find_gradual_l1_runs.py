#!/usr/bin/env python3
"""
Find L1 regularization runs with gradual circuit emergence.
"""

import pandas as pd
from pathlib import Path
import glob

# Find all convergence CSV files
csv_files = glob.glob("convergence_100/logs/*/convergence_*.csv")
csv_files.extend(glob.glob("convergence_analysis_*/convergence_*.csv"))

print(f"Analyzing {len(csv_files)} runs for L1 regularization with gradual emergence...\n")

baseline_runs = []
l1_runs = []

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        continue

    # Skip if doesn't have the right columns
    if 'total_circuits' not in df.columns:
        continue

    # Skip if no circuits found
    if df['total_circuits'].max() == 0:
        continue

    # Check if it's L1 regularization
    has_l1 = 'l1_lambda' in df.columns and df['l1_lambda'].iloc[0] > 0

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
    stable_periods = sum(1 for c in changes if c <= 2)

    # Gradualness score (lower is better)
    gradualness_score = avg_change + (max_change * 0.3)

    # Look for runs with gradual emergence
    if avg_change < 5 and max_change < 30 and df['total_circuits'].max() > 5:
        # Extract seed from filename
        filename = Path(csv_path).stem
        if '_seed' in filename:
            seed = filename.split('_seed')[1].split('_')[0]
        else:
            seed = "unknown"

        run_info = {
            'path': csv_path,
            'seed': seed,
            'avg_change': avg_change,
            'max_change': max_change,
            'gradualness_score': gradualness_score,
            'checkpoints': len(circuit_counts),
            'max_circuits': df['total_circuits'].max(),
            'stable_periods': stable_periods,
            'l1_lambda': df['l1_lambda'].iloc[0] if 'l1_lambda' in df.columns else 0.0
        }

        if has_l1:
            l1_runs.append(run_info)
        else:
            baseline_runs.append(run_info)

# Sort by gradualness score
baseline_runs.sort(key=lambda x: x['gradualness_score'])
l1_runs.sort(key=lambda x: x['gradualness_score'])

print(f"Found {len(baseline_runs)} baseline candidates")
print(f"Found {len(l1_runs)} L1 regularization candidates\n")

print("="*110)
print("TOP 10 L1 REGULARIZATION RUNS (Most Gradual)")
print("="*110)
print(f"{'Seed':<6} {'L1 Lambda':<11} {'Avg Change':<12} {'Max Change':<12} {'Score':<10} {'Checkpoints':<13} {'Max Circuits':<13}")
print("-"*110)

for run in l1_runs[:10]:
    print(f"{run['seed']:<6} {run['l1_lambda']:<11.4f} {run['avg_change']:<12.2f} {run['max_change']:<12.0f} "
          f"{run['gradualness_score']:<10.2f} {run['checkpoints']:<13} {run['max_circuits']:<13}")
    print(f"  Path: {run['path']}")
    print()

print("\n" + "="*110)
print("TOP 10 BASELINE RUNS (For Comparison)")
print("="*110)
print(f"{'Seed':<6} {'Avg Change':<12} {'Max Change':<12} {'Score':<10} {'Checkpoints':<13} {'Max Circuits':<13}")
print("-"*110)

for run in baseline_runs[:10]:
    print(f"{run['seed']:<6} {run['avg_change']:<12.2f} {run['max_change']:<12.0f} "
          f"{run['gradualness_score']:<10.2f} {run['checkpoints']:<13} {run['max_circuits']:<13}")
    print(f"  Path: {run['path']}")
    print()

# Find matching seeds
if l1_runs and baseline_runs:
    print("\n" + "="*110)
    print("MATCHING SEEDS (Same seed in both baseline and L1)")
    print("="*110)

    baseline_seeds = {r['seed']: r for r in baseline_runs}
    l1_seeds = {r['seed']: r for r in l1_runs}

    matching = []
    for seed in baseline_seeds:
        if seed in l1_seeds:
            b = baseline_seeds[seed]
            l = l1_seeds[seed]
            # Combined score - prefer pairs where both are gradual
            combined_score = b['gradualness_score'] + l['gradualness_score']
            matching.append({
                'seed': seed,
                'baseline': b,
                'l1': l,
                'combined_score': combined_score
            })

    matching.sort(key=lambda x: x['combined_score'])

    print(f"\nFound {len(matching)} matching seed pairs\n")
    print(f"{'Seed':<6} {'Baseline Score':<16} {'L1 Score':<16} {'Combined':<12}")
    print("-"*60)

    for m in matching[:10]:
        print(f"{m['seed']:<6} {m['baseline']['gradualness_score']:<16.2f} {m['l1']['gradualness_score']:<16.2f} {m['combined_score']:<12.2f}")
        print(f"  Baseline: {m['baseline']['max_circuits']} circuits, {m['baseline']['checkpoints']} checkpoints")
        print(f"  L1:       {m['l1']['max_circuits']} circuits, {m['l1']['checkpoints']} checkpoints")
        print()

    if matching:
        best = matching[0]
        print("\n" + "="*110)
        print("RECOMMENDATION FOR COMPARISON ANIMATION")
        print("="*110)
        print(f"\nBest matching seed: {best['seed']}")
        print(f"  Baseline gradualness: {best['baseline']['gradualness_score']:.2f}")
        print(f"  L1 gradualness: {best['l1']['gradualness_score']:.2f}")
        print()
        print("Run these commands to create detailed tracking:")
        print()
        print("# Baseline:")
        print(f"python main_detailed_tracking.py \\")
        print(f"    --seed {best['seed']} \\")
        print(f"    --n-experiments 1 \\")
        print(f"    --size 3 --depth 2 \\")
        print(f"    --target-logic-gates XOR \\")
        print(f"    --track-detailed-circuits \\")
        print(f"    --convergence-frequency 1 \\")
        print(f"    --device cuda:0")
        print()
        print("# L1 Regularization:")
        print(f"python main_detailed_tracking.py \\")
        print(f"    --seed {best['seed']} \\")
        print(f"    --n-experiments 1 \\")
        print(f"    --size 3 --depth 2 \\")
        print(f"    --target-logic-gates XOR \\")
        print(f"    --l1-lambda {best['l1']['l1_lambda']:.4f} \\")
        print(f"    --track-detailed-circuits \\")
        print(f"    --convergence-frequency 1 \\")
        print(f"    --device cuda:0")
