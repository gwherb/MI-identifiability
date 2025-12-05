#!/usr/bin/env python3
"""
Find the runs that best match the mean trajectory for L1 and baseline regularization.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def identify_regularization_type(df_row):
    """Identify regularization type from CSV row."""
    l1 = df_row['l1_lambda']
    l2 = df_row['l2_lambda']
    dropout = df_row['dropout_rate']

    if l1 == 0.0 and l2 == 0.0 and dropout == 0.0:
        return 'baseline'
    elif l1 > 0.0:
        return 'l1'
    elif l2 > 0.0:
        return 'l2'
    elif dropout > 0.0:
        return 'dropout'
    return 'unknown'

def load_convergence_data(log_dirs):
    """Load all convergence CSV files from specified log directories."""
    all_data = {}

    for log_dir in log_dirs:
        pattern = os.path.join(log_dir, 'convergence_*.csv')
        files = glob.glob(pattern)

        print(f"Found {len(files)} convergence files in {log_dir}")

        for file in files:
            try:
                df = pd.read_csv(file)
                # Get regularization type from first row
                reg_type = identify_regularization_type(df.iloc[0])

                if reg_type not in all_data:
                    all_data[reg_type] = []

                all_data[reg_type].append({
                    'file': file,
                    'data': df
                })
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return all_data

def compute_mean_trajectory(runs, metric='total_circuits'):
    """Compute mean trajectory across all runs for a given metric."""
    # Find max epochs across all runs
    max_epochs = max(len(run['data']) for run in runs)

    # Create matrix to hold all trajectories (runs x epochs)
    trajectories = []

    for run in runs:
        trajectory = run['data'][metric].values.astype(float)
        # Pad with NaN if needed
        if len(trajectory) < max_epochs:
            padded = np.full(max_epochs, np.nan)
            padded[:len(trajectory)] = trajectory
            trajectory = padded
        trajectories.append(trajectory)

    trajectories = np.array(trajectories)

    # Compute mean, ignoring NaNs
    mean_trajectory = np.nanmean(trajectories, axis=0)

    return mean_trajectory, trajectories

def find_best_matching_run(runs, mean_trajectory, metric='total_circuits', max_epochs=200, min_epochs=None):
    """Find the run that best matches the mean trajectory.

    Args:
        runs: List of run dictionaries
        mean_trajectory: Mean trajectory to match against
        metric: Metric to compare
        max_epochs: Maximum number of epochs to consider for comparison
        min_epochs: Minimum number of epochs required (None = no minimum)
    """
    best_run = None
    best_distance = float('inf')
    num_candidates = 0

    for run in runs:
        trajectory = run['data'][metric].values

        # Skip runs that don't meet minimum length requirement
        if min_epochs and len(trajectory) < min_epochs:
            continue

        # Only compare up to max_epochs or the length of this run's trajectory
        n = min(len(trajectory), max_epochs, len(mean_trajectory))

        # Skip runs that are too short (less than 10 epochs)
        if n < 10:
            continue

        num_candidates += 1

        # Compute distance (mean squared error) to mean trajectory
        distance = np.mean((trajectory[:n] - mean_trajectory[:n])**2)

        if distance < best_distance:
            best_distance = distance
            best_run = run

    return best_run, best_distance, num_candidates

def main():
    # Specify the two run directories
    base_dir = '/Users/gwherb/Desktop/CSE5469_Project/MI-identifiability/convergence_100/logs'
    log_dirs = [
        os.path.join(base_dir, 'run_20251201-194142'),
        os.path.join(base_dir, 'run_20251201-212120')
    ]

    # Maximum epochs to consider for comparison
    MAX_EPOCHS = 200

    # Verify directories exist
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            print(f"Warning: {log_dir} does not exist")
            return

    print("="*80)
    print("Finding Representative Runs for L1 and Baseline")
    print(f"(Using first {MAX_EPOCHS} epochs for comparison)")
    print("="*80)
    print()

    # Load all convergence data
    all_data = load_convergence_data(log_dirs)

    print()
    print("Summary of loaded data:")
    for reg_type, runs in all_data.items():
        print(f"  {reg_type}: {len(runs)} runs")
    print()

    # Focus on baseline and L1
    for reg_type in ['baseline', 'l1']:
        if reg_type not in all_data:
            print(f"\n⚠️  No {reg_type} runs found!")
            continue

        runs = all_data[reg_type]

        print("="*80)
        print(f"Analyzing {reg_type.upper()} Regularization ({len(runs)} runs)")
        print("="*80)

        # Compute mean trajectory
        mean_trajectory, all_trajectories = compute_mean_trajectory(runs, metric='total_circuits')

        print(f"\nMean trajectory (first 10 epochs):")
        print(mean_trajectory[:10])
        print(f"\nMean trajectory epochs 190-200:")
        print(mean_trajectory[190:200])

        # Find best matching run - first try with all runs
        best_run_all, best_distance_all, num_candidates_all = find_best_matching_run(
            runs, mean_trajectory, metric='total_circuits', max_epochs=MAX_EPOCHS, min_epochs=None)

        # Also try with runs that have at least 150 epochs
        best_run_long, best_distance_long, num_candidates_long = find_best_matching_run(
            runs, mean_trajectory, metric='total_circuits', max_epochs=MAX_EPOCHS, min_epochs=150)

        # Display results for all runs
        if best_run_all is None:
            print(f"\n⚠️  No suitable runs found for {reg_type.upper()}")
            continue

        print(f"\n{'='*80}")
        print(f"BEST MATCHING RUN for {reg_type.upper()} (considering all {num_candidates_all} runs):")
        print(f"{'='*80}")
        print(f"File: {best_run_all['file']}")
        print(f"Mean Squared Error from mean (first {MAX_EPOCHS} epochs): {best_distance_all:.4f}")
        print(f"Number of epochs in this run: {len(best_run_all['data'])}")
        print(f"\nTrajectory (first 10 epochs):")
        print(best_run_all['data']['total_circuits'].values[:10])
        print(f"\nTrajectory (epochs 190-200 if available):")
        if len(best_run_all['data']) >= 200:
            print(best_run_all['data']['total_circuits'].values[190:200])
        else:
            print(f"  Run only has {len(best_run_all['data'])} epochs")
        print(f"\nFinal circuit count: {best_run_all['data']['total_circuits'].values[-1]}")

        # Extract seed from filename
        filename = os.path.basename(best_run_all['file'])
        if 'seed' in filename:
            seed_part = filename.split('seed')[1].split('_')[0]
            print(f"Seed: {seed_part}")

        # Display results for long runs if available
        if best_run_long is not None and num_candidates_long > 0:
            print(f"\n{'='*80}")
            print(f"BEST MATCHING RUN for {reg_type.upper()} (only runs ≥150 epochs, {num_candidates_long} candidates):")
            print(f"{'='*80}")
            print(f"File: {best_run_long['file']}")
            print(f"Mean Squared Error from mean (first {MAX_EPOCHS} epochs): {best_distance_long:.4f}")
            print(f"Number of epochs in this run: {len(best_run_long['data'])}")
            print(f"\nTrajectory (first 10 epochs):")
            print(best_run_long['data']['total_circuits'].values[:10])
            if len(best_run_long['data']) >= 200:
                print(f"\nTrajectory (epochs 190-200):")
                print(best_run_long['data']['total_circuits'].values[190:200])
            else:
                print(f"\nTrajectory (last 10 epochs):")
                print(best_run_long['data']['total_circuits'].values[-10:])
            print(f"\nFinal circuit count: {best_run_long['data']['total_circuits'].values[-1]}")

            # Extract seed from filename
            filename = os.path.basename(best_run_long['file'])
            if 'seed' in filename:
                seed_part = filename.split('seed')[1].split('_')[0]
                print(f"Seed: {seed_part}")
        else:
            print(f"\n(No runs with ≥150 epochs found for {reg_type.upper()})")

        print()

    print("="*80)
    print("Analysis Complete")
    print("="*80)

if __name__ == '__main__':
    main()
