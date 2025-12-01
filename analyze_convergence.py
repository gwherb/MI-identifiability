"""
Analyze and visualize convergence tracking results.

This script loads convergence data from multiple experimental runs and creates
visualizations comparing circuit emergence and sparsity across different
regularization strategies.

Usage:
    python analyze_convergence.py --run-dirs logs/run_20231120-120000 logs/run_20231120-130000

Or from within Python/Colab:
    from analyze_convergence import analyze_convergence_from_dirs
    analyze_convergence_from_dirs(['logs/run_20231120-120000'])
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from mi_identifiability.convergence_visualization import (
    organize_data_by_regularization,
    plot_combined_convergence,
    create_convergence_summary
)


def analyze_convergence_from_dirs(run_dirs, output_dir='convergence_analysis'):
    """
    Analyze convergence data from specified run directories.

    Args:
        run_dirs: List of paths to run directories (strings or Path objects)
        output_dir: Directory to save analysis outputs

    Returns:
        Dictionary with summary statistics and figure objects
    """
    # Convert to Path objects
    run_dirs = [Path(d) for d in run_dirs]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Analyzing convergence data from {len(run_dirs)} run(s)...")

    # Organize data by regularization type
    organized_data = organize_data_by_regularization(run_dirs)

    # Print summary of data found
    print("\nData summary:")
    for reg_name, dfs in organized_data.items():
        print(f"  {reg_name}: {len(dfs)} runs")

    # Create visualizations
    print("\nGenerating visualizations...")
    circuits_fig, sparsity_fig, train_loss_fig, val_loss_fig = plot_combined_convergence(
        organized_data,
        output_dir=output_path
    )

    print(f"  - Saved circuits plot to: {output_path / 'circuits_vs_epochs.png'}")
    print(f"  - Saved sparsity plot to: {output_path / 'sparsity_vs_epochs.png'}")
    print(f"  - Saved training loss plot to: {output_path / 'train_loss_vs_epochs.png'}")
    print(f"  - Saved validation loss plot to: {output_path / 'val_loss_vs_epochs.png'}")

    # Create summary table
    summary_df = create_convergence_summary(organized_data)
    summary_path = output_path / 'convergence_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSummary statistics saved to: {summary_path}")
    print("\nConvergence Summary:")
    print(summary_df.to_string(index=False))

    return {
        'organized_data': organized_data,
        'summary': summary_df,
        'circuits_fig': circuits_fig,
        'sparsity_fig': sparsity_fig,
        'train_loss_fig': train_loss_fig,
        'val_loss_fig': val_loss_fig
    }


def find_convergence_runs(base_dir='logs', pattern='run_*'):
    """
    Find all run directories containing convergence data.

    Args:
        base_dir: Base directory to search in
        pattern: Glob pattern for run directories

    Returns:
        List of Path objects for runs with convergence data
    """
    base_path = Path(base_dir)
    runs_with_convergence = []

    for run_dir in base_path.glob(pattern):
        if run_dir.is_dir():
            # Check if this run has convergence data
            convergence_files = list(run_dir.glob('convergence_*.csv'))
            if convergence_files:
                runs_with_convergence.append(run_dir)

    return runs_with_convergence


def main():
    parser = argparse.ArgumentParser(
        description='Analyze convergence tracking results from experiments'
    )

    parser.add_argument('--run-dirs', nargs='+', type=str,
                       help='List of run directories to analyze')
    parser.add_argument('--base-dir', type=str, default='logs',
                       help='Base directory to search for runs (default: logs)')
    parser.add_argument('--auto-find', action='store_true',
                       help='Automatically find all runs with convergence data')
    parser.add_argument('--output-dir', type=str, default='convergence_analysis',
                       help='Output directory for analysis results (default: convergence_analysis)')

    args = parser.parse_args()

    # Determine which runs to analyze
    if args.auto_find:
        print(f"Auto-finding convergence runs in {args.base_dir}...")
        run_dirs = find_convergence_runs(args.base_dir)
        if not run_dirs:
            print(f"No runs with convergence data found in {args.base_dir}")
            return
        print(f"Found {len(run_dirs)} run(s) with convergence data:")
        for rd in run_dirs:
            print(f"  - {rd}")
    elif args.run_dirs:
        run_dirs = args.run_dirs
    else:
        print("Error: Must specify either --run-dirs or --auto-find")
        parser.print_help()
        return

    # Perform analysis
    results = analyze_convergence_from_dirs(run_dirs, args.output_dir)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
