"""
Visualization tools for convergence tracking analysis.

This module provides functions to visualize circuit emergence and sparsity
during training, comparing different regularization strategies.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def load_convergence_data(run_dir, pattern="convergence_*.csv"):
    """
    Load all convergence tracking CSV files from a run directory.

    Args:
        run_dir: Path to the run directory
        pattern: Glob pattern for convergence files

    Returns:
        List of (filename, DataFrame) tuples
    """
    run_path = Path(run_dir)
    convergence_files = list(run_path.glob(pattern))

    data = []
    for file in sorted(convergence_files):
        df = pd.read_csv(file)
        data.append((file.name, df))

    return data


def plot_circuits_vs_epochs(convergence_data_dict, output_path=None, title="Circuit Emergence During Training"):
    """
    Plot number of circuits vs training epochs for different regularization methods.

    Args:
        convergence_data_dict: Dictionary mapping regularization names to list of DataFrames
                              e.g., {'Normal': [df1, df2], 'L1': [df3, df4], ...}
        output_path: Optional path to save the figure
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'Normal': '#1f77b4',
        'L1': '#ff7f0e',
        'L2': '#2ca02c',
        'Dropout': '#d62728'
    }

    for reg_name, dfs in convergence_data_dict.items():
        if not dfs:
            continue

        # Aggregate data across runs
        all_epochs = []
        all_circuits = []

        for df in dfs:
            all_epochs.extend(df['epoch'].tolist())
            all_circuits.extend(df['total_circuits'].tolist())

        if not all_epochs:
            continue

        # Group by epoch and compute mean/std
        epoch_data = {}
        for epoch, circuits in zip(all_epochs, all_circuits):
            if epoch not in epoch_data:
                epoch_data[epoch] = []
            epoch_data[epoch].append(circuits)

        epochs = sorted(epoch_data.keys())
        mean_circuits = [np.mean(epoch_data[e]) for e in epochs]
        std_circuits = [np.std(epoch_data[e]) for e in epochs]

        color = colors.get(reg_name, None)

        # Plot mean line
        ax.plot(epochs, mean_circuits, label=reg_name, linewidth=2, color=color, marker='o', markersize=4)

        # Plot confidence band (mean ± std)
        ax.fill_between(epochs,
                        np.array(mean_circuits) - np.array(std_circuits),
                        np.array(mean_circuits) + np.array(std_circuits),
                        alpha=0.2, color=color)

    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Number of Perfect Circuits', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_sparsity_vs_epochs(convergence_data_dict, output_path=None, title="Circuit Sparsity During Training"):
    """
    Plot average circuit sparsity vs training epochs for different regularization methods.

    Args:
        convergence_data_dict: Dictionary mapping regularization names to list of DataFrames
        output_path: Optional path to save the figure
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'Normal': '#1f77b4',
        'L1': '#ff7f0e',
        'L2': '#2ca02c',
        'Dropout': '#d62728'
    }

    for reg_name, dfs in convergence_data_dict.items():
        if not dfs:
            continue

        # Aggregate data across runs
        all_epochs = []
        all_sparsities = []

        for df in dfs:
            for idx, row in df.iterrows():
                epoch = row['epoch']
                # avg_sparsities is a list (one per output)
                sparsities = eval(row['avg_sparsities']) if isinstance(row['avg_sparsities'], str) else row['avg_sparsities']

                if isinstance(sparsities, list) and len(sparsities) > 0:
                    avg_sparsity = np.mean(sparsities)
                else:
                    avg_sparsity = sparsities if not isinstance(sparsities, list) else 0.0

                all_epochs.append(epoch)
                all_sparsities.append(avg_sparsity)

        if not all_epochs:
            continue

        # Group by epoch and compute mean/std
        epoch_data = {}
        for epoch, sparsity in zip(all_epochs, all_sparsities):
            if epoch not in epoch_data:
                epoch_data[epoch] = []
            epoch_data[epoch].append(sparsity)

        epochs = sorted(epoch_data.keys())
        mean_sparsity = [np.mean(epoch_data[e]) for e in epochs]
        std_sparsity = [np.std(epoch_data[e]) for e in epochs]

        color = colors.get(reg_name, None)

        # Plot mean line
        ax.plot(epochs, mean_sparsity, label=reg_name, linewidth=2, color=color, marker='o', markersize=4)

        # Plot confidence band
        ax.fill_between(epochs,
                        np.array(mean_sparsity) - np.array(std_sparsity),
                        np.array(mean_sparsity) + np.array(std_sparsity),
                        alpha=0.2, color=color)

    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Average Circuit Sparsity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_combined_convergence(convergence_data_dict, output_dir=None):
    """
    Create a combined figure with both circuit count and sparsity plots.

    Args:
        convergence_data_dict: Dictionary mapping regularization names to list of DataFrames
        output_dir: Optional directory to save the figures

    Returns:
        Tuple of (circuits_fig, sparsity_fig)
    """
    circuits_fig = plot_circuits_vs_epochs(
        convergence_data_dict,
        output_path=Path(output_dir) / "circuits_vs_epochs.png" if output_dir else None
    )

    sparsity_fig = plot_sparsity_vs_epochs(
        convergence_data_dict,
        output_path=Path(output_dir) / "sparsity_vs_epochs.png" if output_dir else None
    )

    return circuits_fig, sparsity_fig


def organize_data_by_regularization(run_dirs):
    """
    Organize convergence data from multiple run directories by regularization type.

    Args:
        run_dirs: List of run directory paths

    Returns:
        Dictionary mapping regularization names to lists of DataFrames
    """
    organized = {
        'Normal': [],
        'L1': [],
        'L2': [],
        'Dropout': []
    }

    for run_dir in run_dirs:
        convergence_files = load_convergence_data(run_dir)

        for filename, df in convergence_files:
            if len(df) == 0:
                continue

            # Determine regularization type from the data
            l1 = df['l1_lambda'].iloc[0]
            l2 = df['l2_lambda'].iloc[0]
            dropout = df['dropout_rate'].iloc[0]

            if l1 > 0:
                organized['L1'].append(df)
            elif l2 > 0:
                organized['L2'].append(df)
            elif dropout > 0:
                organized['Dropout'].append(df)
            else:
                organized['Normal'].append(df)

    return organized


def create_convergence_summary(convergence_data_dict):
    """
    Create a summary table of convergence characteristics.

    Args:
        convergence_data_dict: Dictionary mapping regularization names to list of DataFrames

    Returns:
        pandas DataFrame with summary statistics
    """
    summary_data = []

    for reg_name, dfs in convergence_data_dict.items():
        if not dfs:
            continue

        # Compute statistics across all runs
        final_circuits = []
        final_sparsities = []

        for df in dfs:
            if len(df) > 0:
                final_circuits.append(df['total_circuits'].iloc[-1])

                # Get final sparsity
                final_sparsity_list = df['avg_sparsities'].iloc[-1]
                if isinstance(final_sparsity_list, str):
                    final_sparsity_list = eval(final_sparsity_list)
                if isinstance(final_sparsity_list, list):
                    final_sparsities.append(np.mean(final_sparsity_list))
                else:
                    final_sparsities.append(final_sparsity_list)

        summary_data.append({
            'Regularization': reg_name,
            'Num Runs': len(dfs),
            'Avg Final Circuits': np.mean(final_circuits) if final_circuits else 0,
            'Std Final Circuits': np.std(final_circuits) if final_circuits else 0,
            'Avg Final Sparsity': np.mean(final_sparsities) if final_sparsities else 0,
            'Std Final Sparsity': np.std(final_sparsities) if final_sparsities else 0
        })

    return pd.DataFrame(summary_data)
