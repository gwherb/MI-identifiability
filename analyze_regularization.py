#!/usr/bin/env python3
"""
Analysis script for regularization experiments on parallel circuits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

def load_results(results_dir):
    """Load all experiment results from a directory."""
    results_dir = Path(results_dir)
    all_data = []
    
    for csv_file in results_dir.glob("**/df_out.csv"):
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No results found in {results_dir}")
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def compute_summary_statistics(df, group_by='l1_lambda'):
    """Compute summary statistics for each regularization level."""
    summary = df.groupby(group_by).agg({
        'perfect_circuits': ['mean', 'std', 'count'],
        'formulas': ['mean', 'std'],
    }).round(3)
    
    return summary


def plot_circuits_vs_regularization(df, reg_type='l1', output_file=None):
    """
    Plot the number of circuits vs regularization strength.
    
    Args:
        df: DataFrame with results
        reg_type: 'l1', 'l2', or 'dropout'
        output_file: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Effect of {reg_type.upper()} Regularization on Circuits', fontsize=16)
    
    col_name = f'{reg_type}_lambda' if reg_type in ['l1', 'l2'] else 'dropout_rate'
    
    # Flatten circuit counts (they're stored as lists)
    df_exploded = df.copy()
    df_exploded['n_circuits'] = df_exploded['perfect_circuits'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    
    # Plot 1: Number of circuits
    sns.boxplot(data=df_exploded, x=col_name, y='n_circuits', ax=axes[0, 0])
    axes[0, 0].set_title('Number of Circuits Found')
    axes[0, 0].set_ylabel('Circuit Count')
    
    # Plot 2: Number of formulas
    df_exploded['n_formulas'] = df_exploded['formulas'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    sns.boxplot(data=df_exploded, x=col_name, y='n_formulas', ax=axes[0, 1])
    axes[0, 1].set_title('Number of Formulas Matched')
    axes[0, 1].set_ylabel('Formula Count')
    
    # Plot 3: Mean trend line
    mean_circuits = df_exploded.groupby(col_name)['n_circuits'].mean()
    axes[1, 0].plot(mean_circuits.index, mean_circuits.values, marker='o', linewidth=2)
    axes[1, 0].set_title('Mean Circuit Count Trend')
    axes[1, 0].set_xlabel(f'{reg_type.upper()} Regularization')
    axes[1, 0].set_ylabel('Mean Circuit Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution of circuits
    for reg_val in df_exploded[col_name].unique():
        subset = df_exploded[df_exploded[col_name] == reg_val]['n_circuits']
        axes[1, 1].hist(subset, alpha=0.5, label=f'{reg_type}={reg_val}', bins=15)
    axes[1, 1].set_title('Distribution of Circuit Counts')
    axes[1, 1].set_xlabel('Number of Circuits')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    return fig


def statistical_comparison(df, baseline_lambda=0.0, reg_type='l1'):
    """
    Perform statistical tests comparing each regularization level to baseline.
    
    Returns:
        DataFrame with test results
    """
    col_name = f'{reg_type}_lambda' if reg_type in ['l1', 'l2'] else 'dropout_rate'
    
    # Get baseline data
    baseline = df[df[col_name] == baseline_lambda]['perfect_circuits'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    
    results = []
    
    for reg_val in sorted(df[col_name].unique()):
        if reg_val == baseline_lambda:
            continue
            
        treatment = df[df[col_name] == reg_val]['perfect_circuits'].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(baseline, treatment)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((baseline.std()**2 + treatment.std()**2) / 2)
        cohens_d = (treatment.mean() - baseline.mean()) / pooled_std
        
        results.append({
            'regularization': reg_val,
            'baseline_mean': baseline.mean(),
            'treatment_mean': treatment.mean(),
            'difference': treatment.mean() - baseline.mean(),
            'percent_change': 100 * (treatment.mean() - baseline.mean()) / baseline.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'significant': 'Yes' if p_val < 0.05 else 'No'
        })
    
    return pd.DataFrame(results)


def create_comprehensive_report(df, output_dir='analysis_output'):
    """
    Generate a comprehensive analysis report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE REGULARIZATION ANALYSIS REPORT")
    print("="*60)
    
    # Determine which regularization types were tested
    reg_types = []
    if 'l1_lambda' in df.columns and df['l1_lambda'].nunique() > 1:
        reg_types.append('l1')
    if 'l2_lambda' in df.columns and df['l2_lambda'].nunique() > 1:
        reg_types.append('l2')
    if 'dropout_rate' in df.columns and df['dropout_rate'].nunique() > 1:
        reg_types.append('dropout')
    
    for reg_type in reg_types:
        print(f"\n{'='*60}")
        print(f"{reg_type.upper()} REGULARIZATION ANALYSIS")
        print(f"{'='*60}\n")
        
        # Summary statistics
        col_name = f'{reg_type}_lambda' if reg_type in ['l1', 'l2'] else 'dropout_rate'
        print(f"Summary Statistics by {col_name}:")
        summary = compute_summary_statistics(df, group_by=col_name)
        print(summary)
        print()
        
        # Statistical tests
        print("Statistical Comparison to Baseline:")
        stats_results = statistical_comparison(df, reg_type=reg_type)
        print(stats_results.to_string(index=False))
        print()
        
        # Save results
        stats_results.to_csv(output_dir / f'{reg_type}_statistical_tests.csv', index=False)
        
        # Generate plots
        plot_file = output_dir / f'{reg_type}_circuits_analysis.png'
        plot_circuits_vs_regularization(df, reg_type=reg_type, output_file=plot_file)
    
    # Generate summary
    summary_file = output_dir / 'analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("REGULARIZATION EFFECTS ON PARALLEL CIRCUITS\n")
        f.write("="*60 + "\n\n")
        
        for reg_type in reg_types:
            f.write(f"\n{reg_type.upper()} Regularization:\n")
            f.write("-"*40 + "\n")
            
            stats_df = statistical_comparison(df, reg_type=reg_type)
            
            # Find strongest effect
            max_effect_idx = stats_df['cohens_d'].abs().idxmax()
            max_effect = stats_df.loc[max_effect_idx]
            
            f.write(f"Strongest effect at {col_name}={max_effect['regularization']}\n")
            f.write(f"  Mean change: {max_effect['difference']:.2f} circuits ")
            f.write(f"({max_effect['percent_change']:.1f}%)\n")
            f.write(f"  Effect size (Cohen's d): {max_effect['cohens_d']:.3f}\n")
            f.write(f"  Statistical significance: p={max_effect['p_value']:.4f}\n")
            
            # Hypothesis check
            if reg_type == 'l1':
                expected = "decrease"
                observed = "decrease" if max_effect['difference'] < 0 else "increase"
            else:  # l2 or dropout
                expected = "increase"
                observed = "increase" if max_effect['difference'] > 0 else "decrease"
            
            hypothesis_supported = expected == observed and max_effect['p_value'] < 0.05
            f.write(f"  Hypothesis supported: {'YES' if hypothesis_supported else 'NO'}\n")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze regularization experiment results')
    parser.add_argument('results_dir', type=str, help='Directory containing df_out.csv files')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Load and analyze
    df = load_results(args.results_dir)
    create_comprehensive_report(df, output_dir=args.output_dir)