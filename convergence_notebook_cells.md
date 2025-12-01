# New Cells to Add to MI_Identifiability_Experiments.ipynb
# Add these after Section 10 (Analyze Results) as a new section

## Section: Convergence Tracking Experiments

### Cell 1: Markdown Header
```markdown
## 11. Convergence Tracking Experiments

Track how circuits emerge during training to understand the dynamics of circuit formation with different regularization methods.

This will:
- Run training while tracking circuits at regular intervals
- Save circuit counts and sparsity at each checkpoint
- Generate visualizations showing circuit emergence over time
```

### Cell 2: Quick Convergence Test (Single Run)
```python
%%time
# Quick test - single convergence tracking run
print("="*60)
print("Testing Convergence Tracking (1 run, baseline)")
print("="*60)

!python main.py --verbose --val-frequency 10 --noise-std 0.0 \
    --target-logic-gates XOR \
    --n-samples-val 20 \
    --n-repeats 1 \
    --min-sparsity 0 \
    --use-gpu-batching \
    --gpu-batch-size 4096 \
    --n-experiments 1 --size 3 --depth 2 \
    --track-convergence \
    --convergence-frequency 20 \
    --device {DEVICE}

print("\n✓ Convergence tracking test complete!")
print("Check logs/run_*/convergence_*.csv for results")
```

### Cell 3: Full Convergence Tracking - Baseline
```python
%%time
# Full convergence tracking - baseline (no regularization)
print("="*60)
print("Running Full Convergence Tracking - BASELINE")
print("="*60)

!python main.py --verbose --val-frequency 10 --noise-std 0.0 \
    --target-logic-gates XOR \
    --n-samples-val 20 \
    --n-repeats 1 \
    --min-sparsity 0 \
    --use-gpu-batching \
    --gpu-batch-size 4096 \
    --n-experiments 10 --size 3 --depth 2 \
    --track-convergence \
    --convergence-frequency 20 \
    --device {DEVICE}

save_latest_results_to_drive()
print("\n✓ Baseline convergence tracking complete!")
```

### Cell 4: Full Convergence Tracking - L1
```python
%%time
# Full convergence tracking - L1 regularization
print("="*60)
print("Running Full Convergence Tracking - L1 (lambda=0.001)")
print("="*60)

!python main.py --verbose --val-frequency 10 --noise-std 0.0 \
    --target-logic-gates XOR \
    --n-samples-val 20 \
    --n-repeats 1 \
    --min-sparsity 0 \
    --use-gpu-batching \
    --gpu-batch-size 4096 \
    --n-experiments 10 --size 3 --depth 2 \
    --l1-lambda 0.001 \
    --track-convergence \
    --convergence-frequency 20 \
    --device {DEVICE}

save_latest_results_to_drive()
print("\n✓ L1 convergence tracking complete!")
```

### Cell 5: Full Convergence Tracking - L2
```python
%%time
# Full convergence tracking - L2 regularization
print("="*60)
print("Running Full Convergence Tracking - L2 (lambda=0.001)")
print("="*60)

!python main.py --verbose --val-frequency 10 --noise-std 0.0 \
    --target-logic-gates XOR \
    --n-samples-val 20 \
    --n-repeats 1 \
    --min-sparsity 0 \
    --use-gpu-batching \
    --gpu-batch-size 4096 \
    --n-experiments 10 --size 3 --depth 2 \
    --l2-lambda 0.001 \
    --track-convergence \
    --convergence-frequency 20 \
    --device {DEVICE}

save_latest_results_to_drive()
print("\n✓ L2 convergence tracking complete!")
```

### Cell 6: Full Convergence Tracking - Dropout
```python
%%time
# Full convergence tracking - Dropout
print("="*60)
print("Running Full Convergence Tracking - Dropout (rate=0.2)")
print("="*60)

!python main.py --verbose --val-frequency 10 --noise-std 0.0 \
    --target-logic-gates XOR \
    --n-samples-val 20 \
    --n-repeats 1 \
    --min-sparsity 0 \
    --use-gpu-batching \
    --gpu-batch-size 4096 \
    --n-experiments 10 --size 3 --depth 2 \
    --dropout-rate 0.2 \
    --track-convergence \
    --convergence-frequency 20 \
    --device {DEVICE}

save_latest_results_to_drive()
print("\n✓ Dropout convergence tracking complete!")
```

### Cell 7: Analyze Convergence Data
```python
# Analyze convergence tracking results
print("="*60)
print("Analyzing Convergence Data")
print("="*60)

!python analyze_convergence.py --auto-find --output-dir convergence_analysis

# Save convergence analysis to Drive
if IN_COLAB and DRIVE_SAVE_DIR:
    import shutil
    if os.path.exists('convergence_analysis'):
        drive_convergence = f'{DRIVE_SAVE_DIR}/convergence_analysis'
        if os.path.exists(drive_convergence):
            shutil.rmtree(drive_convergence)
        shutil.copytree('convergence_analysis', drive_convergence)
        print(f"\n✓ Convergence analysis saved to: {drive_convergence}")
```

### Cell 8: Display Convergence Summary
```python
# Display convergence summary
import pandas as pd

summary_file = 'convergence_analysis/convergence_summary.csv'
if os.path.exists(summary_file):
    print("\n" + "="*60)
    print("CONVERGENCE SUMMARY")
    print("="*60 + "\n")

    df_summary = pd.read_csv(summary_file)
    display(df_summary)
else:
    print("No convergence summary found. Run convergence tracking experiments first.")
```

### Cell 9: Display Convergence Plots
```python
# Display convergence plots
from IPython.display import Image, display

print("\n" + "="*60)
print("CONVERGENCE PLOTS")
print("="*60 + "\n")

# Circuit count plot
circuits_plot = 'convergence_analysis/circuits_vs_epochs.png'
if os.path.exists(circuits_plot):
    print("Circuit Emergence During Training:")
    display(Image(filename=circuits_plot, width=800))
else:
    print("Circuit emergence plot not found")

# Sparsity plot
sparsity_plot = 'convergence_analysis/sparsity_vs_epochs.png'
if os.path.exists(sparsity_plot):
    print("\nCircuit Sparsity During Training:")
    display(Image(filename=sparsity_plot, width=800))
else:
    print("Sparsity plot not found")
```

### Cell 10: Manual Convergence Visualization (Alternative)
```python
# Alternative: Create convergence plots manually using the module
from mi_identifiability.convergence_visualization import (
    organize_data_by_regularization,
    plot_combined_convergence,
    find_convergence_runs
)

# Find all runs with convergence data
convergence_runs = find_convergence_runs('logs')
print(f"Found {len(convergence_runs)} runs with convergence data")

if convergence_runs:
    # Organize by regularization type
    organized_data = organize_data_by_regularization(convergence_runs)

    # Create plots
    circuits_fig, sparsity_fig = plot_combined_convergence(organized_data)

    # Display
    from IPython.display import display
    display(circuits_fig)
    display(sparsity_fig)
else:
    print("No convergence data found. Run experiments with --track-convergence first")
```

### Cell 11: Examine Individual Convergence Files
```python
# Examine individual convergence CSV files
import glob
import pandas as pd

convergence_files = glob.glob('logs/*/convergence_*.csv')
print(f"Found {len(convergence_files)} convergence tracking files\n")

if convergence_files:
    # Show first file as example
    example_file = convergence_files[0]
    print(f"Example: {example_file}")
    print("="*60)

    df = pd.read_csv(example_file)
    display(df)

    # Plot this individual run
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Circuit count over time
    ax1.plot(df['epoch'], df['total_circuits'], marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Number of Circuits')
    ax1.set_title('Circuit Count vs Epoch')
    ax1.grid(True, alpha=0.3)

    # Sparsity over time
    # Parse the avg_sparsities column (it's a list stored as string)
    avg_sparsities = df['avg_sparsities'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    avg_sparsity_values = [s[0] if isinstance(s, list) and len(s) > 0 else s for s in avg_sparsities]

    ax2.plot(df['epoch'], avg_sparsity_values, marker='o', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Sparsity')
    ax2.set_title('Circuit Sparsity vs Epoch')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("No convergence files found. Run experiments with --track-convergence")
```
