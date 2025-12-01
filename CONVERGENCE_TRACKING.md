# Convergence Tracking Documentation

This document explains how to use the convergence tracking feature to analyze how circuits emerge during training.

## Overview

The convergence tracking system monitors circuit formation throughout the training process, allowing you to:
- Track the number of perfect circuits at each epoch
- Measure average sparsity of circuits over time
- Compare circuit emergence patterns across different regularization strategies
- Visualize the speed of convergence

## Quick Start

### 1. Run a Single Convergence Experiment

```bash
python main.py \
  --target-logic-gates XOR \
  --n-experiments 1 \
  --size 3 \
  --depth 2 \
  --track-convergence \
  --convergence-frequency 10 \
  --verbose
```

This will:
- Train a single XOR model
- Track circuits every 10 epochs
- Save convergence data to `logs/run_*/convergence_*.csv`

### 2. Run Convergence Comparison Experiments

Use the provided script to compare all regularization methods:

```bash
bash run_convergence_experiments.sh
```

This runs convergence tracking for:
- Baseline (no regularization)
- L1 regularization
- L2 regularization
- Dropout

### 3. Analyze and Visualize Results

```bash
python analyze_convergence.py --auto-find --output-dir convergence_analysis
```

This generates:
- `circuits_vs_epochs.png` - Circuit count over training
- `sparsity_vs_epochs.png` - Circuit sparsity over training
- `convergence_summary.csv` - Summary statistics

## Command-Line Arguments

### Convergence Tracking Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--track-convergence` | flag | False | Enable convergence tracking |
| `--convergence-frequency` | int | 10 | Track circuits every N epochs |

### Example Usage

```bash
# Track every 5 epochs
python main.py --track-convergence --convergence-frequency 5 ...

# Track every 20 epochs (faster, less data)
python main.py --track-convergence --convergence-frequency 20 ...

# Don't track (default behavior)
python main.py ...
```

## Output Files

### 1. Individual Convergence Files

Location: `logs/run_TIMESTAMP/convergence_k3_seed0_depth2_lr0.001_loss0.01.csv`

Format:
```csv
epoch,circuit_counts,avg_sparsities,total_circuits,l1_lambda,l2_lambda,dropout_rate
10,[15],["[0.234]"],15,0.0,0.0,0.0
20,[18],["[0.267]"],18,0.0,0.0,0.0
30,[22],["[0.289]"],22,0.0,0.0,0.0
```

### 2. Main Experiment File (Extended)

Location: `logs/run_TIMESTAMP/data_tmp.csv`

When convergence tracking is enabled, additional columns are added:
- `convergence_epochs`: List of epochs where circuits were tracked
- `convergence_circuit_counts`: List of circuit counts at each epoch
- `convergence_avg_sparsities`: List of average sparsities at each epoch
- `convergence_total_circuits`: List of total circuit counts at each epoch

## Using in Google Colab

Add these cells to your Colab notebook:

### Setup
```python
# Import convergence visualization tools
from mi_identifiability.convergence_visualization import (
    organize_data_by_regularization,
    plot_combined_convergence
)
from analyze_convergence import analyze_convergence_from_dirs
```

### Run Experiments
```python
# Baseline
!python main.py --track-convergence --convergence-frequency 20 \
  --target-logic-gates XOR --n-experiments 10 --size 3 --depth 2 \
  --device cuda:0 --verbose

# L1
!python main.py --track-convergence --convergence-frequency 20 \
  --target-logic-gates XOR --n-experiments 10 --size 3 --depth 2 \
  --l1-lambda 0.001 --device cuda:0 --verbose

# L2
!python main.py --track-convergence --convergence-frequency 20 \
  --target-logic-gates XOR --n-experiments 10 --size 3 --depth 2 \
  --l2-lambda 0.001 --device cuda:0 --verbose

# Dropout
!python main.py --track-convergence --convergence-frequency 20 \
  --target-logic-gates XOR --n-experiments 10 --size 3 --depth 2 \
  --dropout-rate 0.2 --device cuda:0 --verbose
```

### Analyze
```python
# Automatic analysis
results = analyze_convergence_from_dirs(['logs/run_*'], output_dir='convergence_analysis')

# Display summary
display(results['summary'])

# Show plots
display(results['circuits_fig'])
display(results['sparsity_fig'])
```

### Manual Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load a specific convergence file
df = pd.read_csv('logs/run_20231120-120000/convergence_k3_seed0_depth2_lr0.001_loss0.01.csv')

# Plot circuit emergence
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['total_circuits'], marker='o')
plt.xlabel('Training Epoch')
plt.ylabel('Number of Perfect Circuits')
plt.title('Circuit Emergence During Training')
plt.grid(True, alpha=0.3)
plt.show()
```

## Python API

### ConvergenceTracker Class

```python
from mi_identifiability.convergence_tracker import ConvergenceTracker

# Create tracker
tracker = ConvergenceTracker(
    tracking_frequency=10,
    x_val=x_val,
    y_val=y_val,
    accuracy_threshold=0.99,
    min_sparsity=0.0,
    use_gpu_batching=True,
    gpu_batch_size=128
)

# Use during training
model.do_train(
    x=x_train,
    y=y_train,
    x_val=x_val,
    y_val=y_val,
    batch_size=100,
    learning_rate=0.001,
    epochs=1000,
    convergence_tracker=tracker  # Pass tracker here
)

# Get results
history = tracker.get_history()
data_dict = tracker.to_dict()
```

### Visualization Functions

```python
from mi_identifiability.convergence_visualization import (
    load_convergence_data,
    plot_circuits_vs_epochs,
    plot_sparsity_vs_epochs,
    plot_combined_convergence,
    organize_data_by_regularization,
    create_convergence_summary
)

# Load data from run directory
convergence_files = load_convergence_data('logs/run_20231120-120000')

# Organize multiple runs by regularization type
run_dirs = ['logs/run_1', 'logs/run_2', 'logs/run_3']
organized = organize_data_by_regularization(run_dirs)

# Create plots
data_dict = {
    'Normal': [df1, df2],
    'L1': [df3, df4],
    'L2': [df5, df6],
    'Dropout': [df7, df8]
}

circuits_fig = plot_circuits_vs_epochs(data_dict, output_path='circuits.png')
sparsity_fig = plot_sparsity_vs_epochs(data_dict, output_path='sparsity.png')

# Or create both at once
circuits_fig, sparsity_fig = plot_combined_convergence(data_dict, output_dir='output')

# Create summary statistics
summary_df = create_convergence_summary(data_dict)
print(summary_df)
```

## Understanding the Output

### Circuit Count Plot

Shows the number of perfect circuits (accuracy > threshold) discovered at each epoch.

**Interpretation:**
- **Steep initial rise**: Rapid formation of circuits early in training
- **Plateau**: Model has stabilized, no new circuits emerging
- **Higher final count**: More diverse solutions (common with L1)
- **Lower final count**: More focused solutions (common with L2)

### Sparsity Plot

Shows the average node sparsity of perfect circuits at each epoch.

**Interpretation:**
- **Increasing sparsity**: Circuits becoming simpler over time
- **High sparsity**: Circuits use fewer neurons (more efficient)
- **Low sparsity**: Circuits use more neurons (more distributed)
- **L1 regularization**: Typically increases sparsity
- **L2 regularization**: Typically decreases sparsity

## Performance Considerations

### Tracking Frequency

The `--convergence-frequency` parameter controls overhead:

| Frequency | Overhead | Use Case |
|-----------|----------|----------|
| 1 | High | Detailed analysis of early dynamics |
| 5-10 | Medium | Good balance for most analyses |
| 20-50 | Low | Quick overview, less data |

### GPU Batching

Always use `--use-gpu-batching` for convergence tracking:

```bash
python main.py \
  --track-convergence \
  --use-gpu-batching \
  --gpu-batch-size 128
```

This can speed up circuit enumeration by 10-100x.

### Memory Usage

Convergence tracking stores circuit data at each checkpoint. For large networks or frequent tracking:
- Use higher `--convergence-frequency` values
- Reduce `--n-experiments`
- Monitor disk space in `logs/` directory

## Example Research Questions

### 1. Speed of Convergence

**Question:** Does L1 regularization cause circuits to emerge faster?

**Method:**
```bash
# Run convergence experiments
bash run_convergence_experiments.sh

# Analyze
python analyze_convergence.py --auto-find
```

**Look for:** Earlier plateau in L1 vs. baseline in circuits plot

### 2. Sparsity Evolution

**Question:** How does circuit sparsity change during training?

**Method:** Examine sparsity plot, compare early vs. late training

**Look for:** Monotonic increase = progressive simplification

### 3. Regularization Impact

**Question:** Do different regularization methods lead to different final circuit distributions?

**Method:** Compare final values in convergence summary table

**Look for:** Variance in "Avg Final Circuits" and "Avg Final Sparsity"

## Troubleshooting

### No convergence files generated

**Problem:** Ran experiment but no `convergence_*.csv` files

**Solution:**
- Check that you used `--track-convergence` flag
- Ensure model actually converged (check validation accuracy)
- Look for errors in training log

### Plots show no data

**Problem:** Visualization script runs but plots are empty

**Solution:**
- Verify convergence CSV files exist in `logs/run_*/`
- Check that files contain data (not just headers)
- Run `analyze_convergence.py --auto-find` to find all runs

### Out of memory errors

**Problem:** GPU runs out of memory during circuit tracking

**Solution:**
- Increase `--convergence-frequency` (track less often)
- Reduce `--gpu-batch-size`
- Use `--min-sparsity > 0` to reduce circuit search space

### Tracking is too slow

**Problem:** Training takes much longer with convergence tracking

**Solution:**
- Increase `--convergence-frequency` from 10 to 20 or 50
- Ensure `--use-gpu-batching` is enabled
- Increase `--gpu-batch-size` if you have GPU memory

## Files Created by This Feature

1. `mi_identifiability/convergence_tracker.py` - Core tracking module
2. `mi_identifiability/convergence_visualization.py` - Visualization tools
3. `run_convergence_experiments.sh` - Bash script for experiments
4. `analyze_convergence.py` - Analysis CLI tool
5. `convergence_notebook_cells.md` - Colab notebook examples

## Citation

If you use this convergence tracking feature in your research, please cite:

```bibtex
@software{mi_identifiability_convergence,
  title={Convergence Tracking for Neural Circuit Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MI-identifiability}
}
```
