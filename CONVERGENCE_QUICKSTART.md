# Convergence Tracking Quick Start

Track circuit emergence during training to understand how regularization affects circuit formation speed and patterns.

## 🚀 Quick Commands

### 1. Single Test Run
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

### 2. Compare All Regularization Methods
```bash
bash run_convergence_experiments.sh
```

### 3. Analyze and Visualize
```bash
python analyze_convergence.py --auto-find
```

## 📊 What You Get

Two main plots comparing Normal, L1, L2, and Dropout:

1. **Circuit Count vs Epochs** - Shows how many circuits emerge over training
2. **Sparsity vs Epochs** - Shows average circuit complexity over training

Plus a summary table with statistics.

## 🎯 Key Findings to Look For

- **Faster convergence**: Does one method reach plateau earlier?
- **Final circuit count**: Does one method find more/fewer circuits?
- **Sparsity trends**: Do circuits get simpler or more complex?
- **Stability**: Does one method show more variance?

## 📂 Output Files

```
logs/run_TIMESTAMP/
├── convergence_k3_seed0_depth2_lr0.001_loss0.01.csv  # Individual run data
└── data_tmp.csv  # Aggregated results

convergence_analysis/
├── circuits_vs_epochs.png  # Main visualization 1
├── sparsity_vs_epochs.png  # Main visualization 2
└── convergence_summary.csv # Statistics table
```

## 💡 Google Colab Usage

See `convergence_notebook_cells.md` for cells to add to your notebook, or:

```python
# Run experiments
!python main.py --track-convergence --convergence-frequency 20 \
  --target-logic-gates XOR --n-experiments 10 --device cuda:0

# Analyze
from analyze_convergence import analyze_convergence_from_dirs
results = analyze_convergence_from_dirs(['logs/run_*'])
display(results['circuits_fig'])
display(results['sparsity_fig'])
```

## ⚙️ Main Parameters

| Parameter | Recommended | Purpose |
|-----------|-------------|---------|
| `--convergence-frequency` | 10-20 | How often to count circuits (epochs) |
| `--n-experiments` | 10+ | Number of random seeds |
| `--use-gpu-batching` | Always | Speeds up circuit counting 10-100x |

## 📖 Full Documentation

See [CONVERGENCE_TRACKING.md](CONVERGENCE_TRACKING.md) for complete documentation.
