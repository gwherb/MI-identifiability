# Circuit Evolution Animation Guide

This guide explains how to create animations showing circuit emergence during neural network training.

## Overview

The animation system visualizes:
- **Neuron activation strength** - via green color gradient
- **Circuit participation** - via neuron border thickness
- **Connection importance** - via edge opacity/width
- **Circuit emergence over time** - via frame-by-frame animation

## Visualization Encoding

### Neurons (Nodes)
- **Color**: Activation strength
  - Gray: Not in any circuit
  - Light green: Low activation (in circuits)
  - Dark green: High activation (in circuits)
- **Border thickness**: Circuit participation (relative to total circuits)
  - No border: Not in any circuit
  - Thin border: Used in few circuits
  - Thick border: Used in many circuits
- **Text overlay**: Mean activation value (for active neurons)

### Connections (Edges)
- **Opacity**: How many circuits use this connection (relative to total circuits)
  - Nearly invisible: Not in circuits
  - Semi-transparent: Used in few circuits
  - Fully opaque: Used in many circuits
- **Width**: Scales with circuit participation

## Quick Start

### 1. Run Training with Detailed Tracking

First, collect detailed circuit data during training:

```bash
# Baseline (no regularization)
python main_detailed_tracking.py \
    --verbose \
    --seed 48 \
    --target-logic-gates XOR \
    --n-samples-val 20 \
    --n-repeats 1 \
    --min-sparsity 0 \
    --use-gpu-batching \
    --gpu-batch-size 4096 \
    --n-experiments 1 \
    --size 3 \
    --depth 2 \
    --track-detailed-circuits \
    --convergence-frequency 10 \
    --device cuda:0

# L1 Regularization
python main_detailed_tracking.py \
    --verbose \
    --seed 78 \
    --target-logic-gates XOR \
    --n-samples-val 20 \
    --n-repeats 1 \
    --min-sparsity 0 \
    --use-gpu-batching \
    --gpu-batch-size 4096 \
    --n-experiments 1 \
    --size 3 \
    --depth 2 \
    --l1-lambda 0.001 \
    --track-detailed-circuits \
    --convergence-frequency 10 \
    --device cuda:0
```

This creates JSON files like:
- `logs/run_TIMESTAMP/detailed_circuits_k3_seed48_depth2_lr0.001_loss0.01.json`

### 2. Create Single Run Animation

```bash
python animate_circuits.py \
    --json logs/run_TIMESTAMP/detailed_circuits_k3_seed48_depth2_lr0.001_loss0.01.json \
    --output animations/baseline_evolution.mp4 \
    --fps 2 \
    --dpi 150
```

### 3. Create Comparison Animation (Baseline vs L1)

```bash
python animate_circuits.py \
    --compare \
    --json1 logs/baseline/detailed_circuits_k3_seed48_depth2_lr0.001_loss0.01.json \
    --json2 logs/l1/detailed_circuits_k3_seed78_depth2_lr0.001_loss0.01.json \
    --output animations/baseline_vs_l1.mp4 \
    --label1 "Baseline (Seed 48)" \
    --label2 "L1 Regularization (Seed 78)" \
    --fps 2 \
    --dpi 150
```

### 4. Create Static Snapshot

To examine a specific epoch:

```bash
python animate_circuits.py \
    --json logs/run_TIMESTAMP/detailed_circuits.json \
    --snapshot \
    --epoch 100 \
    --output snapshots/epoch_100.png
```

## Command Line Options

### Basic Options
- `--json`: Path to detailed tracking JSON (single animation mode)
- `--output`: Output path (`.mp4` for video, `.gif` for GIF, `.png` for snapshot)
- `--fps`: Frames per second (default: 2)
- `--dpi`: Resolution (default: 150, higher = better quality but larger files)

### Comparison Mode
- `--compare`: Enable comparison mode
- `--json1`: First JSON file (e.g., baseline)
- `--json2`: Second JSON file (e.g., L1)
- `--label1`: Label for first run (default: "Baseline")
- `--label2`: Label for second run (default: "L1")

### Snapshot Mode
- `--snapshot`: Create static image instead of animation
- `--epoch`: Epoch index to visualize (default: 0)

### Visualization Customization
- `--figsize`: Figure size as `width height` (default: 12 8)
- `--node-size`: Size of neuron nodes (default: 800)

### Other
- `--quiet`: Suppress progress messages

## Python API

You can also use the animation tools programmatically:

### Create Animation

```python
from mi_identifiability.circuit_animation import create_animation_from_json

# Create animation
create_animation_from_json(
    json_path='logs/run_XXX/detailed_circuits.json',
    output_path='animation.mp4',
    fps=2,
    dpi=150
)
```

### Create Comparison

```python
from mi_identifiability.circuit_animation import create_comparison_from_jsons

create_comparison_from_jsons(
    json_path1='logs/baseline/detailed_circuits.json',
    json_path2='logs/l1/detailed_circuits.json',
    output_path='comparison.mp4',
    labels=('Baseline', 'L1'),
    fps=2,
    dpi=150
)
```

### Create Static Visualization

```python
from mi_identifiability.circuit_visualization import load_and_visualize_epoch

fig = load_and_visualize_epoch(
    json_path='logs/run_XXX/detailed_circuits.json',
    epoch_idx=50,
    output_path='epoch_50.png'
)
```

### Advanced: Custom Visualization

```python
import json
from mi_identifiability.circuit_animation import CircuitAnimator
import matplotlib.pyplot as plt

# Load data
with open('detailed_circuits.json', 'r') as f:
    data = json.load(f)

# Create animator with custom settings
animator = CircuitAnimator(
    data,
    figsize=(16, 10),
    node_size=1000,
    layer_spacing=2.5,
    neuron_spacing=1.2
)

# Create animation
animator.create_animation(
    output_path='custom_animation.mp4',
    fps=5,
    dpi=200
)
```

## Prerequisites

### Installing ffmpeg (Optional but Recommended)

For MP4 video output (smaller files, better quality), you need ffmpeg:

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg  # Ubuntu/Debian
# or
conda install ffmpeg  # If using conda
```

**Google Colab:**
```python
# ffmpeg is already installed in Colab
```

**Without ffmpeg:**
- The system will automatically fall back to GIF format
- GIFs work but are larger files

## Output Formats

### MP4 Video (Recommended)
- Best quality and compression
- Requires ffmpeg installed
- Use `.mp4` extension
- Automatically falls back to GIF if ffmpeg not available

```bash
python animate_circuits.py --json data.json --output video.mp4
```

### GIF Animation
- Good for sharing (no player needed)
- Larger file size
- Use `.gif` extension

```bash
python animate_circuits.py --json data.json --output animation.gif
```

### PNG Snapshot
- Static image of single epoch
- Use `--snapshot` flag

```bash
python animate_circuits.py --json data.json --snapshot --epoch 100 --output frame.png
```

## Tips and Best Practices

### Performance
- **Tracking frequency**: Use `--convergence-frequency 10` or higher during training
  - Too frequent = huge JSON files and slow rendering
  - Recommended: 10-20 for smooth animations

- **Animation DPI**:
  - 150 DPI: Good for presentations (default)
  - 300 DPI: Publication quality (larger files)
  - 100 DPI: Quick previews

- **FPS**:
  - 2 FPS: Good default, easy to follow
  - 5 FPS: Faster, still clear
  - 1 FPS: Very slow, for detailed inspection

### Interpretation

**What to look for:**
1. **Early epochs**: Watch for first circuits to emerge (neurons turn green)
2. **Mid training**: See which connections thicken (shared by multiple circuits)
3. **Late training**: Stable structure shows final circuit configuration
4. **Comparison**: L1 should show sparser circuits (fewer thick connections)

**Common patterns:**
- **Circuit emergence**: Neurons suddenly activate (gray → green)
- **Circuit consolidation**: Borders thicken as neurons join more circuits
- **Pruning (with L1)**: Fewer connections remain opaque

### Troubleshooting

**No neurons visible:**
- Check that your JSON has circuit data (not just empty epochs)
- Verify circuits were found during training (check convergence CSV)

**Animation too slow/fast:**
- Adjust `--fps` parameter
- Higher FPS = faster playback

**Poor quality:**
- Increase `--dpi` (150 → 300)
- Increase `--figsize` for larger output

**ffmpeg not found:**
- Install ffmpeg: `conda install ffmpeg` or `brew install ffmpeg`
- Or use GIF format: `--output animation.gif`

## File Locations

After running training and animations, your directory structure should look like:

```
MI-identifiability/
├── logs/
│   ├── run_20251201_baseline/
│   │   ├── detailed_circuits_k3_seed48_depth2_lr0.001_loss0.01.json
│   │   └── convergence_k3_seed48_depth2_lr0.001_loss0.01.csv
│   └── run_20251201_l1/
│       ├── detailed_circuits_k3_seed78_depth2_lr0.001_loss0.01.json
│       └── convergence_k3_seed78_depth2_lr0.001_loss0.01.csv
├── animations/
│   ├── baseline_evolution.mp4
│   ├── l1_evolution.mp4
│   └── baseline_vs_l1_comparison.mp4
└── snapshots/
    ├── baseline_epoch_50.png
    └── l1_epoch_50.png
```

## Example Workflow: Complete Analysis

Here's a complete workflow from training to final comparison animation:

```bash
# 1. Train baseline run (seed 48)
python main_detailed_tracking.py \
    --seed 48 \
    --n-experiments 1 \
    --size 3 --depth 2 \
    --target-logic-gates XOR \
    --track-detailed-circuits \
    --convergence-frequency 10 \
    --device cuda:0

# 2. Train L1 run (seed 78)
python main_detailed_tracking.py \
    --seed 78 \
    --n-experiments 1 \
    --size 3 --depth 2 \
    --target-logic-gates XOR \
    --l1-lambda 0.001 \
    --track-detailed-circuits \
    --convergence-frequency 10 \
    --device cuda:0

# 3. Create individual animations
python animate_circuits.py \
    --json logs/run_BASELINE/detailed_circuits_*.json \
    --output animations/baseline.mp4

python animate_circuits.py \
    --json logs/run_L1/detailed_circuits_*.json \
    --output animations/l1.mp4

# 4. Create comparison
python animate_circuits.py \
    --compare \
    --json1 logs/run_BASELINE/detailed_circuits_*.json \
    --json2 logs/run_L1/detailed_circuits_*.json \
    --output animations/comparison.mp4 \
    --label1 "Baseline" \
    --label2 "L1 Regularization"

# 5. Create snapshots at key epochs
for epoch in 0 50 100 150 200; do
    python animate_circuits.py \
        --json logs/run_BASELINE/detailed_circuits_*.json \
        --snapshot --epoch $epoch \
        --output snapshots/baseline_epoch_${epoch}.png
done
```

## Next Steps

Once you have your animations:
1. Review animations to understand circuit emergence patterns
2. Compare baseline vs L1 to see regularization effects
3. Use snapshots in presentations or papers
4. Analyze which neurons are most important (thick borders)
5. Identify shared circuit structure (thick connections)

For more details on the detailed tracking system, see [DETAILED_TRACKING_README.md](DETAILED_TRACKING_README.md).
