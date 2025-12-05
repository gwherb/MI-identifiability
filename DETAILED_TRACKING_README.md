# Detailed Circuit Tracking for Animation

This extension tracks detailed circuit information during training, including which neurons belong to each circuit and their mean activations. This data can be used to create animations showing how circuits emerge and evolve.

## Quick Start

### 1. Run Training with Detailed Tracking

```bash
# Example: Track detailed circuits for baseline (no regularization)
python main_detailed_tracking.py \
    --verbose \
    --val-frequency 10 \
    --noise-std 0.0 \
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
    --convergence-frequency 20 \
    --device cuda:0
```

### 2. Run with L1 Regularization

```bash
python main_detailed_tracking.py \
    --verbose \
    --val-frequency 10 \
    --noise-std 0.0 \
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
    --convergence-frequency 20 \
    --device cuda:0
```

## Output Files

When `--track-detailed-circuits` is enabled, the script creates:

### 1. JSON File with Detailed Circuit Data
**Filename:** `detailed_circuits_k{size}_seed{seed}_depth{depth}_lr{lr}_loss{loss}.json`

Contains:
- Per-epoch snapshots of all perfect circuits
- For each circuit:
  - Node masks (which neurons are active)
  - Edge masks (which connections are active)
  - Sparsity metrics
  - Per-neuron information:
    - Layer and neuron index
    - Mean activation (across validation samples)
    - Std, min, max activation

### 2. Summary CSV (Backward Compatible)
**Filename:** `convergence_k{size}_seed{seed}_depth{depth}_lr{lr}_loss{loss}.csv`

Standard convergence tracking format (compatible with existing analysis tools).

## Data Structure

### JSON Format Example

```json
[
  {
    "epoch": 20,
    "train_loss": 0.152,
    "val_loss": 0.143,
    "total_circuits": 3,
    "outputs": [
      {
        "output_idx": 0,
        "n_circuits": 3,
        "avg_sparsity": 0.667,
        "circuits": [
          {
            "circuit_idx": 0,
            "node_masks": [[1, 1], [1, 0, 1], [1, 1, 0], [1]],
            "edge_masks": [...],
            "sparsity": [0.667, 0.444, 0.556],
            "sparsity_value": 0.667,
            "neurons": [
              {
                "layer": 1,
                "neuron_idx": 0,
                "mean_activation": 0.523,
                "std_activation": 0.112,
                "min_activation": 0.001,
                "max_activation": 0.982
              },
              ...
            ]
          }
        ]
      }
    ]
  }
]
```

## Loading and Using the Data

### Python Example

```python
from mi_identifiability.detailed_circuit_tracker import DetailedCircuitTracker

# Load detailed tracking data
tracker = DetailedCircuitTracker.load_from_json(
    'logs/run_XXX/detailed_circuits_k3_seed0_depth2_lr0.001_loss0.01.json'
)

# Access history
history = tracker.get_history()

# Iterate through epochs
for checkpoint in history:
    epoch = checkpoint['epoch']
    total_circuits = checkpoint['total_circuits']

    print(f"Epoch {epoch}: {total_circuits} total circuits")

    # Access circuit details for first output
    for circuit in checkpoint['outputs'][0]['circuits']:
        print(f"  Circuit {circuit['circuit_idx']}:")
        print(f"    Sparsity: {circuit['sparsity_value']:.3f}")
        print(f"    Neurons:")

        for neuron in circuit['neurons']:
            layer = neuron['layer']
            idx = neuron['neuron_idx']
            activation = neuron['mean_activation']
            print(f"      Layer {layer}, Neuron {idx}: mean_act={activation:.3f}")
```

## Creating Animations

The detailed tracking data provides everything needed for animation:

1. **Which circuits exist at each epoch** - Track circuit emergence
2. **Which neurons belong to each circuit** - Visualize circuit structure
3. **Neuron activations** - Color code neurons by activation strength

### Suggested Animation Workflow

1. Load detailed tracking JSON
2. For each epoch:
   - Parse circuit node/edge masks
   - Get neuron activations
   - Render network diagram with:
     - Active nodes highlighted
     - Nodes colored by activation
     - Circuits outlined/labeled
3. Compile frames into video

## Performance Notes

**Detailed tracking is slower** than standard convergence tracking because it:
- Stores full circuit information (masks + activations)
- Computes activations for all neurons in all circuits
- Writes larger JSON files

**Recommendations:**
- Use `--convergence-frequency 20` or higher (not every epoch)
- Enable `--use-gpu-batching` for faster circuit discovery
- Increase `--gpu-batch-size` if you have GPU memory

## Comparing with Existing Representative Runs

You can use the detailed tracker on specific seeds you've already identified:

```bash
# Run detailed tracking on representative baseline run (seed 48)
python main_detailed_tracking.py \
    --seed 48 \
    --n-experiments 1 \
    --track-detailed-circuits \
    --convergence-frequency 10 \
    ...

# Run detailed tracking on representative L1 run (seed 78)
python main_detailed_tracking.py \
    --seed 78 \
    --n-experiments 1 \
    --l1-lambda 0.001 \
    --track-detailed-circuits \
    --convergence-frequency 10 \
    ...
```

## Next Steps

1. Run detailed tracking on your representative runs
2. Load the JSON data
3. Create visualization/animation script using the neuron activation data
4. Experiment with different activation metrics if needed (the data includes mean, std, min, max)
