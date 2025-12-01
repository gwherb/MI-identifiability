#!/bin/bash

# Script to run convergence tracking experiments with different regularization methods
# This will track circuit emergence and sparsity during training

# Configuration
DEVICE="cuda:0"
N_EXPERIMENTS=10  # Number of random seeds per configuration
SIZE=3
DEPTH=2
EPOCHS=1000
BATCH_SIZE=100
TARGET_GATE="XOR"
LOSS_TARGET=0.01
LEARNING_RATE=0.001

# Convergence tracking settings
TRACK_CONVERGENCE="--track-convergence"
CONVERGENCE_FREQUENCY=10  # Track every 10 epochs

# Common arguments
COMMON_ARGS="--verbose \
  --val-frequency 1 \
  --noise-std 0.0 \
  --target-logic-gates $TARGET_GATE \
  --n-experiments $N_EXPERIMENTS \
  --size $SIZE \
  --depth $DEPTH \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --loss-target $LOSS_TARGET \
  --device $DEVICE \
  --use-gpu-batching \
  --gpu-batch-size 128 \
  $TRACK_CONVERGENCE \
  --convergence-frequency $CONVERGENCE_FREQUENCY"

echo "=========================================="
echo "Running Convergence Tracking Experiments"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Target Gate: $TARGET_GATE"
echo "  Network: $SIZE hidden units, $DEPTH layers"
echo "  Experiments: $N_EXPERIMENTS runs per configuration"
echo "  Track every: $CONVERGENCE_FREQUENCY epochs"
echo ""

# Baseline (no regularization)
echo "=========================================="
echo "Running BASELINE (no regularization)"
echo "=========================================="
python main.py $COMMON_ARGS \
  --l1-lambda 0.0 \
  --l2-lambda 0.0 \
  --dropout-rate 0.0

echo ""
echo "Baseline complete!"
echo ""

# L1 Regularization
echo "=========================================="
echo "Running L1 Regularization (lambda=0.001)"
echo "=========================================="
python main.py $COMMON_ARGS \
  --l1-lambda 0.001 \
  --l2-lambda 0.0 \
  --dropout-rate 0.0

echo ""
echo "L1 complete!"
echo ""

# L2 Regularization
echo "=========================================="
echo "Running L2 Regularization (lambda=0.001)"
echo "=========================================="
python main.py $COMMON_ARGS \
  --l1-lambda 0.0 \
  --l2-lambda 0.001 \
  --dropout-rate 0.0

echo ""
echo "L2 complete!"
echo ""

# Dropout
echo "=========================================="
echo "Running Dropout (rate=0.2)"
echo "=========================================="
python main.py $COMMON_ARGS \
  --l1-lambda 0.0 \
  --l2-lambda 0.0 \
  --dropout-rate 0.2

echo ""
echo "Dropout complete!"
echo ""

echo "=========================================="
echo "All convergence experiments complete!"
echo "=========================================="
echo ""
echo "Results saved in: logs/run_*/"
echo "  - convergence_*.csv files contain temporal data"
echo "  - data_tmp.csv contains aggregated results"
echo ""
echo "To visualize results, use the provided Python notebook or scripts."
