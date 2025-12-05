#!/bin/bash
# Collect detailed tracking data for 100 baseline runs (no regularization)

echo "Starting detailed tracking for 100 baseline experiments..."
echo "=========================================================="
echo ""

python main_detailed_tracking.py \
    --n-experiments 100 \
    --size 3 --depth 2 \
    --target-logic-gates XOR \
    --track-detailed-circuits \
    --convergence-frequency 1 \
    --device cuda:0

echo ""
echo "=========================================================="
echo "Baseline detailed tracking complete!"
echo "=========================================================="
