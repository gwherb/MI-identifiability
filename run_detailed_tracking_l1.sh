#!/bin/bash
# Collect detailed tracking data for 100 L1 regularization runs

echo "Starting detailed tracking for 100 L1 regularization experiments..."
echo "====================================================================="
echo ""

python main_detailed_tracking.py \
    --n-experiments 100 \
    --size 3 --depth 2 \
    --target-logic-gates XOR \
    --l1-lambda 0.001 \
    --track-detailed-circuits \
    --convergence-frequency 1 \
    --device cuda:0

echo ""
echo "====================================================================="
echo "L1 detailed tracking complete!"
echo "====================================================================="
