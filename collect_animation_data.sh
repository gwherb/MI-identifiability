#!/bin/bash
# Script to collect detailed tracking data for animation-ready runs

echo "Collecting detailed tracking data for smooth animations..."
echo "============================================================"
echo ""

# Seed 62 - Best matching pair (baseline + L1)
echo "Run 1/2: Seed 62 Baseline"
echo "-------------------------"
python main_detailed_tracking.py \
    --seed 62 \
    --n-experiments 1 \
    --size 3 --depth 2 \
    --target-logic-gates XOR \
    --track-detailed-circuits \
    --convergence-frequency 1 \
    --device cuda:0

echo ""
echo "Run 2/2: Seed 62 L1 Regularization"
echo "-----------------------------------"
python main_detailed_tracking.py \
    --seed 62 \
    --n-experiments 1 \
    --size 3 --depth 2 \
    --target-logic-gates XOR \
    --l1-lambda 0.001 \
    --track-detailed-circuits \
    --convergence-frequency 1 \
    --device cuda:0

echo ""
echo "============================================================"
echo "Data collection complete!"
echo ""
echo "Next steps:"
echo "1. Find the generated JSON files in logs/"
echo "2. Create animations with animate_circuits.py"
echo "============================================================"
