#!/bin/bash

# Baseline
python main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR \
  --n-experiments 100 --size 3 --depth 2

# L1 Experiments
for lambda in 0.0001 0.0005 0.001 0.005 0.01 0.05; do
    python main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR \
      --n-experiments 100 --size 3 --depth 2 --l1-lambda $lambda
done

# L2 Experiments
for lambda in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1; do
    python main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR \
      --n-experiments 100 --size 3 --depth 2 --l2-lambda $lambda
done

# Dropout Experiments
for rate in 0.1 0.2 0.3 0.4 0.5; do
    python main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR \
      --n-experiments 100 --size 3 --depth 2 --dropout-rate $rate
done

echo "All experiments complete!"