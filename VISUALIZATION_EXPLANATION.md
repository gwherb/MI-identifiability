# Understanding Circuit Visualization

This document explains what you're seeing in the circuit evolution animations and why certain patterns appear.

## What You're Looking At

The visualization shows the **full network structure** with circuits highlighted. Each circuit is a sparse subnetwork that perfectly solves the task.

### Neuron Color Legend

1. **Gray (no border)**: Neuron not used in any circuit
2. **Very pale green (with border)**: Neuron is in a circuit but has near-zero activation
3. **Light→Dark green (with border)**: Neuron is in a circuit and has increasing activation strength
4. **Border thickness**: How many circuits use this neuron (relative to total circuits)

### Edge (Connection) Visualization

- **Opacity**: How many circuits use this connection (faint = few, opaque = many)
- **Width**: Also scales with number of circuits using it

## Common Questions

### Q: Why do some neurons have edges going to white/inactive neurons?

**A:** This is **expected and correct behavior**. Here's why:

Circuits are defined by **structural sparsity** (which neurons and edges are used), not just activation patterns. A neuron can be structurally part of a circuit even if it has very low activation on the validation set.

**Example scenario:**
- A circuit might include neuron X in its structure (node_mask = 1)
- Neuron X has near-zero activation because it only activates for certain input patterns
- Those input patterns might not be in the validation set
- The circuit still works perfectly even with neuron X being mostly inactive

This is particularly common with:
- **Input layer neurons**: Some inputs may be rarely used for certain logic gates
- **Intermediate neurons**: May only activate for specific truth table entries
- **Pruned connections**: Structurally present but functionally dormant

### Q: Why do some neurons have no outgoing edges?

**A:** Several reasons:

1. **Output layer**: The final output neuron naturally has no outgoing edges
2. **Dead ends**: Some neurons in sparse circuits may only receive inputs but not send outputs (especially if their activation is gated/conditional)
3. **Pruned pathways**: Edges to certain neurons may have been pruned by the circuit discovery algorithm

This is more common with:
- **L1 regularization**: Creates sparser circuits with more dead ends
- **Complex logic gates**: XOR often requires asymmetric circuit structures

### Q: Should I worry about these inactive neurons?

**A:** No! This is normal. The circuit discovery algorithm finds all neurons that are **structurally necessary** for the circuit to work, even if some have low activation.

What matters:
- ✅ The circuit achieves perfect accuracy on the task
- ✅ The sparsity metrics are reasonable
- ✅ The network converged during training

What doesn't matter:
- ❌ Some neurons have near-zero activation
- ❌ Some neurons lack outgoing edges
- ❌ Some edges connect to inactive neurons

## Interpreting Different Patterns

### Dense Circuits (Baseline, No Regularization)
- Many thick, opaque edges
- Most neurons have borders (in multiple circuits)
- Lots of connections to semi-active neurons

### Sparse Circuits (L1 Regularization)
- Fewer, thinner edges
- Clearer separation between active and inactive neurons
- More dramatic differences in border thickness
- More "dead end" neurons

### Over Time (Animation)
- **Early epochs**: Few circuits, mostly gray neurons
- **Mid training**: Circuits emerge, neurons light up green
- **Late training**: Stable structure, clear circuit participation patterns

## Technical Details

### Node Masks vs Activations

The visualization uses two separate pieces of information:

1. **Node masks** (`node_masks` in JSON): Binary indicators of which neurons are structurally in the circuit
   - Used to determine: Border presence and thickness
   - Source: Circuit discovery algorithm

2. **Activations** (`neurons[*].mean_activation` in JSON): Mean activation values across validation set
   - Used to determine: Green color intensity
   - Source: Forward pass through network with validation data

**Key point**: A neuron can have `node_mask = 1` (in circuit) but `mean_activation ≈ 0` (inactive).

### Why This Matters

Understanding this distinction helps interpret:
- **Structural complexity**: How many neurons/edges are needed (node masks)
- **Functional activity**: Which neurons actually activate (activations)
- **Sparsity**: The circuit uses few neurons, but not all used neurons are always active

## Example Interpretation

**What you might see:**

```
Input Layer: [1, 1]          (both inputs used)
Hidden 1:    [1, 0.8, 0.1]   (neuron 0: green, neuron 1: light green, neuron 2: pale green w/ border)
Hidden 2:    [0.9, 0.05, 0]  (neuron 0: dark green, neuron 1: very pale w/ border, neuron 2: gray)
Output:      [1]             (output neuron)
```

**Interpretation:**
- The circuit uses 2 inputs, 3 neurons in Hidden 1, 2 neurons in Hidden 2, and 1 output
- Most work is done by Hidden1[0,1] and Hidden2[0] (dark/light green = high activation)
- Hidden1[2] and Hidden2[1] are structurally in the circuit but mostly dormant (pale green)
- Hidden2[2] is not used at all (gray, no border)

**This is completely normal!** The circuit works perfectly even with some structurally-included neurons being inactive.

## Debugging Checklist

If you see unexpected patterns:

- [ ] Check total circuit count - is it reasonable? (1-50 typical for k=3, depth=2, XOR)
- [ ] Check sparsity values - are they < 1.0? (should be less than full network)
- [ ] Check convergence - did training reach target loss? (< 0.01 typical)
- [ ] Compare baseline vs L1 - L1 should show sparser circuits

If all these check out, your visualization is showing **real circuit structure**, even if it looks unusual!

## Further Reading

For more on circuit discovery and interpretability:
- See the main project README for circuit discovery algorithm details
- Check convergence plots to understand training dynamics
- Compare multiple runs to see variability in circuit emergence
