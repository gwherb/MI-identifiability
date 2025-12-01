"""
Test script for convergence tracking functionality.

This script runs a quick test to verify that convergence tracking is working correctly.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mi_identifiability.logic_gates import ALL_LOGIC_GATES
from mi_identifiability.neural_model import MLP
from mi_identifiability.convergence_tracker import ConvergenceTracker
from mi_identifiability.utils import set_seeds


def test_convergence_tracking():
    """
    Test convergence tracking on a simple XOR problem.
    """
    print("=" * 60)
    print("Testing Convergence Tracking Functionality")
    print("=" * 60)

    # Setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    set_seeds(42)

    # Create XOR data
    xor_gate = ALL_LOGIC_GATES['XOR']
    x_train, y_train = xor_gate.generate_noisy_data(n_repeats=100, noise_std=0.0, device=device)
    x_val, y_val = xor_gate.generate_noisy_data(n_repeats=20, noise_std=0.0, device=device)

    print(f"Training data: {x_train.shape}")
    print(f"Validation data: {x_val.shape}")

    # Create model
    model = MLP(
        hidden_sizes=[3, 3],
        input_size=2,
        output_size=1,
        device=device,
        dropout_rate=0.0
    )
    print(f"\nModel architecture: 2 -> 3 -> 3 -> 1")

    # Create convergence tracker
    tracker = ConvergenceTracker(
        tracking_frequency=5,  # Track every 5 epochs for testing
        x_val=x_val,
        y_val=y_val,
        accuracy_threshold=0.99,
        min_sparsity=0.0,
        use_gpu_batching=torch.cuda.is_available(),
        gpu_batch_size=128
    )
    print(f"Convergence tracker: tracking every 5 epochs")

    # Train with convergence tracking
    print("\n" + "=" * 60)
    print("Training with Convergence Tracking")
    print("=" * 60 + "\n")

    avg_loss = model.do_train(
        x=x_train,
        y=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=32,
        learning_rate=0.001,
        epochs=100,
        loss_target=0.01,
        val_frequency=10,
        early_stopping_steps=30,
        logger=None,
        l1_lambda=0.0,
        l2_lambda=0.0,
        convergence_tracker=tracker
    )

    print(f"\nTraining complete! Final loss: {avg_loss:.4f}")

    # Check results
    val_acc = model.do_eval(x_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Get tracking history
    history = tracker.get_history()
    print(f"\n" + "=" * 60)
    print("Convergence Tracking Results")
    print("=" * 60)
    print(f"\nNumber of checkpoints: {len(history)}")

    if history:
        print("\nCheckpoint Summary:")
        print(f"{'Epoch':<10} {'Total Circuits':<15} {'Avg Sparsity':<15}")
        print("-" * 40)
        for checkpoint in history:
            epoch = checkpoint['epoch']
            total_circuits = checkpoint['total_circuits']
            avg_sparsities = checkpoint['avg_sparsities']
            avg_sparsity = avg_sparsities[0] if len(avg_sparsities) > 0 else 0.0

            print(f"{epoch:<10} {total_circuits:<15} {avg_sparsity:<15.4f}")

        # Verify data integrity
        data_dict = tracker.to_dict()
        assert len(data_dict['epochs']) == len(history), "Epochs mismatch"
        assert len(data_dict['circuit_counts']) == len(history), "Circuit counts mismatch"
        assert len(data_dict['avg_sparsities']) == len(history), "Sparsities mismatch"

        print("\n✓ Data integrity check passed")

        # Check that circuits were actually found
        final_checkpoint = history[-1]
        if final_checkpoint['total_circuits'] > 0:
            print(f"✓ Successfully found {final_checkpoint['total_circuits']} circuits")
        else:
            print("⚠ Warning: No circuits found (model may not have converged)")

        print("\n" + "=" * 60)
        print("SUCCESS: Convergence tracking is working correctly!")
        print("=" * 60)
        return True
    else:
        print("\n⚠ ERROR: No tracking checkpoints recorded!")
        print("This may indicate that tracking frequency was too high for the number of epochs")
        return False


if __name__ == '__main__':
    try:
        success = test_convergence_tracking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
