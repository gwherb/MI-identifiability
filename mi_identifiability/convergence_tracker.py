"""
Convergence tracking module for monitoring circuit emergence during training.

This module provides functionality to track the number of circuits and their
sparsity at various points during training, enabling analysis of how circuits
emerge over time with different regularization strategies.
"""

import numpy as np
import torch
from .circuit import find_circuits


class ConvergenceTracker:
    """
    Tracks circuit emergence and sparsity during model training.

    Attributes:
        tracking_frequency (int): How often to track circuits (in epochs)
        x_val (torch.Tensor): Validation input data
        y_val (torch.Tensor): Validation target data
        accuracy_threshold (float): Minimum accuracy for perfect circuits
        min_sparsity (float): Minimum sparsity threshold for circuits
        use_gpu_batching (bool): Whether to use GPU batching for circuit search
        gpu_batch_size (int): Batch size for GPU circuit evaluation
        history (list): List of tracking checkpoints
    """

    def __init__(self, tracking_frequency, x_val, y_val,
                 accuracy_threshold=0.99, min_sparsity=0.0,
                 use_gpu_batching=False, gpu_batch_size=128):
        """
        Initialize the convergence tracker.

        Args:
            tracking_frequency: How often to track circuits (in epochs)
            x_val: Validation input data
            y_val: Validation target data
            accuracy_threshold: Minimum accuracy for perfect circuits
            min_sparsity: Minimum sparsity threshold for circuits
            use_gpu_batching: Whether to use GPU batching
            gpu_batch_size: Batch size for GPU circuit evaluation
        """
        self.tracking_frequency = tracking_frequency
        self.x_val = x_val
        self.y_val = y_val
        self.accuracy_threshold = accuracy_threshold
        self.min_sparsity = min_sparsity
        self.use_gpu_batching = use_gpu_batching
        self.gpu_batch_size = gpu_batch_size
        self.history = []

    def should_track(self, epoch):
        """
        Determine if we should track circuits at this epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            True if we should track at this epoch
        """
        return (epoch + 1) % self.tracking_frequency == 0

    def track_epoch(self, epoch, model, train_loss=None, val_loss=None, logger=None):
        """
        Track circuits at the current epoch.

        Args:
            epoch: Current training epoch
            model: The neural network model
            train_loss: Optional training loss at this epoch
            val_loss: Optional validation loss at this epoch
            logger: Optional logger for progress messages

        Returns:
            Dictionary with tracking data for this epoch
        """
        model.eval()

        with torch.no_grad():
            # Separate into submodels if multi-output
            submodels = model.separate_into_k_mlps()
            n_outputs = len(submodels)

            all_circuit_counts = []
            all_avg_sparsities = []

            for i, submodel in enumerate(submodels):
                # Get validation targets for this output
                y_val_i = self.y_val[..., i].unsqueeze(1)

                # Find circuits for this submodel
                circuits, sparsities, _ = find_circuits(
                    submodel,
                    self.x_val,
                    y_val_i,
                    accuracy_threshold=self.accuracy_threshold,
                    min_sparsity=self.min_sparsity,
                    use_gpu_batching=self.use_gpu_batching,
                    batch_size=self.gpu_batch_size,
                    use_tqdm=False
                )

                n_circuits = len(circuits)
                avg_sparsity = np.mean(sparsities) if len(sparsities) > 0 else 0.0

                all_circuit_counts.append(n_circuits)
                all_avg_sparsities.append(avg_sparsity)

                if logger:
                    logger.info(f'Epoch {epoch + 1} - Output {i}: '
                              f'{n_circuits} circuits, avg sparsity: {avg_sparsity:.4f}')

        # Record this checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'circuit_counts': all_circuit_counts,
            'avg_sparsities': all_avg_sparsities,
            'total_circuits': sum(all_circuit_counts),
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        self.history.append(checkpoint)

        return checkpoint

    def get_history(self):
        """
        Get the full tracking history.

        Returns:
            List of checkpoint dictionaries
        """
        return self.history

    def to_dict(self):
        """
        Convert history to a dictionary suitable for DataFrame creation.

        Returns:
            Dictionary with epochs, circuit counts, sparsities, and losses as lists
        """
        if not self.history:
            return {
                'epochs': [],
                'circuit_counts': [],
                'avg_sparsities': [],
                'total_circuits': [],
                'train_losses': [],
                'val_losses': []
            }

        return {
            'epochs': [h['epoch'] for h in self.history],
            'circuit_counts': [h['circuit_counts'] for h in self.history],
            'avg_sparsities': [h['avg_sparsities'] for h in self.history],
            'total_circuits': [h['total_circuits'] for h in self.history],
            'train_losses': [h.get('train_loss', None) for h in self.history],
            'val_losses': [h.get('val_loss', None) for h in self.history]
        }
