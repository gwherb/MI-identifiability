"""
Detailed circuit tracking module for monitoring circuit evolution during training.

This module extends convergence tracking to capture detailed information about
each circuit, including which neurons are active and their mean activations.
This enables creating animations showing how circuits emerge and evolve.
"""

import json
import numpy as np
import torch
from .circuit import find_circuits


class DetailedCircuitTracker:
    """
    Tracks detailed circuit information during model training, including
    neuron membership and activations for animation purposes.

    Attributes:
        tracking_frequency (int): How often to track circuits (in epochs)
        x_val (torch.Tensor): Validation input data
        y_val (torch.Tensor): Validation target data
        accuracy_threshold (float): Minimum accuracy for perfect circuits
        min_sparsity (float): Minimum sparsity threshold for circuits
        use_gpu_batching (bool): Whether to use GPU batching for circuit search
        gpu_batch_size (int): Batch size for GPU circuit evaluation
        history (list): List of detailed tracking checkpoints
    """

    def __init__(self, tracking_frequency, x_val, y_val,
                 accuracy_threshold=0.99, min_sparsity=0.0,
                 use_gpu_batching=False, gpu_batch_size=128):
        """
        Initialize the detailed circuit tracker.

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

    def _extract_circuit_details(self, circuit, submodel):
        """
        Extract detailed information about a circuit including neuron activations.

        Args:
            circuit: The Circuit object
            submodel: The MLP submodel

        Returns:
            Dictionary with circuit details
        """
        # Get activations for this circuit
        with torch.no_grad():
            activations = submodel(self.x_val, circuit=circuit, return_activations=True)

        circuit_info = {
            'node_masks': [mask.tolist() for mask in circuit.node_masks],
            'edge_masks': [mask.tolist() for mask in circuit.edge_masks],
            'sparsity': circuit.sparsity(),  # (node_sparsity, edge_sparsity, combined_sparsity)
            'neurons': []  # Will store per-neuron information
        }

        # Extract mean activations for each neuron in the circuit
        for layer_idx, node_mask in enumerate(circuit.node_masks[1:], start=1):
            # Skip input layer (layer 0)
            for neuron_idx in range(len(node_mask)):
                if node_mask[neuron_idx] == 1:  # Active neuron
                    # Get activations for this neuron across all validation samples
                    neuron_activations = activations[layer_idx][:, neuron_idx]

                    neuron_info = {
                        'layer': layer_idx,
                        'neuron_idx': neuron_idx,
                        'mean_activation': float(neuron_activations.mean().cpu().item()),
                        'std_activation': float(neuron_activations.std().cpu().item()),
                        'min_activation': float(neuron_activations.min().cpu().item()),
                        'max_activation': float(neuron_activations.max().cpu().item())
                    }

                    circuit_info['neurons'].append(neuron_info)

        return circuit_info

    def track_epoch(self, epoch, model, train_loss=None, val_loss=None, logger=None):
        """
        Track detailed circuit information at the current epoch.

        Args:
            epoch: Current training epoch
            model: The neural network model
            train_loss: Optional training loss at this epoch
            val_loss: Optional validation loss at this epoch
            logger: Optional logger for progress messages

        Returns:
            Dictionary with detailed tracking data for this epoch
        """
        model.eval()

        with torch.no_grad():
            # Separate into submodels if multi-output
            submodels = model.separate_into_k_mlps()
            n_outputs = len(submodels)

            all_circuit_data = []

            for output_idx, submodel in enumerate(submodels):
                # Get validation targets for this output
                y_val_i = self.y_val[..., output_idx].unsqueeze(1)

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

                if logger:
                    logger.info(f'Epoch {epoch + 1} - Output {output_idx}: '
                              f'{n_circuits} circuits, avg sparsity: {avg_sparsity:.4f}')

                # Extract detailed information for each circuit
                circuits_details = []
                for circuit_idx, circuit in enumerate(circuits):
                    circuit_info = self._extract_circuit_details(circuit, submodel)
                    circuit_info['circuit_idx'] = circuit_idx
                    circuit_info['sparsity_value'] = sparsities[circuit_idx]
                    circuits_details.append(circuit_info)

                all_circuit_data.append({
                    'output_idx': output_idx,
                    'n_circuits': n_circuits,
                    'avg_sparsity': avg_sparsity,
                    'circuits': circuits_details
                })

        # Record this checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'total_circuits': sum(output_data['n_circuits'] for output_data in all_circuit_data),
            'outputs': all_circuit_data
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

    def save_to_json(self, filepath):
        """
        Save the tracking history to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath):
        """
        Load tracking history from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            A new DetailedCircuitTracker instance with loaded history
        """
        with open(filepath, 'r') as f:
            history = json.load(f)

        # Create a minimal tracker instance (won't be used for tracking, just for data access)
        tracker = cls(
            tracking_frequency=1,
            x_val=torch.zeros(1, 2),  # Dummy data
            y_val=torch.zeros(1, 1),
            accuracy_threshold=0.99
        )
        tracker.history = history
        return tracker

    def to_summary_dict(self):
        """
        Convert history to a summary dictionary (compatible with original ConvergenceTracker).

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
            'circuit_counts': [[output_data['n_circuits'] for output_data in h['outputs']]
                              for h in self.history],
            'avg_sparsities': [[output_data['avg_sparsity'] for output_data in h['outputs']]
                              for h in self.history],
            'total_circuits': [h['total_circuits'] for h in self.history],
            'train_losses': [h.get('train_loss', None) for h in self.history],
            'val_losses': [h.get('val_loss', None) for h in self.history]
        }
