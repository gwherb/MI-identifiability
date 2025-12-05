"""
Circuit visualization module for rendering neural network circuits.

This module provides functions to visualize circuit structure and neuron activations
in a clear, informative way suitable for animation and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection
import networkx as nx


def normalize_activation(activation, min_val=0.0, max_val=1.0):
    """
    Normalize activation value to [0, 1] range.

    Args:
        activation: Raw activation value
        min_val: Minimum expected activation
        max_val: Maximum expected activation

    Returns:
        Normalized activation in [0, 1]
    """
    if max_val == min_val:
        return 0.5
    return np.clip((activation - min_val) / (max_val - min_val), 0, 1)


def get_activation_color(activation, cmap_name='Greens'):
    """
    Convert activation value to RGB color.

    Args:
        activation: Normalized activation value [0, 1]
        cmap_name: Matplotlib colormap name

    Returns:
        RGB tuple
    """
    cmap = plt.get_cmap(cmap_name)
    return cmap(activation)


class CircuitVisualizer:
    """
    Visualizer for neural network circuits with activations.

    Attributes:
        layer_sizes: List of neurons per layer (including input and output)
        figsize: Figure size (width, height)
        node_size: Base size for neurons
        layer_spacing: Horizontal spacing between layers
        neuron_spacing: Vertical spacing between neurons in same layer
    """

    def __init__(self, layer_sizes, figsize=(12, 8), node_size=800,
                 layer_spacing=2.0, neuron_spacing=1.0):
        """
        Initialize the circuit visualizer.

        Args:
            layer_sizes: List of neurons per layer [input_size, hidden1, hidden2, ..., output_size]
            figsize: Figure size
            node_size: Base size for neuron nodes
            layer_spacing: Horizontal distance between layers
            neuron_spacing: Vertical distance between neurons
        """
        self.layer_sizes = layer_sizes
        self.figsize = figsize
        self.node_size = node_size
        self.layer_spacing = layer_spacing
        self.neuron_spacing = neuron_spacing

        # Compute node positions
        self.positions = self._compute_positions()

    def _compute_positions(self):
        """
        Compute (x, y) positions for all neurons in the network.

        Returns:
            Dictionary mapping (layer_idx, neuron_idx) to (x, y) position
        """
        positions = {}

        for layer_idx, layer_size in enumerate(self.layer_sizes):
            x = layer_idx * self.layer_spacing

            # Center neurons vertically
            total_height = (layer_size - 1) * self.neuron_spacing
            y_start = -total_height / 2

            for neuron_idx in range(layer_size):
                y = y_start + neuron_idx * self.neuron_spacing
                positions[(layer_idx, neuron_idx)] = (x, y)

        return positions

    def render_snapshot(self, epoch_data, ax=None, show_legend=True,
                       activation_range=(0.0, 1.0), edge_color='#444444',
                       debug=False):
        """
        Render a single snapshot of the network at a given epoch.

        Args:
            epoch_data: Dictionary with circuit data for this epoch (from DetailedCircuitTracker)
            ax: Matplotlib axis to draw on (creates new if None)
            show_legend: Whether to show the legend
            activation_range: (min, max) activation values for normalization
            edge_color: Base color for edges
            debug: If True, print debugging information about connectivity

        Returns:
            The matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Extract data
        total_circuits = epoch_data.get('total_circuits', 0)
        epoch = epoch_data.get('epoch', 0)

        # Aggregate circuit participation and activations across all outputs
        neuron_circuit_count = {}  # (layer, idx) -> count of circuits using this neuron
        neuron_activations = {}    # (layer, idx) -> mean activation
        edge_circuit_count = {}    # ((layer1, idx1), (layer2, idx2)) -> count

        for output_data in epoch_data.get('outputs', []):
            for circuit in output_data.get('circuits', []):
                node_masks = circuit['node_masks']
                edge_masks = circuit['edge_masks']
                neurons = circuit['neurons']

                # Count neuron participation from node_masks (includes all neurons in circuit, even low-activation ones)
                for layer_idx, node_mask in enumerate(node_masks):
                    for neuron_idx, is_active in enumerate(node_mask):
                        if is_active == 1 or is_active == 1.0:
                            key = (layer_idx, neuron_idx)
                            neuron_circuit_count[key] = neuron_circuit_count.get(key, 0) + 1

                # Store activations for neurons that have activation data
                for neuron_info in neurons:
                    layer = neuron_info['layer']
                    neuron_idx = neuron_info['neuron_idx']
                    key = (layer, neuron_idx)
                    neuron_activations[key] = neuron_info['mean_activation']

                # Count edge participation
                for layer_idx in range(len(edge_masks)):
                    edge_mask = edge_masks[layer_idx]

                    # edge_mask is a 2D array: [to_neuron][from_neuron]
                    # (rows = output/to neurons, columns = input/from neurons)
                    for to_idx in range(len(edge_mask)):
                        for from_idx in range(len(edge_mask[to_idx])):
                            if edge_mask[to_idx][from_idx] == 1:
                                # Edge from layer_idx, neuron from_idx to layer_idx+1, neuron to_idx
                                from_key = (layer_idx, from_idx)
                                to_key = (layer_idx + 1, to_idx)

                                # Only count edge if both positions exist (bounds check)
                                if from_key in self.positions and to_key in self.positions:
                                    edge_key = (from_key, to_key)
                                    edge_circuit_count[edge_key] = edge_circuit_count.get(edge_key, 0) + 1

        # Debug: Check for neurons with no incoming edges (excluding input layer)
        if debug:
            self._debug_connectivity(neuron_circuit_count, edge_circuit_count, epoch, total_circuits)

        # Draw edges first (so they appear behind nodes)
        self._draw_edges(ax, edge_circuit_count, total_circuits, edge_color)

        # Draw nodes
        self._draw_nodes(ax, neuron_circuit_count, neuron_activations,
                        total_circuits, activation_range)

        # Styling
        ax.set_xlim(-0.5, (len(self.layer_sizes) - 1) * self.layer_spacing + 0.5)

        max_layer_size = max(self.layer_sizes)
        y_margin = max_layer_size * self.neuron_spacing * 0.6
        ax.set_ylim(-(max_layer_size - 1) * self.neuron_spacing / 2 - y_margin,
                    (max_layer_size - 1) * self.neuron_spacing / 2 + y_margin)

        ax.set_aspect('equal')
        ax.axis('off')

        # Title
        title = f"Epoch {epoch} - {total_circuits} Perfect Circuits"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add layer labels
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            x = layer_idx * self.layer_spacing
            y = -(max_layer_size - 1) * self.neuron_spacing / 2 - y_margin + 0.3

            if layer_idx == 0:
                label = "Input"
            elif layer_idx == len(self.layer_sizes) - 1:
                label = "Output"
            else:
                label = f"Hidden {layer_idx}"

            ax.text(x, y, label, ha='center', va='top', fontsize=10,
                   fontweight='bold', color='#333333')

        # Legend
        if show_legend and total_circuits > 0:
            self._add_legend(ax, total_circuits)

        return ax

    def _draw_edges(self, ax, edge_circuit_count, total_circuits, edge_color):
        """Draw edges between neurons with opacity based on circuit participation."""
        if total_circuits == 0:
            return

        for (from_key, to_key), count in edge_circuit_count.items():
            # Skip edges that reference non-existent positions (bounds checking)
            if from_key not in self.positions or to_key not in self.positions:
                continue

            from_pos = self.positions[from_key]
            to_pos = self.positions[to_key]

            # Normalize opacity by total circuits
            opacity = count / total_circuits
            opacity = np.clip(opacity, 0.25, 1.0)  # Minimum visibility (increased from 0.1 to 0.25)

            # Width scales with participation
            width = 0.5 + 2.5 * (count / total_circuits)

            ax.plot([from_pos[0], to_pos[0]],
                   [from_pos[1], to_pos[1]],
                   color=edge_color,
                   alpha=opacity,
                   linewidth=width,
                   zorder=1)

    def _debug_connectivity(self, neuron_circuit_count, edge_circuit_count, epoch, total_circuits):
        """
        Debug connectivity: Print neurons with no incoming/outgoing edges.

        Args:
            neuron_circuit_count: Dict mapping (layer, idx) to circuit count
            edge_circuit_count: Dict mapping ((from_layer, from_idx), (to_layer, to_idx)) to count
            epoch: Current epoch number
            total_circuits: Total number of circuits
        """
        print(f"\n{'='*80}")
        print(f"DEBUG: Epoch {epoch} - {total_circuits} total circuits")
        print(f"{'='*80}")

        # Build sets of neurons with incoming and outgoing edges
        neurons_with_incoming = set()
        neurons_with_outgoing = set()

        for (from_key, to_key), count in edge_circuit_count.items():
            neurons_with_outgoing.add(from_key)
            neurons_with_incoming.add(to_key)

        # Check each neuron in circuits
        neurons_in_circuits = set(neuron_circuit_count.keys())

        print(f"\nNeurons in circuits: {len(neurons_in_circuits)}")
        print(f"Network structure: {self.layer_sizes}")

        # Check for neurons with no incoming edges (excluding input layer)
        no_incoming = []
        for neuron_key in neurons_in_circuits:
            layer_idx, neuron_idx = neuron_key
            if layer_idx > 0 and neuron_key not in neurons_with_incoming:  # Skip input layer (layer 0)
                no_incoming.append(neuron_key)

        if no_incoming:
            print(f"\n⚠️  WARNING: {len(no_incoming)} neurons in circuits with NO INCOMING EDGES:")
            for layer_idx, neuron_idx in sorted(no_incoming):
                circuit_count = neuron_circuit_count.get((layer_idx, neuron_idx), 0)
                print(f"  Layer {layer_idx}, Neuron {neuron_idx} (in {circuit_count} circuits)")
        else:
            print(f"\n✓ All non-input neurons in circuits have incoming edges")

        # Check for neurons with no outgoing edges (excluding output layer)
        output_layer = len(self.layer_sizes) - 1
        no_outgoing = []
        for neuron_key in neurons_in_circuits:
            layer_idx, neuron_idx = neuron_key
            if layer_idx < output_layer and neuron_key not in neurons_with_outgoing:
                no_outgoing.append(neuron_key)

        if no_outgoing:
            print(f"\n⚠️  INFO: {len(no_outgoing)} neurons in circuits with NO OUTGOING EDGES (may be normal):")
            for layer_idx, neuron_idx in sorted(no_outgoing):
                circuit_count = neuron_circuit_count.get((layer_idx, neuron_idx), 0)
                print(f"  Layer {layer_idx}, Neuron {neuron_idx} (in {circuit_count} circuits)")
        else:
            print(f"\n✓ All non-output neurons in circuits have outgoing edges")

        # Show edge statistics
        print(f"\nEdge statistics:")
        print(f"  Total edges in circuits: {len(edge_circuit_count)}")

        edge_counts = list(edge_circuit_count.values())
        if edge_counts:
            print(f"  Min edges sharing a connection: {min(edge_counts)}")
            print(f"  Max edges sharing a connection: {max(edge_counts)}")
            print(f"  Avg edges sharing a connection: {sum(edge_counts)/len(edge_counts):.2f}")

        print(f"{'='*80}\n")

    def _draw_nodes(self, ax, neuron_circuit_count, neuron_activations,
                   total_circuits, activation_range):
        """Draw neurons with color based on activation and border based on circuit participation."""
        min_act, max_act = activation_range

        for (layer_idx, neuron_idx), pos in self.positions.items():
            key = (layer_idx, neuron_idx)

            # Get activation (default to 0 if neuron not in any circuit)
            activation = neuron_activations.get(key, 0.0)
            norm_activation = normalize_activation(activation, min_act, max_act)

            # Get circuit participation count
            circuit_count = neuron_circuit_count.get(key, 0)

            # Node color based on activation
            if circuit_count == 0:
                # Not in any circuit - use light gray
                node_color = '#E0E0E0'
                edge_color = '#CCCCCC'
                edge_width = 1.0
            else:
                # In at least one circuit
                if activation < 0.001:
                    # In circuit but very low/no activation - use very light green
                    node_color = '#F1F8F4'  # Very pale green
                else:
                    # Color by activation
                    node_color = get_activation_color(norm_activation, cmap_name='Greens')

                edge_color = '#2E7D32'  # Dark green border

                # Border thickness based on circuit participation (relative to total)
                if total_circuits > 0:
                    participation_ratio = circuit_count / total_circuits
                    edge_width = 1.0 + 5.0 * participation_ratio  # 1 to 6
                else:
                    edge_width = 1.0

            # Draw the node
            circle = plt.Circle(pos, radius=0.15,
                              facecolor=node_color,
                              edgecolor=edge_color,
                              linewidth=edge_width,
                              zorder=2)
            ax.add_patch(circle)

            # Add activation text for neurons in circuits with meaningful activation
            if circuit_count > 0 and activation > 0.01:
                ax.text(pos[0], pos[1], f'{activation:.2f}',
                       ha='center', va='center',
                       fontsize=7, fontweight='bold',
                       color='white' if norm_activation > 0.5 else 'black',
                       zorder=3)

    def _add_legend(self, ax, total_circuits):
        """Add legend explaining the visualization encoding."""
        legend_elements = [
            mpatches.Patch(facecolor='#E0E0E0', edgecolor='#CCCCCC',
                          label='Not in any circuit'),
            mpatches.Patch(facecolor='#F1F8F4', edgecolor='#2E7D32', linewidth=2,
                          label='In circuit (inactive)'),
            mpatches.Patch(facecolor='#A5D6A7', edgecolor='#2E7D32', linewidth=2,
                          label='In circuit (low activation)'),
            mpatches.Patch(facecolor='#2E7D32', edgecolor='#2E7D32', linewidth=3,
                          label='In circuit (high activation)'),
        ]

        ax.legend(handles=legend_elements, loc='upper right',
                 fontsize=8, framealpha=0.9, title='Neuron State')


def visualize_epoch(json_data, epoch_idx, output_path=None, debug=False, **kwargs):
    """
    Visualize a specific epoch from detailed tracking data.

    Args:
        json_data: Loaded JSON data from DetailedCircuitTracker
        epoch_idx: Index of the epoch to visualize (0-based)
        output_path: Optional path to save the figure
        debug: Enable debug output
        **kwargs: Additional arguments passed to CircuitVisualizer

    Returns:
        The matplotlib figure
    """
    if epoch_idx >= len(json_data):
        raise ValueError(f"Epoch index {epoch_idx} out of range (max {len(json_data) - 1})")

    epoch_data = json_data[epoch_idx]

    # Infer layer sizes from node_masks of first circuit
    layer_sizes = None
    for output_data in epoch_data.get('outputs', []):
        if output_data.get('circuits'):
            first_circuit = output_data['circuits'][0]
            layer_sizes = [len(mask) for mask in first_circuit['node_masks']]
            break

    if layer_sizes is None:
        raise ValueError("Could not infer layer sizes from epoch data (no circuits found)")

    # Create visualizer
    visualizer = CircuitVisualizer(layer_sizes, **kwargs)

    # Create figure
    fig, ax = plt.subplots(figsize=visualizer.figsize)
    visualizer.render_snapshot(epoch_data, ax=ax, debug=debug)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def load_and_visualize_epoch(json_path, epoch_idx, output_path=None, debug=False, **kwargs):
    """
    Load detailed tracking JSON and visualize a specific epoch.

    Args:
        json_path: Path to the detailed tracking JSON file
        epoch_idx: Index of the epoch to visualize
        output_path: Optional path to save the figure
        debug: Enable debug output
        **kwargs: Additional arguments passed to CircuitVisualizer

    Returns:
        The matplotlib figure
    """
    import json

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    return visualize_epoch(json_data, epoch_idx, output_path, debug=debug, **kwargs)
