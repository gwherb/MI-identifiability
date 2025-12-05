"""
Circuit animation module for creating videos of circuit evolution during training.

This module provides functions to create animated visualizations showing how
circuits emerge and evolve over the course of training.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from pathlib import Path

from .circuit_visualization import CircuitVisualizer


class CircuitAnimator:
    """
    Creates animations of circuit evolution during training.

    Attributes:
        json_data: Loaded detailed tracking data
        layer_sizes: Network architecture
        visualizer: CircuitVisualizer instance
        activation_range: (min, max) for activation normalization
    """

    def __init__(self, json_data, figsize=(12, 8), **visualizer_kwargs):
        """
        Initialize the circuit animator.

        Args:
            json_data: Loaded JSON data from DetailedCircuitTracker
            figsize: Figure size for animation
            **visualizer_kwargs: Additional arguments for CircuitVisualizer
        """
        self.json_data = json_data

        # Infer layer sizes from first epoch with circuits
        self.layer_sizes = self._infer_layer_sizes()

        # Compute global activation range across all epochs
        self.activation_range = self._compute_activation_range()

        # Create visualizer
        self.visualizer = CircuitVisualizer(
            self.layer_sizes,
            figsize=figsize,
            **visualizer_kwargs
        )

    def _infer_layer_sizes(self):
        """Infer network layer sizes from the JSON data."""
        for epoch_data in self.json_data:
            for output_data in epoch_data.get('outputs', []):
                if output_data.get('circuits'):
                    first_circuit = output_data['circuits'][0]
                    return [len(mask) for mask in first_circuit['node_masks']]

        raise ValueError("Could not infer layer sizes - no circuits found in data")

    def _compute_activation_range(self):
        """
        Compute global min/max activation values across all epochs.

        Returns:
            (min_activation, max_activation) tuple
        """
        all_activations = []

        for epoch_data in self.json_data:
            for output_data in epoch_data.get('outputs', []):
                for circuit in output_data.get('circuits', []):
                    for neuron in circuit.get('neurons', []):
                        all_activations.append(neuron['mean_activation'])

        if not all_activations:
            return (0.0, 1.0)

        return (min(all_activations), max(all_activations))

    def create_animation(self, output_path, fps=2, dpi=150,
                        show_progress=True, writer='auto'):
        """
        Create an animation showing circuit evolution across all epochs.

        Args:
            output_path: Path to save the animation (e.g., 'animation.mp4' or 'animation.gif')
            fps: Frames per second
            dpi: Resolution
            show_progress: Whether to print progress
            writer: Animation writer ('auto', 'ffmpeg' for MP4, 'pillow' for GIF)

        Returns:
            Path to the saved animation file
        """
        fig, ax = plt.subplots(figsize=self.visualizer.figsize)

        n_frames = len(self.json_data)

        def update_frame(frame_idx):
            """Update function for animation."""
            ax.clear()

            if show_progress and frame_idx % 10 == 0:
                print(f"Rendering frame {frame_idx + 1}/{n_frames}")

            epoch_data = self.json_data[frame_idx]
            self.visualizer.render_snapshot(
                epoch_data,
                ax=ax,
                activation_range=self.activation_range,
                show_legend=(frame_idx == 0)  # Only show legend on first frame
            )

            return ax,

        # Create animation
        anim = FuncAnimation(
            fig,
            update_frame,
            frames=n_frames,
            interval=1000 / fps,
            blit=False,
            repeat=True
        )

        # Save animation
        output_path = Path(output_path)

        # Determine writer based on file extension or explicit parameter
        if writer == 'auto':
            if output_path.suffix == '.gif':
                writer = 'pillow'
            else:
                writer = 'ffmpeg'

        # Try to create the writer, with fallback
        try:
            if writer == 'ffmpeg' or output_path.suffix == '.mp4':
                writer_obj = FFMpegWriter(fps=fps, bitrate=1800)
            elif writer == 'pillow' or output_path.suffix == '.gif':
                writer_obj = PillowWriter(fps=fps)
            else:
                raise ValueError(f"Unknown writer: {writer}")

            if show_progress:
                print(f"Saving animation to {output_path}")

            anim.save(str(output_path), writer=writer_obj, dpi=dpi)

        except (RuntimeError, FileNotFoundError) as e:
            # FFmpeg not available, fall back to GIF
            if 'ffmpeg' in str(e).lower() or isinstance(e, FileNotFoundError):
                print(f"\nWarning: ffmpeg not found. Falling back to GIF format.")

                # Change extension to .gif if it was .mp4
                if output_path.suffix == '.mp4':
                    output_path = output_path.with_suffix('.gif')
                    print(f"Output changed to: {output_path}")

                writer_obj = PillowWriter(fps=fps)

                if show_progress:
                    print(f"Saving animation to {output_path}")

                anim.save(str(output_path), writer=writer_obj, dpi=dpi)
            else:
                raise

        plt.close(fig)

        if show_progress:
            print(f"Animation saved to {output_path}")

        return output_path

    def create_comparison_animation(self, other_json_data, output_path,
                                   labels=("Baseline", "L1"),
                                   fps=2, dpi=150, show_progress=True):
        """
        Create side-by-side comparison animation of two runs.

        Args:
            other_json_data: JSON data for the second run to compare
            output_path: Path to save the comparison animation
            labels: Tuple of (left_label, right_label) for the two runs
            fps: Frames per second
            dpi: Resolution
            show_progress: Whether to print progress

        Returns:
            Path to the saved animation file
        """
        # Create second animator
        other_animator = CircuitAnimator(other_json_data)

        # Use global activation range across both runs
        global_min = min(self.activation_range[0], other_animator.activation_range[0])
        global_max = max(self.activation_range[1], other_animator.activation_range[1])
        global_range = (global_min, global_max)

        # Create side-by-side figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        n_frames = min(len(self.json_data), len(other_json_data))

        def update_frame(frame_idx):
            """Update function for comparison animation."""
            ax1.clear()
            ax2.clear()

            if show_progress and frame_idx % 10 == 0:
                print(f"Rendering comparison frame {frame_idx + 1}/{n_frames}")

            # Left: this run
            epoch_data_1 = self.json_data[frame_idx]
            self.visualizer.render_snapshot(
                epoch_data_1,
                ax=ax1,
                activation_range=global_range,
                show_legend=False
            )
            ax1.set_title(f"{labels[0]} - Epoch {epoch_data_1['epoch']} - "
                         f"{epoch_data_1['total_circuits']} circuits",
                         fontsize=14, fontweight='bold')

            # Right: other run
            epoch_data_2 = other_json_data[frame_idx]
            other_animator.visualizer.render_snapshot(
                epoch_data_2,
                ax=ax2,
                activation_range=global_range,
                show_legend=False
            )
            ax2.set_title(f"{labels[1]} - Epoch {epoch_data_2['epoch']} - "
                         f"{epoch_data_2['total_circuits']} circuits",
                         fontsize=14, fontweight='bold')

            return ax1, ax2

        # Create animation
        anim = FuncAnimation(
            fig,
            update_frame,
            frames=n_frames,
            interval=1000 / fps,
            blit=False,
            repeat=True
        )

        # Save animation
        output_path = Path(output_path)

        # Try to create the writer, with fallback
        try:
            if output_path.suffix == '.mp4':
                writer_obj = FFMpegWriter(fps=fps, bitrate=3600)  # Higher bitrate for comparison
            elif output_path.suffix == '.gif':
                writer_obj = PillowWriter(fps=fps)
            else:
                writer_obj = FFMpegWriter(fps=fps, bitrate=3600)

            if show_progress:
                print(f"Saving comparison animation to {output_path}")

            anim.save(str(output_path), writer=writer_obj, dpi=dpi)

        except (RuntimeError, FileNotFoundError) as e:
            # FFmpeg not available, fall back to GIF
            if 'ffmpeg' in str(e).lower() or isinstance(e, FileNotFoundError):
                print(f"\nWarning: ffmpeg not found. Falling back to GIF format.")

                # Change extension to .gif if it was .mp4
                if output_path.suffix == '.mp4':
                    output_path = output_path.with_suffix('.gif')
                    print(f"Output changed to: {output_path}")

                writer_obj = PillowWriter(fps=fps)

                if show_progress:
                    print(f"Saving comparison animation to {output_path}")

                anim.save(str(output_path), writer=writer_obj, dpi=dpi)
            else:
                raise

        plt.close(fig)

        if show_progress:
            print(f"Comparison animation saved to {output_path}")

        return output_path


def create_animation_from_json(json_path, output_path, fps=2, dpi=150,
                               show_progress=True, **kwargs):
    """
    Load detailed tracking JSON and create an animation.

    Args:
        json_path: Path to the detailed tracking JSON file
        output_path: Path to save the animation
        fps: Frames per second
        dpi: Resolution
        show_progress: Whether to print progress
        **kwargs: Additional arguments for CircuitAnimator

    Returns:
        Path to the saved animation file
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    animator = CircuitAnimator(json_data, **kwargs)
    return animator.create_animation(output_path, fps=fps, dpi=dpi,
                                    show_progress=show_progress)


def create_comparison_from_jsons(json_path1, json_path2, output_path,
                                labels=("Baseline", "L1"),
                                fps=2, dpi=150, show_progress=True, **kwargs):
    """
    Load two detailed tracking JSONs and create a comparison animation.

    Args:
        json_path1: Path to first JSON file (e.g., baseline)
        json_path2: Path to second JSON file (e.g., L1)
        output_path: Path to save the comparison animation
        labels: Tuple of (label1, label2)
        fps: Frames per second
        dpi: Resolution
        show_progress: Whether to print progress
        **kwargs: Additional arguments for CircuitAnimator

    Returns:
        Path to the saved animation file
    """
    with open(json_path1, 'r') as f:
        json_data1 = json.load(f)

    with open(json_path2, 'r') as f:
        json_data2 = json.load(f)

    animator = CircuitAnimator(json_data1, **kwargs)
    return animator.create_comparison_animation(
        json_data2,
        output_path,
        labels=labels,
        fps=fps,
        dpi=dpi,
        show_progress=show_progress
    )
