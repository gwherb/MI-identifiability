#!/usr/bin/env python3
"""
Example script demonstrating how to use the circuit animation tools.

This script shows various ways to create animations and visualizations
from detailed tracking data.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

from mi_identifiability.circuit_visualization import (
    CircuitVisualizer,
    visualize_epoch,
    load_and_visualize_epoch
)
from mi_identifiability.circuit_animation import (
    CircuitAnimator,
    create_animation_from_json,
    create_comparison_from_jsons
)


def example_1_static_visualization(json_path):
    """
    Example 1: Create a static visualization of a single epoch.
    """
    print("Example 1: Static visualization of epoch 50")

    # Simple way: using helper function
    fig = load_and_visualize_epoch(
        json_path,
        epoch_idx=50,
        output_path='example_epoch_50.png'
    )
    plt.show()

    print("  ✓ Saved to example_epoch_50.png")


def example_2_custom_visualization(json_path):
    """
    Example 2: Create custom visualization with specific settings.
    """
    print("Example 2: Custom visualization with larger nodes")

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get first epoch with circuits
    epoch_data = data[0]

    # Infer layer sizes
    layer_sizes = None
    for output_data in epoch_data['outputs']:
        if output_data['circuits']:
            layer_sizes = [len(mask) for mask in output_data['circuits'][0]['node_masks']]
            break

    if layer_sizes:
        # Create custom visualizer
        visualizer = CircuitVisualizer(
            layer_sizes,
            figsize=(14, 10),
            node_size=1200,  # Larger nodes
            layer_spacing=2.5,
            neuron_spacing=1.2
        )

        fig, ax = plt.subplots(figsize=(14, 10))
        visualizer.render_snapshot(epoch_data, ax=ax)

        plt.savefig('example_custom.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("  ✓ Saved to example_custom.png")


def example_3_simple_animation(json_path):
    """
    Example 3: Create a simple animation from JSON data.
    """
    print("Example 3: Creating animation (this may take a minute...)")

    # Simple one-liner
    create_animation_from_json(
        json_path,
        'example_animation.mp4',
        fps=2,
        dpi=150
    )

    print("  ✓ Saved to example_animation.mp4")


def example_4_custom_animation(json_path):
    """
    Example 4: Create animation with custom settings.
    """
    print("Example 4: Creating high-quality animation")

    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create animator with custom settings
    animator = CircuitAnimator(
        data,
        figsize=(16, 12),
        node_size=1000,
        layer_spacing=3.0
    )

    # Create animation
    animator.create_animation(
        'example_high_quality.mp4',
        fps=5,  # Faster playback
        dpi=200,  # Higher quality
        show_progress=True
    )

    print("  ✓ Saved to example_high_quality.mp4")


def example_5_comparison(json_path1, json_path2):
    """
    Example 5: Create side-by-side comparison animation.
    """
    print("Example 5: Creating comparison animation")

    create_comparison_from_jsons(
        json_path1,
        json_path2,
        'example_comparison.mp4',
        labels=('Run 1', 'Run 2'),
        fps=2,
        dpi=150
    )

    print("  ✓ Saved to example_comparison.mp4")


def example_6_multiple_snapshots(json_path):
    """
    Example 6: Create snapshots at multiple epochs.
    """
    print("Example 6: Creating snapshots at multiple epochs")

    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create output directory
    Path('snapshots').mkdir(exist_ok=True)

    # Create snapshots at regular intervals
    n_snapshots = 5
    interval = len(data) // n_snapshots

    for i in range(n_snapshots):
        epoch_idx = min(i * interval, len(data) - 1)

        fig = visualize_epoch(
            data,
            epoch_idx,
            output_path=f'snapshots/epoch_{data[epoch_idx]["epoch"]}.png'
        )
        plt.close(fig)

        print(f"  ✓ Created snapshot for epoch {data[epoch_idx]['epoch']}")

    print(f"  ✓ All snapshots saved to snapshots/")


def main():
    """
    Main function - demonstrates all examples.

    To use this script, you need at least one detailed tracking JSON file.
    Modify the paths below to point to your actual JSON files.
    """
    print("=" * 80)
    print("Circuit Animation Examples")
    print("=" * 80)
    print()

    # MODIFY THESE PATHS to point to your actual JSON files
    json_path = 'logs/run_TIMESTAMP/detailed_circuits_k3_seed48_depth2_lr0.001_loss0.01.json'
    json_path2 = 'logs/run_TIMESTAMP/detailed_circuits_k3_seed78_depth2_lr0.001_loss0.01.json'

    # Check if files exist
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found")
        print()
        print("To use this example script:")
        print("1. Run training with --track-detailed-circuits")
        print("2. Update the json_path variable in this script")
        print("3. Run this script again")
        return

    # Run examples
    try:
        # Static visualizations
        example_1_static_visualization(json_path)
        print()

        example_2_custom_visualization(json_path)
        print()

        # Animations (uncomment to run - these take longer)
        # example_3_simple_animation(json_path)
        # print()

        # example_4_custom_animation(json_path)
        # print()

        # Comparison (requires two JSON files)
        # if Path(json_path2).exists():
        #     example_5_comparison(json_path, json_path2)
        #     print()

        # Multiple snapshots
        example_6_multiple_snapshots(json_path)
        print()

        print("=" * 80)
        print("Examples complete!")
        print("=" * 80)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
