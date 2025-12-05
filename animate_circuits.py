#!/usr/bin/env python3
"""
Command-line tool for creating circuit evolution animations.

This script loads detailed tracking data and creates animations showing how
circuits emerge and evolve during training.

Usage:
    # Create single animation
    python animate_circuits.py --json logs/run_XXX/detailed_circuits.json --output animation.mp4

    # Compare two runs
    python animate_circuits.py --compare \
        --json1 logs/baseline/detailed_circuits.json \
        --json2 logs/l1/detailed_circuits.json \
        --output comparison.mp4
"""

import argparse
from pathlib import Path
from mi_identifiability.circuit_animation import (
    create_animation_from_json,
    create_comparison_from_jsons
)
from mi_identifiability.circuit_visualization import load_and_visualize_epoch


def main():
    parser = argparse.ArgumentParser(
        description='Create circuit evolution animations from detailed tracking data'
    )

    # Mode selection
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison animation of two runs')

    # Input files
    parser.add_argument('--json', type=str,
                       help='Path to detailed tracking JSON file (single animation mode)')
    parser.add_argument('--json1', type=str,
                       help='Path to first JSON file (comparison mode)')
    parser.add_argument('--json2', type=str,
                       help='Path to second JSON file (comparison mode)')

    # Output
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for animation (e.g., animation.mp4 or animation.gif)')

    # Animation settings
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second (default: 2)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Resolution/DPI (default: 150)')

    # Comparison labels
    parser.add_argument('--label1', type=str, default='Baseline',
                       help='Label for first run in comparison (default: Baseline)')
    parser.add_argument('--label2', type=str, default='L1',
                       help='Label for second run in comparison (default: L1)')

    # Visualization settings
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 8],
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--node-size', type=int, default=800,
                       help='Base size for neuron nodes (default: 800)')

    # Static snapshot mode
    parser.add_argument('--snapshot', action='store_true',
                       help='Create a static snapshot instead of animation')
    parser.add_argument('--epoch', type=int, default=0,
                       help='Epoch index for snapshot (default: 0)')

    # Verbose
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (shows connectivity information)')

    args = parser.parse_args()

    # Validate arguments
    if args.compare:
        if not args.json1 or not args.json2:
            parser.error('--compare mode requires --json1 and --json2')
    else:
        if not args.json:
            parser.error('Single animation mode requires --json')

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare kwargs
    visualizer_kwargs = {
        'figsize': tuple(args.figsize),
        'node_size': args.node_size
    }

    show_progress = not args.quiet

    # Snapshot mode
    if args.snapshot:
        if args.compare:
            print("Warning: --snapshot mode does not support --compare, using --json1 only")

        json_path = args.json1 if args.compare else args.json
        print(f"Creating snapshot from {json_path} at epoch {args.epoch}")

        fig = load_and_visualize_epoch(
            json_path,
            args.epoch,
            output_path=args.output,
            debug=args.debug,
            **visualizer_kwargs
        )

        print(f"Snapshot saved to {args.output}")
        return

    # Animation mode
    if args.compare:
        print(f"Creating comparison animation:")
        print(f"  {args.label1}: {args.json1}")
        print(f"  {args.label2}: {args.json2}")
        print(f"  Output: {args.output}")
        print(f"  FPS: {args.fps}, DPI: {args.dpi}")

        create_comparison_from_jsons(
            args.json1,
            args.json2,
            args.output,
            labels=(args.label1, args.label2),
            fps=args.fps,
            dpi=args.dpi,
            show_progress=show_progress,
            **visualizer_kwargs
        )
    else:
        print(f"Creating animation from {args.json}")
        print(f"  Output: {args.output}")
        print(f"  FPS: {args.fps}, DPI: {args.dpi}")

        create_animation_from_json(
            args.json,
            args.output,
            fps=args.fps,
            dpi=args.dpi,
            show_progress=show_progress,
            **visualizer_kwargs
        )

    print("Done!")


if __name__ == '__main__':
    main()
