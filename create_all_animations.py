#!/usr/bin/env python3
"""
Batch create animations from all detailed tracking JSON files.
Creates animations organized by condition in an 'animations' folder with subfolders:
- animations/baseline/
- animations/l1/
- animations/l2/
- animations/dropout/
"""

import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import json

# Find all detailed tracking JSON files
json_files = glob.glob("detailed_circuit_tracking_100/logs/*/detailed_circuits*.json")

print(f"Found {len(json_files)} detailed tracking files")
print()

# Create main animations directory with condition subfolders
animations_base_dir = Path("animations")
animations_base_dir.mkdir(exist_ok=True)

# Create condition subdirectories
condition_dirs = {
    'baseline': animations_base_dir / 'baseline',
    'l1': animations_base_dir / 'l1',
    'l2': animations_base_dir / 'l2',
    'dropout': animations_base_dir / 'dropout'
}

for condition_dir in condition_dirs.values():
    condition_dir.mkdir(exist_ok=True)

successful = 0
failed = 0
skipped = 0

# Track counts per condition
condition_counts = {'baseline': 0, 'l1': 0, 'l2': 0, 'dropout': 0}

for json_path in tqdm(json_files, desc="Creating animations"):
    json_path = Path(json_path)

    # Parse filename to extract metadata
    filename = json_path.stem  # e.g., "detailed_circuits_seed42_baseline" or "detailed_circuits_seed42_l1_0.001"

    # Extract seed and regularization type
    if '_seed' in filename:
        parts = filename.split('_seed')[1].split('_')
        seed = parts[0]
        # Remaining parts are the regularization identifier
        reg_id = '_'.join(parts[1:]) if len(parts) > 1 else 'baseline'
    else:
        seed = "unknown"
        reg_id = "unknown"

    # Determine condition folder
    if reg_id == 'baseline':
        condition = 'baseline'
    elif reg_id.startswith('l1'):
        condition = 'l1'
    elif reg_id.startswith('l2'):
        condition = 'l2'
    elif reg_id.startswith('dropout'):
        condition = 'dropout'
    else:
        condition = 'baseline'  # Default to baseline if unknown

    # Quick check: does the JSON have circuits?
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Check if any checkpoint exists with circuits
            has_circuits = any(checkpoint.get('total_circuits', 0) > 0 for checkpoint in data)
            if not has_circuits:
                # print(f"  Skipping {seed}_{reg_id}: No circuits found")
                skipped += 1
                continue
    except Exception as e:
        print(f"  Error reading {json_path}: {e}")
        failed += 1
        continue

    # Determine output path in organized folder structure
    output_path = condition_dirs[condition] / f"animation_seed{seed}_{reg_id}.gif"

    # Skip if already exists
    if output_path.exists():
        skipped += 1
        continue

    # Create animation
    cmd = [
        "python", "animate_circuits.py",
        "--json", str(json_path),
        "--output", str(output_path),
        "--fps", "3",
        "--quiet"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            successful += 1
            condition_counts[condition] += 1
        else:
            print(f"  Failed seed {seed}: {result.stderr[:100]}")
            failed += 1
    except subprocess.TimeoutExpired:
        print(f"  Timeout for seed {seed}")
        failed += 1
    except Exception as e:
        print(f"  Error for seed {seed}: {e}")
        failed += 1

print()
print("="*80)
print("Animation Generation Summary")
print("="*80)
print(f"Successful: {successful}")
print(f"Failed:     {failed}")
print(f"Skipped:    {skipped}")
print(f"Total:      {len(json_files)}")
print()
print("Animations organized by condition in: animations/")
print(f"  - Baseline: {condition_counts['baseline']} animations in animations/baseline/")
print(f"  - L1:       {condition_counts['l1']} animations in animations/l1/")
print(f"  - L2:       {condition_counts['l2']} animations in animations/l2/")
print(f"  - Dropout:  {condition_counts['dropout']} animations in animations/dropout/")
print("="*80)

# Print some example paths
if successful > 0:
    print("\nExample animation locations:")
    for condition in ['baseline', 'l1', 'l2', 'dropout']:
        example_anims = list(condition_dirs[condition].glob("*.gif"))[:1]
        if example_anims:
            print(f"  {example_anims[0]}")
    print("="*80)
