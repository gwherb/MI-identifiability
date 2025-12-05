#!/usr/bin/env python3
"""
Batch create animations from all detailed tracking JSON files.
"""

import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import json

# Find all detailed tracking JSON files
json_files = glob.glob("logs/*/detailed_circuits*.json")

print(f"Found {len(json_files)} detailed tracking files")
print()

# Create output directory
output_dir = Path("animations")
output_dir.mkdir(exist_ok=True)

successful = 0
failed = 0
skipped = 0

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

    # Quick check: does the JSON have L1 lambda?
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Check if any checkpoint exists with circuits
            has_circuits = any(checkpoint.get('total_circuits', 0) > 0 for checkpoint in data)
            if not has_circuits:
                print(f"  Skipping {seed}: No circuits found")
                skipped += 1
                continue
    except Exception as e:
        print(f"  Error reading {json_path}: {e}")
        failed += 1
        continue

    # Determine output filename with regularization type
    output_path = output_dir / f"animation_seed{seed}_{reg_id}.gif"

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
print(f"Animations saved to: {output_dir.absolute()}")
print("="*80)
