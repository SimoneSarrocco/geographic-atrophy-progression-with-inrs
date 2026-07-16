"""
Grid search for SIREN omega parameters in GAP-INR.

Explores combinations of omega_0, omega_start, omega_end, and schedule_type
to find the best configuration for FAF reconstruction quality.

Usage:
    # Sequential (one at a time):
    python grid_search_omega.py

    # Generate SLURM array job script:
    python grid_search_omega.py --slurm

    # Dry run (show all configs without launching):
    python grid_search_omega.py --dry-run

    # Resume from a specific run index:
    python grid_search_omega.py --start-from 10

    # Run only a subset (e.g., SLURM array task):
    python grid_search_omega.py --run-index 5
"""

import os
import sys
import yaml
import copy
import json
import argparse
import subprocess
from datetime import datetime
from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
# Grid Definition
# ─────────────────────────────────────────────────────────────────────────────

OMEGA_0_VALUES = [30, 70, 100, 200, 300]

# Each schedule config is (omega_start, omega_end, schedule_type, label)
SCHEDULE_CONFIGS = [
    # Constant
    (30, 30, 'constant', 'const_30'),
    # Linear increasing
    (30, 100, 'linear', 'lin_30_100'),
    (30, 200, 'linear', 'lin_30_200'),
    (30, 300, 'linear', 'lin_30_300'),
    # Exponential increasing
    (30, 100, 'exponential', 'exp_30_100'),
    (30, 200, 'exponential', 'exp_30_200'),
    (30, 300, 'exponential', 'exp_30_300'),
    # Linear decreasing
    (100, 30, 'linear', 'lin_100_30'),
    (200, 30, 'linear', 'lin_200_30'),
    (300, 30, 'linear', 'lin_300_30'),
    # Exponential decreasing
    (100, 30, 'exponential', 'exp_100_30'),
    (200, 30, 'exponential', 'exp_200_30'),
    (300, 30, 'exponential', 'exp_300_30'),
]


def build_grid():
    """Generate all (omega_0, omega_start, omega_end, schedule_type, label) combos."""
    grid = []
    for omega_0 in OMEGA_0_VALUES:
        for omega_start, omega_end, schedule_type, sched_label in SCHEDULE_CONFIGS:
            label = f"o0_{omega_0}__{sched_label}"
            grid.append({
                'omega_0': omega_0,
                'omega_start': omega_start,
                'omega_end': omega_end,
                'schedule_type': schedule_type,
                'label': label,
            })
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Config Generation
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_MODEL_PATH = os.path.join(REPO_ROOT, 'configs', 'config_model.yaml')
CONFIG_DATA_PATH = os.path.join(REPO_ROOT, 'configs', 'config_data.yaml')


def load_base_config():
    """Load and return a deep copy of the base model config."""
    with open(CONFIG_MODEL_PATH, 'r') as f:
        return yaml.safe_load(f)


def apply_omega_config(base_config, grid_entry, output_root):
    """Create a modified config dict for a specific grid entry."""
    cfg = copy.deepcopy(base_config)

    # Apply omega parameters
    cfg['inr_decoder']['omega_0'] = grid_entry['omega_0']
    cfg['inr_decoder']['omega_start'] = grid_entry['omega_start']
    cfg['inr_decoder']['omega_end'] = grid_entry['omega_end']
    cfg['inr_decoder']['schedule_type'] = grid_entry['schedule_type']

    # Set output directory for this run
    cfg['output_dir'] = os.path.join(output_root, grid_entry['label'])

    return cfg


def write_run_config(cfg, run_dir):
    """Write the modified config to a run-specific directory. Returns the config path."""
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_model.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_single(grid_entry, output_root, base_config):
    """Run a single grid search experiment by writing a temporary config and calling run.py."""
    cfg = apply_omega_config(base_config, grid_entry, output_root)
    run_dir = os.path.join(output_root, grid_entry['label'])
    config_path = write_run_config(cfg, run_dir)

    # We override the config file that run.py reads by temporarily
    # replacing the global config, then restoring it. Instead, we launch
    # run.py via subprocess and modify the config in-place for the
    # duration of the call. A cleaner approach: write a temp config and
    # point run.py at it via a small wrapper.

    print(f"\n{'='*70}")
    print(f"  Grid Search Run: {grid_entry['label']}")
    print(f"  omega_0={grid_entry['omega_0']}, "
          f"schedule={grid_entry['schedule_type']}({grid_entry['omega_start']}→{grid_entry['omega_end']})")
    print(f"  Output: {run_dir}")
    print(f"{'='*70}\n")

    # Launch as subprocess using the grid search wrapper entry point
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, 'run_grid_entry.py'),
        '--config-model', config_path,
        '--config-data', CONFIG_DATA_PATH,
    ]

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    # Save grid entry metadata alongside results
    meta_path = os.path.join(run_dir, 'grid_entry.json')
    with open(meta_path, 'w') as f:
        json.dump({
            **grid_entry,
            'exit_code': result.returncode,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    return result.returncode


def generate_slurm_script(grid, output_root):
    """Generate a SLURM array job script for all grid entries."""
    base_config = load_base_config()

    # Write all configs first
    config_paths = []
    for i, entry in enumerate(grid):
        cfg = apply_omega_config(base_config, entry, output_root)
        run_dir = os.path.join(output_root, entry['label'])
        config_path = write_run_config(cfg, run_dir)
        config_paths.append(config_path)

        # Save grid entry metadata
        meta_path = os.path.join(run_dir, 'grid_entry.json')
        os.makedirs(run_dir, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump({**entry, 'index': i}, f, indent=2)

    # Write config path list for SLURM array indexing
    manifest_path = os.path.join(output_root, 'grid_manifest.txt')
    with open(manifest_path, 'w') as f:
        for p in config_paths:
            f.write(p + '\n')

    # Generate SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=omega_grid
#SBATCH --array=0-{len(grid)-1}
#SBATCH --output={output_root}/logs/slurm_%A_%a.out
#SBATCH --error={output_root}/logs/slurm_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Read the config path for this array task
CONFIG_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {manifest_path})

echo "========================================"
echo "  SLURM Array Task: $SLURM_ARRAY_TASK_ID"
echo "  Config: $CONFIG_PATH"
echo "========================================"

cd {REPO_ROOT}
python run_grid_entry.py --config-model "$CONFIG_PATH" --config-data {CONFIG_DATA_PATH}
"""

    slurm_path = os.path.join(output_root, 'submit_grid.sh')
    with open(slurm_path, 'w') as f:
        f.write(slurm_script)
    os.makedirs(os.path.join(output_root, 'logs'), exist_ok=True)

    print(f"Generated SLURM array job script: {slurm_path}")
    print(f"Config manifest: {manifest_path}")
    print(f"Total runs: {len(grid)}")
    print(f"\nSubmit with: sbatch {slurm_path}")

    return slurm_path


def print_grid_summary(grid):
    """Print a formatted summary of all grid configurations."""
    print(f"\n{'='*80}")
    print(f"  OMEGA GRID SEARCH — {len(grid)} configurations")
    print(f"{'='*80}")
    print(f"{'Index':>6}  {'Label':<30}  {'ω₀':>5}  {'ω_start':>8}  {'ω_end':>6}  {'Schedule':<12}")
    print(f"{'-'*80}")
    for i, entry in enumerate(grid):
        print(f"{i:>6}  {entry['label']:<30}  {entry['omega_0']:>5}  "
              f"{entry['omega_start']:>8}  {entry['omega_end']:>6}  {entry['schedule_type']:<12}")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid search over SIREN omega parameters")
    parser.add_argument('--dry-run', action='store_true',
                        help='Print all configurations without running')
    parser.add_argument('--slurm', action='store_true',
                        help='Generate a SLURM array job script instead of running sequentially')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Root directory for grid search outputs')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from this run index (for resuming)')
    parser.add_argument('--run-index', type=int, default=None,
                        help='Run only a single index (for SLURM array tasks)')
    args = parser.parse_args()

    grid = build_grid()

    # Default output root
    if args.output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_config = load_base_config()
        args.output_root = os.path.join(
            base_config.get('output_dir', './tmp'),
            f'omega_grid_{timestamp}'
        )

    if args.dry_run:
        print_grid_summary(grid)
        return

    if args.slurm:
        generate_slurm_script(grid, args.output_root)
        return

    # Sequential execution
    base_config = load_base_config()
    os.makedirs(args.output_root, exist_ok=True)

    # Save full grid manifest
    manifest_path = os.path.join(args.output_root, 'grid_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(grid, f, indent=2)

    if args.run_index is not None:
        # Single run mode
        entry = grid[args.run_index]
        print_grid_summary([entry])
        run_single(entry, args.output_root, base_config)
    else:
        # Run all from start_from
        print_grid_summary(grid[args.start_from:])
        results = {}
        for i, entry in enumerate(grid[args.start_from:], start=args.start_from):
            print(f"\n>>> Run {i+1}/{len(grid)}: {entry['label']}")
            exit_code = run_single(entry, args.output_root, base_config)
            results[entry['label']] = {'exit_code': exit_code, 'index': i}

            # Save running results
            results_path = os.path.join(args.output_root, 'grid_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        # Final summary
        print(f"\n{'='*70}")
        print(f"  Grid Search Complete: {len(results)}/{len(grid)} runs")
        print(f"  Results saved to: {args.output_root}")
        failed = [k for k, v in results.items() if v['exit_code'] != 0]
        if failed:
            print(f"  FAILED runs: {failed}")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
