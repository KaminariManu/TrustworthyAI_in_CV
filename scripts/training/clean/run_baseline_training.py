"""
Master training script for running all baseline experiments.
Automatically trains all models on both CIFAR-10 and GTSRB datasets.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def print_header(text, width=120):
    """Print formatted header."""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def run_experiment(experiment_name):
    """Run a training experiment."""
    print(f"\n{'=' * 120}")
    print(f"Training Experiment: {experiment_name}")
    print(f"{'=' * 120}\n")
    
    start_time = time.time()
    
    # Use path relative to this script's location
    batch_train_path = Path(__file__).parent / "batch_train.py"
    
    cmd = [
        sys.executable,
        str(batch_train_path),
        "--experiment", experiment_name,
        "--output-dir", "results/models/clean",
        "--continue-on-error"
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        duration = timedelta(seconds=int(time.time() - start_time))
        
        if result.returncode == 0:
            print(f"\n✓ Completed: {experiment_name} (Duration: {duration})")
            return True
        else:
            print(f"\n✗ Failed: {experiment_name}")
            return False
    except Exception as e:
        print(f"\n✗ Error running {experiment_name}: {e}")
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive baseline training for backdoor attack analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training modes:
  1  All experiments (scratch + finetune + ViT)
  2  Scratch only (faster baseline)  [default]
  3  CIFAR-10 only
  4  GTSRB only

Examples:
  python run_baseline_training.py --mode 1
  python run_baseline_training.py --mode 3 --visualize
        """
    )
    parser.add_argument(
        '--mode', type=str, choices=['1', '2', '3', '4'], default='2',
        help='Training mode (1=all, 2=scratch only, 3=CIFAR-10 only, 4=GTSRB only)'
    )
    return parser.parse_args()


def main():
    """Run all training experiments."""
    args = parse_args()

    print_header("COMPREHENSIVE BASELINE TRAINING")
    print("Training all models on CIFAR-10 and GTSRB datasets\n")

    # All experiments
    experiments = [
        "cifar10_scratch",
        "cifar10_finetune",
        "cifar10_vit",
        "gtsrb_scratch",
        "gtsrb_finetune",
        "gtsrb_vit"
    ]

    mode_map = {
        "1": experiments,
        "2": ["cifar10_scratch", "gtsrb_scratch"],
        "3": ["cifar10_scratch", "cifar10_finetune", "cifar10_vit"],
        "4": ["gtsrb_scratch", "gtsrb_finetune", "gtsrb_vit"],
    }
    selected = mode_map.get(args.mode, ["cifar10_scratch", "gtsrb_scratch"])

    print(f"\nWill train {len(selected)} experiment(s):")
    for exp in selected:
        print(f"  • {exp}")
    
    # Run experiments
    overall_start = time.time()
    completed = []
    failed = []
    
    for i, exp in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Starting: {exp}")
        
        if run_experiment(exp):
            completed.append(exp)
        else:
            failed.append(exp)
    
    # Summary
    total_duration = timedelta(seconds=int(time.time() - overall_start))
    
    print_header("TRAINING COMPLETE!")
    print(f"Total Duration: {total_duration}\n")
    
    print(f"Completed Experiments ({len(completed)}):")
    for exp in completed:
        print(f"  ✓ {exp}")
    
    if failed:
        print(f"\nFailed Experiments ({len(failed)}):")
        for exp in failed:
            print(f"  ✗ {exp}")
    
    print(f"\nResults saved to: {Path('results/models').absolute()}")
    print("\nNext steps:")
    print("  1. Run evaluation:  python scripts/evaluation/evaluate_model.py --checkpoint <path> --model <model> --dataset <dataset> --output results/eval.json")
    print("  2. Generate figures: python scripts/visualization/generate_figures.py --results results/eval.json")
    print("  3. Generate tables:  python scripts/visualization/generate_tables.py --results results/eval.json")

    print("\n" + "=" * 120)
    print("All done! 🎉\n")


if __name__ == "__main__":
    main()
