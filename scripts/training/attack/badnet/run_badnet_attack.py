"""
Master script for running BadNet (or other) poisoning attack experiments.
Trains all specified models on both CIFAR-10 and GTSRB with the chosen attack.

Mirrors the structure of run_baseline_training.py.
Checkpoints are saved to: results/models/attack/<poison_type>/
"""

import argparse
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


def print_header(text, width=120):
    print('\n' + '=' * width)
    print(f'  {text}')
    print('=' * width + '\n')


def run_experiment(experiment_name, args):
    """Delegate one experiment group to batch_train_attack.py."""
    print(f"\n{'=' * 120}")
    print(f"Attack Experiment: {experiment_name}  |  {args.poison_type}  rate={args.poison_rate}")
    print(f"{'=' * 120}\n")

    batch_script = Path(__file__).parent / 'batch_train_attack.py'

    cmd = [
        'python', str(batch_script),
        '--experiment', experiment_name,
        '--poison-type', args.poison_type,
        '--poison-rate', str(args.poison_rate),
        '--continue-on-error',
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        duration = timedelta(seconds=int(time.time() - start))
        if result.returncode == 0:
            print(f"\n✓ Completed: {experiment_name} ({duration})")
            return True
        else:
            print(f"\n✗ Failed: {experiment_name}")
            return False
    except Exception as e:
        print(f"\n✗ Error running {experiment_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='BadNet attack training — all models on CIFAR-10 and GTSRB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training modes:
  1  All experiments (scratch + finetune + ViT) on both datasets
  2  Scratch only — cifar10_scratch + gtsrb_scratch  [default]
  3  CIFAR-10 only (scratch + finetune + ViT)
  4  GTSRB only   (scratch + finetune + ViT)

Examples:
  # BadNet with 10% poison rate, scratch models only
  python run_badnet_attack.py --poison-type badnet --poison-rate 0.1

  # All experiments with blend attack
  python run_badnet_attack.py --poison-type blend --poison-rate 0.1 --mode 1

  # CIFAR-10 only, all model families
  python run_badnet_attack.py --mode 3
        """
    )

    # Mode
    parser.add_argument('--mode', choices=['1', '2', '3', '4'], default='2',
                        help='Training mode (1=all, 2=scratch only, 3=CIFAR-10 only, 4=GTSRB only)')

    # Attack parameters
    parser.add_argument('--poison-type', default='badnet',
                        choices=['badnet', 'blend', 'basic', 'trojan', 'SIG', 'WaNet',
                                 'adaptive_blend', 'adaptive_patch', 'TaCT', 'dynamic',
                                 'clean_label', 'ISSBA', 'SleeperAgent'],
                        help='Backdoor attack type')
    parser.add_argument('--poison-rate', type=float, default=0.1,
                        help='Fraction of training samples to poison (default: 0.1 = 10%%)')

    args = parser.parse_args()

    # ── select experiment groups ───────────────────────────────────────────────
    all_experiments = [
        'cifar10_scratch', 'cifar10_finetune', 'cifar10_vit',
        'gtsrb_scratch',   'gtsrb_finetune',   'gtsrb_vit',
    ]
    mode_map = {
        '1': all_experiments,
        '2': ['cifar10_scratch', 'gtsrb_scratch'],
        '3': ['cifar10_scratch', 'cifar10_finetune', 'cifar10_vit'],
        '4': ['gtsrb_scratch',   'gtsrb_finetune',   'gtsrb_vit'],
    }
    selected = mode_map[args.mode]

    print_header(f"BACKDOOR ATTACK TRAINING — {args.poison_type.upper()}")
    print(f"Poison type:  {args.poison_type}")
    print(f"Poison rate:  {args.poison_rate:.1%}")
    print(f"Mode:         {args.mode}  →  {len(selected)} experiment(s)")
    print(f"Output:       results/models/attack/{args.poison_type}/\n")
    print("Experiments:")
    for exp in selected:
        print(f"  • {exp}")

    # ── run ────────────────────────────────────────────────────────────────────
    overall_start = time.time()
    completed, failed = [], []

    for i, exp in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Starting: {exp}")
        if run_experiment(exp, args):
            completed.append(exp)
        else:
            failed.append(exp)

    total_duration = timedelta(seconds=int(time.time() - overall_start))

    print_header("ATTACK TRAINING COMPLETE!")
    print(f"Total duration: {total_duration}")
    print(f"Poison type:    {args.poison_type}  rate={args.poison_rate}\n")

    print(f"Completed ({len(completed)}):")
    for exp in completed:
        print(f"  ✓ {exp}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for exp in failed:
            print(f"  ✗ {exp}")

    print(f"\nCheckpoints saved to: {Path('results/models/attack') / args.poison_type}")
    print("\nNext steps:")
    print("  1. Evaluate ASR: python scripts/evaluation/evaluate_model.py --checkpoint <path> ...")
    print("  2. Run defenses: backdoor-toolbox-main/other_defense.py or cleanser.py")
    print('\n' + '=' * 120 + '\n')


if __name__ == '__main__':
    main()
