"""
Master orchestrator for all WaNet backdoor attack experiments.

Delegates to batch_train_wanet_attack.py, one experiment group at a time.

Modes
-----
  1  — ALL 4 groups (cifar10_finetuned, cifar10_scratch, gtsrb_finetuned, gtsrb_scratch)
  2  — scratch only  (cifar10_scratch, gtsrb_scratch)
  3  — CIFAR-10 only (cifar10_finetuned, cifar10_scratch)
  4  — GTSRB only   (gtsrb_finetuned,   gtsrb_scratch)
  5  — finetuned only (cifar10_finetuned, gtsrb_finetuned)

Usage examples
--------------
# Run all experiments with default settings
conda run -n backdoor-toolbox python scripts/training/attack/wanet/run_wanet_attack.py --mode 1

# CIFAR-10 only, heavier poison rate, custom cover rate
conda run -n backdoor-toolbox python scripts/training/attack/wanet/run_wanet_attack.py \\
    --mode 3 --poison-rate 0.10 --cover-rate 0.1

# Dry-run to preview commands
conda run -n backdoor-toolbox python scripts/training/attack/wanet/run_wanet_attack.py \\
    --mode 1 --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BATCH_SCRIPT = Path(__file__).resolve().parent / 'batch_train_wanet_attack.py'

MODE_MAP: dict[int, list[str]] = {
    1: ['cifar10_finetuned', 'cifar10_scratch', 'gtsrb_finetuned', 'gtsrb_scratch'],
    2: ['cifar10_scratch',   'gtsrb_scratch'],
    3: ['cifar10_finetuned', 'cifar10_scratch'],
    4: ['gtsrb_finetuned',   'gtsrb_scratch'],
    5: ['cifar10_finetuned', 'gtsrb_finetuned'],
}


def run_experiment(experiment: str, args: argparse.Namespace) -> bool:
    bar = '=' * 80
    print(f"\n{bar}")
    print(f"  STARTING BATCH: {experiment}")
    print(f"  attack=WaNet  poison_rate={args.poison_rate}  cover_rate={args.cover_rate}")
    print(f"{bar}\n")

    cmd = [
        'python', str(BATCH_SCRIPT),
        '--experiment',  experiment,
        '--poison-rate', str(args.poison_rate),
        '--cover-rate',  str(args.cover_rate),
        '--epochs',      str(args.epochs),
        '--batch-size',  str(args.batch_size),
        '--lr',          str(args.lr),
        '--optimizer',   args.optimizer,
        '--scheduler',   args.scheduler,
        '--patience',    str(args.patience),
        '--output-dir',  str(Path(args.output_dir).resolve()),
        '--gpu',         str(args.gpu),
    ]

    if args.skip_failed:
        cmd.append('--skip-failed')
    if args.dry_run:
        cmd.append('--dry-run')
    if not args.tensorboard:
        cmd.append('--no-tensorboard')
    if args.models:
        cmd += ['--models'] + args.models
    if args.datasets:
        cmd += ['--datasets'] + args.datasets

    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    t_start = time.time()
    result  = subprocess.run(cmd, cwd=str(project_root))
    elapsed = time.time() - t_start

    status = 'SUCCESS' if result.returncode == 0 else f'FAILED (code {result.returncode})'
    print(f"\n{bar}")
    print(f"  BATCH FINISHED: {experiment}")
    print(f"  Status: {status}  |  Time: {elapsed/60:.1f} min")
    print(f"{bar}\n")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Run all WaNet backdoor attack experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  1  All 4 groups (default)
  2  Scratch only
  3  CIFAR-10 only
  4  GTSRB only
  5  Fine-tuned only

Examples:
  # Full run with defaults
  python run_wanet_attack.py --mode 1

  # GTSRB only, 10%% poison, 10%% cover
  python run_wanet_attack.py --mode 4 --poison-rate 0.10 --cover-rate 0.1

  # Preview everything without executing
  python run_wanet_attack.py --mode 1 --dry-run
""",
    )

    parser.add_argument('--mode', type=int, default=1, choices=list(MODE_MAP.keys()),
                        help='Experiment mode (see above)')

    # WaNet attack parameters
    parser.add_argument('--poison-rate', type=float, default=0.05,
                        help='Fraction of training samples to poison')
    parser.add_argument('--cover-rate',  type=float, default=0.1,
                        help='Fraction of clean cover samples (WaNet-specific)')

    # Training hyper-parameters
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=0.01)
    parser.add_argument('--optimizer',  default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--scheduler',  default='multistep',
                        choices=['step', 'multistep', 'cosine', 'none'])
    parser.add_argument('--patience',   type=int,   default=20)

    # System
    parser.add_argument('--gpu',        type=int,   default=0)

    # Output
    parser.add_argument('--output-dir', default='./results/models/attack/WaNet')
    parser.add_argument('--tensorboard', action='store_true', default=True)
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false')

    # Control
    parser.add_argument('--skip-failed', action='store_true',
                        help='Continue to the next group if a batch fails')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands but do not execute them')
    parser.add_argument('--models', nargs='+',
                        help='Restrict to these model names across all groups')
    parser.add_argument('--datasets', nargs='+', choices=['cifar10', 'gtsrb'],
                        help='Restrict to these datasets across all groups')

    args = parser.parse_args()

    experiments = MODE_MAP[args.mode]

    bar = '=' * 80
    print(f"\n{bar}")
    print(f"  WaNet Attack Master Orchestrator")
    print(f"  mode         : {args.mode}")
    print(f"  experiments  : {', '.join(experiments)}")
    print(f"  poison_rate  : {args.poison_rate}")
    print(f"  cover_rate   : {args.cover_rate}")
    print(f"  epochs       : {args.epochs}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"{bar}\n")

    failed  = []
    t_total = time.time()

    for exp in experiments:
        success = run_experiment(exp, args)
        if not success:
            failed.append(exp)
            if not args.skip_failed:
                print("Stopping due to failure. Use --skip-failed to continue.")
                sys.exit(1)

    total_time = time.time() - t_total
    print(f"\n{bar}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  {len(experiments) - len(failed)}/{len(experiments)} groups succeeded")
    print(f"  Total time: {total_time/3600:.2f}h")
    if failed:
        print(f"  Failed groups:")
        for f in failed:
            print(f"    {f}")
    print(f"{bar}\n")

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
