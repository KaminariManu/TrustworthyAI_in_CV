"""
Batch trainer for WaNet backdoor attack experiments.

Reads a named experiment group and runs train_wanet_attack.py once for each
(model, dataset, pretrained) combination in that group.

Usage examples
--------------
# Run the default CIFAR-10 fine-tuned group
python scripts/training/attack/wanet/batch_train_wanet_attack.py \\
    --experiment cifar10_finetuned --poison-rate 0.05 --cover-rate 0.1

# Run all ResNet variants on GTSRB from scratch
python scripts/training/attack/wanet/batch_train_wanet_attack.py \\
    --experiment gtsrb_scratch --poison-rate 0.05 --cover-rate 0.1

Available experiment groups
---------------------------
  cifar10_finetuned   VGG16/19, ResNet18/34/50, ViT/DeiT  – CIFAR-10, pretrained
  cifar10_scratch     VGG16/19, ResNet18/34/50, ViT/DeiT  – CIFAR-10, from scratch
  gtsrb_finetuned     VGG16/19, ResNet18/34/50, ViT/DeiT  – GTSRB,    pretrained
  gtsrb_scratch       VGG16/19, ResNet18/34/50, ViT/DeiT  – GTSRB,    from scratch
  cnn_cifar10         small-CNN baselines – CIFAR-10
  cnn_gtsrb           small-CNN baselines – GTSRB
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── experiment definitions ────────────────────────────────────────────────────
MODELS_CLASSIC = ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50']
MODELS_TRANS   = ['vit_small', 'deit_small']
ALL_MODELS     = MODELS_CLASSIC + MODELS_TRANS

ATTACK_EXPERIMENTS = {
    'cifar10_finetuned': [
        {'model': m, 'dataset': 'cifar10', 'pretrained': True}
        for m in ALL_MODELS
    ],
    'cifar10_scratch': [
        {'model': m, 'dataset': 'cifar10', 'pretrained': False}
        for m in ALL_MODELS
    ],
    'gtsrb_finetuned': [
        {'model': m, 'dataset': 'gtsrb', 'pretrained': True}
        for m in ALL_MODELS
    ],
    'gtsrb_scratch': [
        {'model': m, 'dataset': 'gtsrb', 'pretrained': False}
        for m in ALL_MODELS
    ],
    'cnn_cifar10': [
        {'model': m, 'dataset': 'cifar10', 'pretrained': False}
        for m in ['resnet18']          # placeholder for any lightweight baselines
    ],
    'cnn_gtsrb': [
        {'model': m, 'dataset': 'gtsrb', 'pretrained': False}
        for m in ['resnet18']
    ],
}

TRAIN_SCRIPT = Path(__file__).resolve().parent / 'train_wanet_attack.py'


def build_command(cfg: dict, attack_args: argparse.Namespace) -> list[str]:
    cmd = [
        'python', str(TRAIN_SCRIPT),
        '--model',       cfg['model'],
        '--dataset',     cfg['dataset'],
        '--poison-rate', str(attack_args.poison_rate),
        '--cover-rate',  str(attack_args.cover_rate),
        '--output-dir',  str(Path(attack_args.output_dir).resolve()),
        '--epochs',      str(attack_args.epochs),
        '--batch-size',  str(attack_args.batch_size),
        '--lr',          str(attack_args.lr),
        '--optimizer',   attack_args.optimizer,
        '--scheduler',   attack_args.scheduler,
        '--patience',    str(attack_args.patience),
    ]
    if cfg.get('pretrained'):
        cmd.append('--pretrained')
    if attack_args.gpu >= 0:
        cmd += ['--gpu', str(attack_args.gpu)]
    if not attack_args.tensorboard:
        cmd.append('--no-tensorboard')
    return cmd


def run_experiment(cfg: dict, attack_args: argparse.Namespace, idx: int, total: int) -> bool:
    label = (
        f"[{idx}/{total}] {cfg['model']} | {cfg['dataset']} | "
        f"{'pretrained' if cfg.get('pretrained') else 'scratch'}"
    )
    mode = 'pretrained' if cfg.get('pretrained') else 'scratch'
    bar  = '=' * 80

    print(f"\n{bar}")
    print(f"  STARTING: {label}")
    print(f"  attack=WaNet  poison_rate={attack_args.poison_rate}  cover_rate={attack_args.cover_rate}")
    print(f"{bar}\n")

    cmd     = build_command(cfg, attack_args)
    t_start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent.parent.parent.parent),
    )

    elapsed = time.time() - t_start
    status  = 'SUCCESS' if result.returncode == 0 else f'FAILED (code {result.returncode})'

    print(f"\n{bar}")
    print(f"  FINISHED: {label}")
    print(f"  Status: {status}  |  Time: {elapsed/60:.1f} min")
    print(f"{bar}\n")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Batch WaNet attack training for an experiment group',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--experiment', required=True,
                        choices=list(ATTACK_EXPERIMENTS.keys()),
                        help='Experiment group to run')

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
                        help='Continue with remaining experiments if one fails')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands but do not run them')
    parser.add_argument('--models', nargs='+',
                        help='Restrict to these specific model names')
    parser.add_argument('--datasets', nargs='+', choices=['cifar10', 'gtsrb'],
                        help='Restrict to these datasets')

    args = parser.parse_args()

    experiments = ATTACK_EXPERIMENTS[args.experiment]

    if args.models:
        experiments = [e for e in experiments if e['model'] in args.models]
    if args.datasets:
        experiments = [e for e in experiments if e['dataset'] in args.datasets]

    if not experiments:
        print("No experiments match the given filters. Exiting.")
        sys.exit(0)

    total    = len(experiments)
    failed   = []
    overall  = time.time()

    print(f"\n{'='*80}")
    print(f"  WaNet Batch Attack Training")
    print(f"  experiment   : {args.experiment}")
    print(f"  poison_rate  : {args.poison_rate}")
    print(f"  cover_rate   : {args.cover_rate}")
    print(f"  models       : {total}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("DRY RUN — commands that would be executed:\n")
        for idx, cfg in enumerate(experiments, 1):
            cmd = build_command(cfg, args)
            print(f"  [{idx}/{total}] {' '.join(cmd)}\n")
        return

    for idx, cfg in enumerate(experiments, 1):
        success = run_experiment(cfg, args, idx, total)
        if not success:
            failed.append(f"{cfg['model']}/{cfg['dataset']}/{'pretrained' if cfg.get('pretrained') else 'scratch'}")
            if not args.skip_failed:
                print("Stopping batch due to failure. Use --skip-failed to continue.")
                sys.exit(1)

    total_time = time.time() - overall
    print(f"\n{'='*80}")
    print(f"  BATCH COMPLETE  —  {total - len(failed)}/{total} succeeded")
    print(f"  Total time: {total_time/3600:.2f}h")
    if failed:
        print(f"  Failed:")
        for f in failed:
            print(f"    {f}")
    print(f"{'='*80}\n")

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
