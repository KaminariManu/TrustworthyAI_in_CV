"""
Batch attack training script.
Runs train_attack.py for every model in a predefined experiment group.
Mirrors the structure of batch_train.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / 'src'))
from config import get_config, get_experiment, list_experiments


# ── experiment definitions for attack training ────────────────────────────────
ATTACK_EXPERIMENTS = {
    'cifar10_scratch': {
        'dataset': 'cifar10',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'scratch',
        'pretrained': False,
    },
    'cifar10_finetune': {
        'dataset': 'cifar10',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'finetune',
        'pretrained': True,
    },
    'cifar10_vit': {
        'dataset': 'cifar10',
        'models': ['vit_small', 'vit_base', 'deit_small', 'deit_base'],
        'config': 'vit_finetune',
        'pretrained': True,
    },
    'gtsrb_scratch': {
        'dataset': 'gtsrb',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'scratch',
        'pretrained': False,
    },
    'gtsrb_finetune': {
        'dataset': 'gtsrb',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'finetune',
        'pretrained': True,
    },
    'gtsrb_vit': {
        'dataset': 'gtsrb',
        'models': ['vit_small', 'vit_base', 'deit_small', 'deit_base'],
        'config': 'vit_finetune',
        'pretrained': True,
    },
}


def build_command(model, dataset, pretrained, config, attack_args, override_args):
    """Build the train_attack.py command for one model."""
    train_script = Path(__file__).parent / 'train_attack.py'

    cmd = [
        'python', str(train_script),
        '--model', model,
        '--dataset', dataset,
        '--poison-type', attack_args.poison_type,
        '--poison-rate', str(attack_args.poison_rate),
        '--epochs', str(config.get('epochs', 200)),
        '--batch-size', str(config.get('batch_size', 128)),
        '--lr', str(config.get('lr', 0.01)),
        '--optimizer', config.get('optimizer', 'sgd'),
        '--momentum', str(config.get('momentum', 0.9)),
        '--weight-decay', str(config.get('weight_decay', 5e-4)),
        '--scheduler', config.get('scheduler', 'multistep'),
        '--gpu', str(config.get('gpu', 0)),
        '--num-workers', str(config.get('num_workers', 4)),
        '--seed', str(config.get('seed', 42)),
        '--data-dir', config.get('data_dir', './data'),
        '--output-dir', config.get('output_dir', './results/models/attack'),
        '--save-interval', str(config.get('save_interval', 50)),
        '--patience', str(config.get('patience', 20)),
    ]

    if pretrained:
        cmd.append('--pretrained')

    if 'max_grad_norm' in config and config['max_grad_norm'] > 0:
        cmd.extend(['--max-grad-norm', str(config['max_grad_norm'])])

    # Command-line overrides
    if override_args.gpu is not None:
        cmd[cmd.index('--gpu') + 1] = str(override_args.gpu)
    if override_args.data_dir is not None:
        cmd[cmd.index('--data-dir') + 1] = override_args.data_dir
    if override_args.output_dir is not None:
        cmd[cmd.index('--output-dir') + 1] = override_args.output_dir

    return cmd


def run_experiment(experiment_name, attack_args, override_args):
    """Run a complete attack experiment (all models × one dataset)."""
    print('=' * 80)
    print(f"Running attack experiment: {experiment_name}")
    print(f"  poison_type={attack_args.poison_type}  poison_rate={attack_args.poison_rate}")
    print('=' * 80)

    exp = ATTACK_EXPERIMENTS[experiment_name]
    cfg = get_config(exp['config'])
    # attack output subdir includes the attack type
    cfg['output_dir'] = f"./results/models/attack/{attack_args.poison_type}"

    dataset  = exp['dataset']
    models   = exp['models']
    pretrained = exp['pretrained']

    print(f"\nDataset: {dataset}  |  Models: {', '.join(models)}\n")

    results = []
    for i, model in enumerate(models, 1):
        print('\n' + '=' * 80)
        print(f"[{i}/{len(models)}] Training {model} under {attack_args.poison_type}")
        print('=' * 80 + '\n')

        cmd = build_command(model, dataset, pretrained, cfg, attack_args, override_args)

        if override_args.dry_run:
            print('Dry run — would execute:')
            print(' '.join(cmd))
            results.append({'model': model, 'status': 'dry_run'})
            continue

        try:
            subprocess.run(cmd, check=True)
            results.append({'model': model, 'status': 'success'})
            print(f"\n✓ {model} done")
        except subprocess.CalledProcessError as e:
            results.append({'model': model, 'status': 'failed', 'error': str(e)})
            print(f"\n✗ {model} failed: {e}")
            if not override_args.continue_on_error:
                print("Stopping. Use --continue-on-error to skip failures.")
                break

    print('\n' + '=' * 80)
    print(f"Experiment {experiment_name} summary:")
    for r in results:
        sym = '✓' if r['status'] == 'success' else ('~' if r['status'] == 'dry_run' else '✗')
        print(f"  {sym} {r['model']}: {r['status']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch attack training — runs one experiment group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available experiments: {', '.join(ATTACK_EXPERIMENTS)}"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment', choices=list(ATTACK_EXPERIMENTS),
                       help='Predefined experiment group')
    group.add_argument('--models', nargs='+',
                       choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                                'vit_base', 'vit_small', 'deit_base', 'deit_small'],
                       help='Custom model list')

    # Attack parameters
    parser.add_argument('--poison-type', default='badnet',
                        choices=['badnet', 'blend', 'basic', 'trojan', 'SIG', 'WaNet',
                                 'adaptive_blend', 'adaptive_patch', 'TaCT', 'dynamic',
                                 'clean_label', 'ISSBA', 'SleeperAgent'])
    parser.add_argument('--poison-rate', type=float, default=0.1)

    # Custom experiment extras
    parser.add_argument('--dataset', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--config', choices=['scratch', 'finetune', 'vit_scratch', 'vit_finetune', 'test'])

    # Overrides
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--data-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')

    args = parser.parse_args()

    if args.models is not None and (args.dataset is None or args.config is None):
        parser.error('--dataset and --config are required when using --models')

    if args.experiment:
        run_experiment(args.experiment, args, args)
    else:
        # Custom experiment
        exp = {
            'dataset': args.dataset,
            'models': args.models,
            'config': args.config,
            'pretrained': args.config in ('finetune', 'vit_finetune'),
        }
        cfg = get_config(args.config)
        cfg['output_dir'] = f"./results/models/attack/{args.poison_type}"

        results = []
        for i, model in enumerate(args.models, 1):
            cmd = build_command(model, args.dataset, exp['pretrained'], cfg, args, args)
            if args.dry_run:
                print(' '.join(cmd))
                results.append({'model': model, 'status': 'dry_run'})
            else:
                try:
                    subprocess.run(cmd, check=True)
                    results.append({'model': model, 'status': 'success'})
                except subprocess.CalledProcessError as e:
                    results.append({'model': model, 'status': 'failed', 'error': str(e)})
                    if not args.continue_on_error:
                        break

        print('\nCustom experiment summary:')
        for r in results:
            sym = '✓' if r['status'] == 'success' else '✗'
            print(f"  {sym} {r['model']}: {r['status']}")


if __name__ == '__main__':
    main()
