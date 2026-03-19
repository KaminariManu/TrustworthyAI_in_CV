"""
Batch training script for running multiple experiments.
Trains all specified models on specified datasets.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path - go up from scripts/training/clean to project root, then to src
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from config import get_config, get_experiment, list_experiments


def build_train_command(model, dataset, config, args):
    """Build training command for a single model."""
    train_script = Path(__file__).parent / 'train.py'
    
    cmd = [
        sys.executable,  # Python executable
        str(train_script),
        '--model', model,
        '--dataset', dataset,
    ]
    
    # Add configuration parameters
    if config.get('pretrained', False):
        cmd.append('--pretrained')
    
    cmd.extend(['--epochs', str(config.get('epochs', 200))])
    cmd.extend(['--batch-size', str(config.get('batch_size', 128))])
    cmd.extend(['--lr', str(config.get('lr', 0.1))])
    cmd.extend(['--optimizer', config.get('optimizer', 'sgd')])
    cmd.extend(['--momentum', str(config.get('momentum', 0.9))])
    cmd.extend(['--weight-decay', str(config.get('weight_decay', 5e-4))])
    cmd.extend(['--scheduler', config.get('scheduler', 'multistep')])
    cmd.extend(['--gpu', str(config.get('gpu', 0))])
    cmd.extend(['--num-workers', str(config.get('num_workers', 4))])
    cmd.extend(['--seed', str(config.get('seed', 42))])
    cmd.extend(['--data-dir', config.get('data_dir', './data')])
    cmd.extend(['--output-dir', config.get('output_dir', './results/models')])
    cmd.extend(['--save-interval', str(config.get('save_interval', 50))])
    cmd.extend(['--patience', str(config.get('patience', 20))])
    
    # Add gradient clipping if specified
    if 'max_grad_norm' in config and config['max_grad_norm'] > 0:
        cmd.extend(['--max-grad-norm', str(config['max_grad_norm'])])
    
    if not config.get('augmentation', True):
        cmd.append('--no-augmentation')
    
    if not config.get('tensorboard', True):
        cmd.append('--no-tensorboard')
    
    # Override with command line arguments
    if args.gpu is not None:
        cmd[cmd.index('--gpu') + 1] = str(args.gpu)
    
    if args.data_dir is not None:
        cmd[cmd.index('--data-dir') + 1] = args.data_dir
    
    if args.output_dir is not None:
        cmd[cmd.index('--output-dir') + 1] = args.output_dir
    
    return cmd


def run_experiment(experiment_name, args):
    """Run a complete experiment (multiple models on one dataset)."""
    print("=" * 80)
    print(f"Running experiment: {experiment_name}")
    print("=" * 80)
    
    experiment = get_experiment(experiment_name)
    config = get_config(experiment['config'])
    
    dataset = experiment['dataset']
    models = experiment['models']
    
    print(f"\nDataset: {dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Configuration: {experiment['config']}")
    print(f"Number of models to train: {len(models)}\n")
    
    results = []
    
    for i, model in enumerate(models, 1):
        print("\n" + "=" * 80)
        print(f"Training model {i}/{len(models)}: {model}")
        print("=" * 80 + "\n")
        
        cmd = build_train_command(model, dataset, config, args)
        
        if args.dry_run:
            print("Dry run - would execute:")
            print(" ".join(cmd))
            results.append({'model': model, 'status': 'dry_run'})
        else:
            try:
                result = subprocess.run(cmd, check=True)
                results.append({'model': model, 'status': 'success'})
                print(f"\n✓ Successfully trained {model}")
            except subprocess.CalledProcessError as e:
                results.append({'model': model, 'status': 'failed', 'error': str(e)})
                print(f"\n✗ Failed to train {model}: {e}")
                if not args.continue_on_error:
                    print("Stopping due to error. Use --continue-on-error to continue.")
                    break
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Experiment {experiment_name} Summary")
    print("=" * 80)
    for result in results:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_symbol} {result['model']}: {result['status']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch training for backdoor attack analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available experiments:
  {', '.join(list_experiments())}

Examples:
  # Train all CIFAR-10 models from scratch
  python batch_train.py --experiment cifar10_scratch

  # Train all GTSRB models with fine-tuning
  python batch_train.py --experiment gtsrb_finetune

  # Train specific models on CIFAR-10
  python batch_train.py --models vgg16 resnet18 --dataset cifar10 --config scratch

  # Dry run (show commands without executing)
  python batch_train.py --experiment cifar10_scratch --dry-run
        """
    )
    
    # Experiment selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment', type=str,
                       choices=list_experiments(),
                       help='Run predefined experiment')
    group.add_argument('--models', type=str, nargs='+',
                       choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                               'vit_base', 'vit_small', 'deit_base', 'deit_small'],
                       help='Specific models to train')
    
    # Dataset and config (for custom experiments)
    parser.add_argument('--dataset', type=str,
                       choices=['cifar10', 'gtsrb'],
                       help='Dataset (required when using --models)')
    parser.add_argument('--config', type=str,
                       choices=['scratch', 'finetune', 'vit_scratch', 'vit_finetune', 'test'],
                       help='Configuration (required when using --models)')
    
    # Override options
    parser.add_argument('--gpu', type=int,
                       help='GPU id to use (overrides config)')
    parser.add_argument('--data-dir', type=str,
                       help='Data directory (overrides config)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (overrides config)')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without executing')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue training other models if one fails')
    
    args = parser.parse_args()
    
    # Validate custom experiment
    if args.models is not None:
        if args.dataset is None or args.config is None:
            parser.error("--dataset and --config are required when using --models")
    
    # Run experiment
    if args.experiment is not None:
        run_experiment(args.experiment, args)
    else:
        # Custom experiment
        experiment_name = 'custom'
        experiment = {
            'dataset': args.dataset,
            'models': args.models,
            'config': args.config,
        }
        
        print("=" * 80)
        print(f"Running custom experiment")
        print("=" * 80)
        
        config = get_config(args.config)
        
        print(f"\nDataset: {args.dataset}")
        print(f"Models: {', '.join(args.models)}")
        print(f"Configuration: {args.config}")
        print(f"Number of models to train: {len(args.models)}\n")
        
        results = []
        
        for i, model in enumerate(args.models, 1):
            print("\n" + "=" * 80)
            print(f"Training model {i}/{len(args.models)}: {model}")
            print("=" * 80 + "\n")
            
            cmd = build_train_command(model, args.dataset, config, args)
            
            if args.dry_run:
                print("Dry run - would execute:")
                print(" ".join(cmd))
                results.append({'model': model, 'status': 'dry_run'})
            else:
                try:
                    result = subprocess.run(cmd, check=True)
                    results.append({'model': model, 'status': 'success'})
                    print(f"\n✓ Successfully trained {model}")
                except subprocess.CalledProcessError as e:
                    results.append({'model': model, 'status': 'failed', 'error': str(e)})
                    print(f"\n✗ Failed to train {model}: {e}")
                    if not args.continue_on_error:
                        print("Stopping due to error. Use --continue-on-error to continue.")
                        break
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"Custom Experiment Summary")
        print("=" * 80)
        for result in results:
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            print(f"{status_symbol} {result['model']}: {result['status']}")


if __name__ == '__main__':
    main()
