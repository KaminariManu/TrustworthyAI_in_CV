"""
Evaluate a trained model on test set.
Optionally measures Attack Success Rate (ASR) for backdoor attack checkpoints.
"""

import os
import sys
import argparse
import json
import importlib
from types import ModuleType
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLBOX_DIR  = PROJECT_ROOT / 'backdoor-toolbox-main'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(TOOLBOX_DIR))
sys.path.insert(0, str(TOOLBOX_DIR / 'utils'))

from models import get_model, count_parameters
from datasets import get_dataloader, get_num_classes, get_class_names
from evaluation import evaluate_model, load_checkpoint, compute_per_class_accuracy, print_classification_report

# Toolbox modules — imported lazily to avoid hard dependency
_supervisor = None
_tb_config  = None

def _load_toolbox() -> tuple[ModuleType, ModuleType]:
    global _supervisor, _tb_config
    if _supervisor is None:
        _supervisor = importlib.import_module('supervisor')
        _tb_config  = importlib.import_module('config')
        # triggers_dir must be absolute — see train_attack.py for explanation
        setattr(_tb_config, 'triggers_dir', str(TOOLBOX_DIR / 'triggers'))
    return _supervisor, _tb_config  # type: ignore[return-value]


_STATS = {
    'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247,  0.243,  0.261],  'size': 32},
    'gtsrb':   {'mean': [0.3337, 0.3064, 0.3171], 'std': [0.2672, 0.2564, 0.2629], 'size': 32},
}

def _get_trigger_transform(dataset: str):
    s = _STATS[dataset]
    return transforms.Compose([
        transforms.Resize((s['size'], s['size'])),
        transforms.ToTensor(),
        transforms.Normalize(s['mean'], s['std']),
    ])

def validate_asr(model, clean_test_loader, poison_transform, device, target_class) -> float:
    """Fraction of non-target test images classified as target_class after trigger injection."""
    model.eval()
    n_correct = 0
    n_total   = 0
    with torch.no_grad():
        for images, labels in clean_test_loader:
            mask = labels != target_class
            if mask.sum() == 0:
                continue
            images, labels = images[mask], labels[mask]
            poisoned, _ = poison_transform.transform(images.clone(), labels.clone())
            poisoned = poisoned.to(device)
            preds = model(poisoned).argmax(dim=1).cpu()
            n_correct += (preds == target_class).sum().item()
            n_total   += len(preds)
    return 100.0 * n_correct / n_total if n_total > 0 else 0.0


def evaluate(args):
    """Evaluate a trained model."""
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load config if available
    model_dir = Path(args.checkpoint).parent
    config_path = model_dir / 'config.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config['model']
        dataset = config['dataset']
        print(f"Loaded config from: {config_path}")
    else:
        # Use command line arguments
        if args.model is None or args.dataset is None:
            raise ValueError("--model and --dataset are required when config.json is not available")
        model_name = args.model
        dataset = args.dataset
        print("Config file not found, using command line arguments")
    
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}\n")
    
    # Get number of classes
    num_classes = get_num_classes(dataset)
    
    # Create model
    print("Creating model...")
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,  # We're loading weights from checkpoint
        dataset=dataset
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    epoch, metrics = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    
    if metrics:
        print(f"Checkpoint from epoch {epoch}")
        if 'val_acc' in metrics:
            print(f"Validation accuracy at checkpoint: {metrics['val_acc']:.2f}%")
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Load data
    print("Loading data...")
    
    if args.train:
        dataloader = get_dataloader(
            dataset_name=dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train=True,
            num_workers=args.num_workers,
            download=False,
            augmentation=False  # No augmentation for evaluation
        )
        split_name = "Train"
    else:
        dataloader = get_dataloader(
            dataset_name=dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train=False,
            num_workers=args.num_workers,
            download=False,
            augmentation=False
        )
        split_name = "Test"
    
    print(f"{split_name} batches: {len(dataloader)}\n")
    
    # Evaluate
    print(f"Evaluating on {split_name} set...")
    criterion = nn.CrossEntropyLoss()
    
    results = evaluate_model(
        model,
        dataloader,
        device,
        criterion=criterion,
        return_predictions=args.detailed
    )
    
    # Print results
    print("\n" + "=" * 80)
    print(f"{split_name} Set Results")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"Loss: {results['loss']:.4f}")
    
    # Detailed analysis
    if args.detailed:
        print("\n" + "=" * 80)
        print("Detailed Analysis")
        print("=" * 80)
        
        # Per-class accuracy
        per_class_acc = compute_per_class_accuracy(
            results['predictions'],
            results['targets'],
            num_classes=num_classes
        )
        
        class_names = get_class_names(dataset)
        
        print("\nPer-class Accuracy:")
        print("-" * 80)
        for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
            if dataset == 'cifar10':
                print(f"Class {i:2d} ({name:15s}): {acc:6.2f}%")
            else:
                # GTSRB has longer names
                print(f"Class {i:2d}: {acc:6.2f}% - {name}")
        
        print(f"\nMean per-class accuracy: {per_class_acc.mean():.2f}%")
        print(f"Std per-class accuracy: {per_class_acc.std():.2f}%")
        
        # Classification report
        if args.show_report:
            print_classification_report(
                results['predictions'],
                results['targets'],
                class_names=class_names if dataset == 'cifar10' else None  # GTSRB names too long
            )
    
    # ── ASR evaluation (attack checkpoints only) ──────────────────────────────
    asr = None
    if getattr(args, 'poison_type', None):
        print("\n" + "=" * 80)
        print("Attack Success Rate (ASR)")
        print("=" * 80)
        try:
            supervisor, tb_config = _load_toolbox()
            target_class = tb_config.target_class[dataset]

            if args.poison_type == 'WaNet':
                # WaNet's get_poison_transform calls get_poison_set_dir(args) internally
                # which returns a RELATIVE path; torch.load resolves it against cwd.
                # Must chdir to TOOLBOX_DIR so the path resolves correctly.
                import argparse as _ap
                _tb_args = _ap.Namespace(
                    poison_type='WaNet',
                    poison_rate=args.poison_rate,
                    cover_rate=args.cover_rate,
                    dataset=dataset,
                    trigger='none',
                    alpha=0.2,
                )
                _orig_dir = os.getcwd()
                try:
                    os.chdir(str(TOOLBOX_DIR))
                    poison_transform = supervisor.get_poison_transform(
                        poison_type='WaNet',
                        dataset_name=dataset,
                        target_class=target_class,
                        trigger_transform=_get_trigger_transform(dataset),
                        is_normalized_input=True,
                        args=_tb_args,
                    )
                finally:
                    os.chdir(_orig_dir)
            else:
                poison_transform = supervisor.get_poison_transform(
                    poison_type=args.poison_type,
                    dataset_name=dataset,
                    target_class=target_class,
                    trigger_transform=_get_trigger_transform(dataset),
                    is_normalized_input=True,
                )

            # ASR always uses the clean test set (even if --train was passed)
            asr_loader = dataloader if not args.train else get_dataloader(
                dataset_name=dataset, data_dir=args.data_dir,
                batch_size=args.batch_size, train=False,
                num_workers=args.num_workers, download=False, augmentation=False,
            )
            asr = validate_asr(model, asr_loader, poison_transform, device, target_class)
            print(f"Poison type  : {args.poison_type}")
            print(f"Poison rate  : {args.poison_rate}")
            if args.poison_type == 'WaNet':
                print(f"Cover rate   : {args.cover_rate}")
            print(f"Target class : {target_class}")
            print(f"ASR          : {asr:.2f}%")
        except Exception as e:
            print(f"ASR evaluation failed: {e}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_results = {
            'model': model_name,
            'dataset': dataset,
            'checkpoint': str(args.checkpoint),
            'split': split_name.lower(),
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'loss': results['loss'],
        }

        if asr is not None:
            save_results['attack_success_rate'] = round(asr, 3)
            save_results['poison_type'] = args.poison_type
            save_results['poison_rate'] = args.poison_rate

        if args.detailed:
            save_results['per_class_accuracy'] = per_class_acc.tolist()
            save_results['mean_per_class_accuracy'] = per_class_acc.mean()
            save_results['std_per_class_accuracy'] = per_class_acc.std()
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=4)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str,
                        choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                                'vit_base', 'vit_small', 'deit_base', 'deit_small'],
                        help='Model architecture (auto-detected from config if available)')
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'gtsrb'],
                        help='Dataset (auto-detected from config if available)')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--train', action='store_true',
                        help='Evaluate on training set instead of test set')
    
    # Evaluation options
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed per-class analysis')
    parser.add_argument('--show-report', action='store_true',
                        help='Show full classification report (with --detailed)')
    
    # System
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output', type=str,
                        help='Path to save results JSON')

    # Attack / ASR (optional — only for attack checkpoints)
    parser.add_argument('--poison-type', type=str, default=None,
                        help='Poison type to measure ASR (e.g. badnet, blend, WaNet). '
                             'If omitted, ASR is not computed.')
    parser.add_argument('--poison-rate', type=float, default=0.1,
                        help='Poison rate used during training (default: 0.1)')
    parser.add_argument('--cover-rate', type=float, default=0.1,
                        help='Cover rate used during WaNet training (default: 0.1)')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()
