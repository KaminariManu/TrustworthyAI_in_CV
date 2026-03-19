"""
BadNet attack training script.
1. Creates the poisoned dataset via backdoor-toolbox's create_poisoned_set.py (if not already present).
2. Trains any of our model architectures on the poisoned data.
3. Evaluates both Clean Accuracy (CA) and Attack Success Rate (ASR).

Checkpoints are saved to results/models/attack/<poison_type>/<run_dir>/
"""

import os
import sys
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from typing import Any

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    Fore: Any = type('Fore', (), {'GREEN': '', 'YELLOW': '', 'RED': '', 'CYAN': '', 'BLUE': '', 'MAGENTA': ''})
    Style: Any = type('Style', (), {'RESET_ALL': '', 'BRIGHT': ''})

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
TOOLBOX_DIR  = PROJECT_ROOT / 'backdoor-toolbox-main'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(TOOLBOX_DIR))

from models import get_model, count_parameters
from datasets import get_dataloader, get_num_classes

sys.path.insert(0, str(TOOLBOX_DIR / 'utils'))
import importlib
_supervisor = importlib.import_module('supervisor')
_tb_config  = importlib.import_module('config')
# triggers_dir must be absolute — supervisor.get_poison_transform resolves trigger
# PNGs via config.triggers_dir, which defaults to './triggers' (relative to cwd).
setattr(_tb_config, 'triggers_dir', str(TOOLBOX_DIR / 'triggers'))


# ── dataset normalisation stats (must match the toolbox) ──────────────────────
CIFAR10_MEAN  = [0.4914, 0.4822, 0.4465]
CIFAR10_STD   = [0.247,  0.243,  0.261]
GTSRB_MEAN    = [0.3337, 0.3064, 0.3171]
GTSRB_STD     = [0.2672, 0.2564, 0.2629]


def get_toolbox_transforms(dataset: str):
    """Return the same transforms the toolbox uses for training (no aug variant)."""
    if dataset == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    elif dataset == 'gtsrb':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── poisoned-set creation ──────────────────────────────────────────────────────
def get_poison_set_dir(dataset: str, poison_type: str, poison_rate: float) -> Path:
    """Get poison set directory matching the toolbox's naming convention."""
    ratio = f'{poison_rate:.3f}'
    # Match toolbox naming: {poison_type}_{ratio}_poison_seed={seed}
    # Try with poison_seed suffix first (newer format), fallback to without
    poison_seed = 0  # default seed used by toolbox
    dir_core_with_seed = f'{poison_type}_{ratio}_poison_seed={poison_seed}'
    dir_core_no_seed = f'{poison_type}_{ratio}'
    
    # Check which format exists
    path_with_seed = TOOLBOX_DIR / 'poisoned_train_set' / dataset / dir_core_with_seed
    path_no_seed = TOOLBOX_DIR / 'poisoned_train_set' / dataset / dir_core_no_seed
    
    if path_with_seed.exists():
        return path_with_seed
    elif path_no_seed.exists():
        return path_no_seed
    else:
        # Default to the seed format for new creations
        return path_with_seed


def create_poisoned_set_if_needed(args):
    """Creates the poisoned dataset using the toolbox if it does not already exist."""
    poison_dir = get_poison_set_dir(args.dataset, args.poison_type, args.poison_rate)

    imgs_dir = poison_dir / 'imgs'
    labels_file = poison_dir / 'labels'

    if imgs_dir.exists() and labels_file.exists():
        print(f"Poisoned set already exists: {poison_dir}")
        return poison_dir

    print(f"Creating poisoned set at: {poison_dir}")
    cmd = [
        sys.executable,
        str(TOOLBOX_DIR / 'create_poisoned_set.py'),
        '-dataset', args.dataset,
        '-poison_type', args.poison_type,
        '-poison_rate', str(args.poison_rate),
    ]
    result = subprocess.run(cmd, cwd=str(TOOLBOX_DIR), check=True)
    print(f"Poisoned set created successfully.\n")
    return poison_dir


# ── data loading ───────────────────────────────────────────────────────────────
def get_poisoned_train_loader(poison_dir: Path, dataset: str, batch_size: int, num_workers: int):
    """Load poisoned training set from the toolbox's on-disk format."""
    # Import toolbox utilities (must be run after sys.path is set)
    from utils.tools import IMG_Dataset

    imgs_dir = poison_dir / 'imgs'
    labels_file = poison_dir / 'labels'

    # Look for both old (data/) and new (imgs/) dir layouts
    if not imgs_dir.exists():
        imgs_dir = poison_dir / 'data'

    transform = get_toolbox_transforms(dataset)
    poisoned_dataset = IMG_Dataset(
        data_dir=str(imgs_dir),
        label_path=str(labels_file),
        transforms=transform,
    )
    loader = DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ── train / validate ───────────────────────────────────────────────────────────
def train_epoch(model, dataloader, criterion, optimizer, device,
                epoch, total_epochs, max_grad_norm=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f'{Fore.CYAN}[{epoch:3d}/{total_epochs}] Train{Style.RESET_ALL}',
        bar_format='{l_bar}{bar:30}{r_bar}',
        ncols=120,
        leave=True,
    )

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(
            loss=f'{Fore.YELLOW}{running_loss/(batch_idx+1):.4f}{Style.RESET_ALL}',
            acc=f'{Fore.GREEN}{100.*correct/total:.2f}%{Style.RESET_ALL}',
            refresh=False,
        )

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, desc='Val'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f'{Fore.MAGENTA}{desc:>15}{Style.RESET_ALL}',
        bar_format='{l_bar}{bar:30}{r_bar}',
        ncols=120,
        leave=False,
    )

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def validate_asr(model, clean_test_loader, poison_transform, device, target_class, desc='ASR'):
    """Compute Attack Success Rate by applying poison_transform to the clean test set on the fly."""
    model.eval()
    poison_correct = 0
    num_non_target = 0

    pbar = tqdm(
        clean_test_loader,
        desc=f'{Fore.RED}{desc:>15}{Style.RESET_ALL}',
        bar_format='{l_bar}{bar:30}{r_bar}',
        ncols=120,
        leave=False,
    )

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            # Apply trigger on GPU tensor
            p_inputs, p_targets = poison_transform.transform(inputs, targets)
            outputs = model(p_inputs)
            _, predicted = outputs.max(1)
            for i in range(len(targets)):
                if targets[i].item() != target_class:
                    num_non_target += 1
                    if predicted[i].item() == target_class:
                        poison_correct += 1

    if num_non_target == 0:
        return 0.0
    return 100. * poison_correct / num_non_target


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


# ── main training function ─────────────────────────────────────────────────────
def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}\n")

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        import numpy as np
        np.random.seed(args.seed)

    # ── Step 1: create poisoned set ────────────────────────────────────────────
    poison_dir = create_poisoned_set_if_needed(args)

    # ── Step 2: output directory ───────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = (
        f"{args.model}_{args.dataset}"
        f"_{'pretrained' if args.pretrained else 'scratch'}"
        f"_{args.poison_type}_pr{args.poison_rate:.3f}"
        f"_{timestamp}"
    )
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Config saved to: {output_dir / 'config.json'}\n")

    writer = SummaryWriter(log_dir=output_dir / 'tensorboard') if args.tensorboard else None

    # ── Step 3: model ──────────────────────────────────────────────────────────
    num_classes = get_num_classes(args.dataset)
    print(f"Dataset: {args.dataset} ({num_classes} classes)")
    print(f"Attack:  {args.poison_type}  poison_rate={args.poison_rate}\n")

    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dataset=args.dataset,
    )
    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {args.model}  params={total:,}  trainable={trainable:,}\n")

    # ── Step 4: data loaders ───────────────────────────────────────────────────
    print("Loading data...")
    train_loader = get_poisoned_train_loader(
        poison_dir, args.dataset, args.batch_size, args.num_workers
    )
    # Clean test loader (CA)
    clean_test_loader = get_dataloader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        download=False,  # Data already downloaded
        augmentation=False,
    )
    print(f"Train batches (poisoned): {len(train_loader)}")
    print(f"Test batches  (clean):   {len(clean_test_loader)}\n")

    # ── Poison transform for ASR evaluation ───────────────────────────────────
    target_class = _tb_config.target_class[args.dataset]
    try:
        poison_transform = _supervisor.get_poison_transform(
            poison_type=args.poison_type,
            dataset_name=args.dataset,
            target_class=target_class,
            trigger_transform=get_toolbox_transforms(args.dataset),
            is_normalized_input=True,
        )
        asr_available = True
        print(f"ASR evaluation enabled  (target_class={target_class})")
    except Exception as e:
        asr_available = False
        poison_transform = None
        print(f"ASR evaluation disabled: {e}")

    # ── Step 5: optimiser / scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.scheduler == 'multistep':
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        scheduler = None

    # ── Step 6: training loop ──────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'='*120}")
    print(f"Starting Attack Training: {args.epochs} epochs | patience={args.patience}")
    print(f"{'='*120}{Style.RESET_ALL}\n")

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    training_start = time.time()
    training_log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs, max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_acc = validate(model, clean_test_loader, criterion, device, desc='Clean Test (CA)')
        asr = validate_asr(model, clean_test_loader, poison_transform, device, target_class) \
              if asr_available else None

        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
        if scheduler:
            scheduler.step()

        metrics = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 3),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 3),
            'asr': round(asr, 3) if asr is not None else None,
            'lr': current_lr,
        }
        training_log.append(metrics)

        print(f"\n{Fore.CYAN}{'─'*120}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}Epoch {epoch}/{args.epochs}{Style.RESET_ALL} (Time: {epoch_time:.1f}s)")
        print(f"  {Fore.GREEN}Train{Style.RESET_ALL}     → Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  {Fore.MAGENTA}CA (clean){Style.RESET_ALL} → Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")
        if asr is not None:
            print(f"  {Fore.RED}ASR      {Style.RESET_ALL} → {asr:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/clean_test', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/clean_test', val_acc, epoch)
            if asr is not None:
                writer.add_scalar('Accuracy/ASR', asr, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            best_path = output_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, metrics, best_path)
            print(f"  {Fore.GREEN}★ New best model (CA: {best_acc:.2f}%){Style.RESET_ALL}")
        else:
            no_improve += 1
            print(f"  Best so far: {best_acc:.2f}% (epoch {best_epoch})  no-improve: {no_improve}/{args.patience}")

        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, metrics,
                            output_dir / f'checkpoint_epoch_{epoch}.pth')

        elapsed = time.time() - training_start
        eta_h = (elapsed / epoch) * (args.epochs - epoch) / 3600
        if epoch < args.epochs:
            print(f"  ETA: {eta_h:.1f}h")

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\n{Fore.YELLOW}Early stopping: no improvement for {args.patience} epochs.{Style.RESET_ALL}")
            break

    # ── Step 7: save final model and log ──────────────────────────────────────
    save_checkpoint(model, optimizer, epoch, metrics, output_dir / 'final_model.pth')
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=4)

    total_time = time.time() - training_start
    print(f"\n{Fore.GREEN}{'='*120}")
    print(f"ATTACK TRAINING COMPLETE")
    best_asr = max((m['asr'] for m in training_log if m.get('asr') is not None), default=None)
    print(f"  Best CA:      {best_acc:.2f}% (epoch {best_epoch})")
    if best_asr is not None:
        print(f"  Best ASR:     {best_asr:.2f}%")
    print(f"  Poison type:  {args.poison_type}  rate={args.poison_rate}")
    print(f"  Total time:   {total_time/3600:.2f}h")
    print(f"  Output dir:   {output_dir}")
    print(f"{'='*120}{Style.RESET_ALL}\n")

    if writer:
        writer.close()


# ── argument parser ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Train model under BadNet (or other) poisoning attack',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument('--model', required=True,
                        choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                                 'vit_base', 'vit_small', 'deit_base', 'deit_small'])
    parser.add_argument('--pretrained', action='store_true')

    # Dataset
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'gtsrb'])
    parser.add_argument('--data-dir', default='./data')

    # Attack
    parser.add_argument('--poison-type', default='badnet',
                        choices=['badnet', 'blend', 'basic', 'trojan', 'SIG', 'WaNet',
                                 'adaptive_blend', 'adaptive_patch', 'TaCT', 'dynamic',
                                 'clean_label', 'ISSBA', 'SleeperAgent'])
    parser.add_argument('--poison-rate', type=float, default=0.1,
                        help='Fraction of training samples to poison')

    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--scheduler', default='multistep',
                        choices=['step', 'multistep', 'cosine', 'none'])
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--max-grad-norm', type=float, default=5.0)

    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--output-dir', default='./results/models/attack')
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--tensorboard', action='store_true', default=True)
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false')

    args = parser.parse_args()

    # Adjust for pre-trained
    if args.pretrained:
        print("\nPre-trained model: adjusting hyperparameters...")
        if args.lr == 0.01:
            args.lr = 0.001
        if args.epochs == 200:
            args.epochs = 100
        print(f"  lr={args.lr}  epochs={args.epochs}\n")

    train(args)


if __name__ == '__main__':
    main()
