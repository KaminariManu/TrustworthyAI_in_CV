"""
Training script for backdoor attack analysis.
Supports training models from scratch or fine-tuning pre-trained models.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    Fore = type('Fore', (), {'GREEN': '', 'YELLOW': '', 'RED': '', 'CYAN': '', 'BLUE': '', 'MAGENTA': ''})
    Style = type('Style', (), {'RESET_ALL': '', 'BRIGHT': ''})

# Add src to path - go up from scripts/training/clean to project root, then to src
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from models import get_model, count_parameters
from datasets import get_dataloader, get_num_classes


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, max_grad_norm=None, desc='Train'):
    """Train for one epoch with enhanced visualization."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar with custom formatting
    pbar = tqdm(
        dataloader,
        desc=f'{Fore.CYAN}[{epoch:3d}/{total_epochs}] {desc}{Style.RESET_ALL}',
        bar_format='{l_bar}{bar:30}{r_bar}',
        ncols=120,
        leave=True
    )
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosion
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar with colorized metrics
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        
        pbar.set_postfix(
            loss=f'{Fore.YELLOW}{current_loss:.4f}{Style.RESET_ALL}',
            acc=f'{Fore.GREEN}{current_acc:.2f}%{Style.RESET_ALL}',
            refresh=False
        )
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, desc='Val'):
    """Validate model with enhanced visualization."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar with custom formatting
    pbar = tqdm(
        dataloader,
        desc=f'{Fore.MAGENTA}{desc:>15}{Style.RESET_ALL}',
        bar_format='{l_bar}{bar:30}{r_bar}',
        ncols=120,
        leave=False
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
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def train(args):
    """Main training function."""
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}\n")
    
    # Set random seeds
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        import numpy as np
        np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.model}_{args.dataset}_{'pretrained' if args.pretrained else 'scratch'}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Config saved to: {config_path}\n")
    
    # Setup tensorboard
    if args.tensorboard:
        writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    else:
        writer = None
    
    # Get number of classes
    num_classes = get_num_classes(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}\n")
    
    # Create model
    print(f"Creating model: {args.model} (pretrained={args.pretrained})")
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dataset=args.dataset
    )
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Create data loaders
    print("Loading data...")
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        download=True,
        augmentation=args.augmentation
    )
    
    test_loader = get_dataloader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        download=True,
        augmentation=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_step,
            gamma=args.lr_decay_gamma
        )
    elif args.scheduler == 'multistep':
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    # Training loop
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*120}")
    print(f"Starting Training: {args.epochs} epochs")
    if args.patience > 0:
        print(f"Early stopping enabled: patience = {args.patience} epochs")
    print(f"{'='*120}{Style.RESET_ALL}\n")
    
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_start_time = time.time()
    
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, 
            max_grad_norm=args.max_grad_norm, desc='Train'
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, desc='Validation'
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        }
        training_log.append(metrics)
        
        # Print colorized summary
        print(f"\n{Fore.CYAN}{'─'*120}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}Epoch {epoch}/{args.epochs} Summary{Style.RESET_ALL} (Time: {epoch_time:.1f}s)")
        print(f"  {Fore.GREEN}Train{Style.RESET_ALL}      → Loss: {Fore.YELLOW}{train_loss:.4f}{Style.RESET_ALL}  Acc: {Fore.GREEN}{train_acc:.2f}%{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}Validation{Style.RESET_ALL} → Loss: {Fore.YELLOW}{val_loss:.4f}{Style.RESET_ALL}  Acc: {Fore.GREEN}{val_acc:.2f}%{Style.RESET_ALL}")
        print(f"  Learning Rate: {Fore.CYAN}{current_lr:.6f}{Style.RESET_ALL}")
        
        # Tensorboard logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_path = output_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, metrics, best_path)
            print(f"  {Fore.GREEN}{Style.BRIGHT}★ New best model saved! (Acc: {best_acc:.2f}%){Style.RESET_ALL}")
        else:
            epochs_without_improvement += 1
            improvement = val_acc - best_acc
            print(f"  Best so far: {Fore.CYAN}{best_acc:.2f}%{Style.RESET_ALL} (epoch {best_epoch})  Current: {Fore.YELLOW}{improvement:+.2f}%{Style.RESET_ALL}")
            if args.patience > 0:
                print(f"  No improvement: {Fore.RED}{epochs_without_improvement}/{args.patience}{Style.RESET_ALL} epochs")
        
        # Save checkpoint at intervals
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)
            print(f"  {Fore.BLUE}💾 Checkpoint saved{Style.RESET_ALL}")
        
        # Calculate and display ETA
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = elapsed_time / epoch
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_hours = eta_seconds / 3600
        
        if remaining_epochs > 0:
            print(f"  {Fore.CYAN}⏱ ETA: {eta_hours:.1f}h ({avg_epoch_time:.1f}s/epoch){Style.RESET_ALL}")
        
        print()
        
        # Early stopping check
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}{'='*120}")
            print(f"Early stopping triggered: No improvement for {args.patience} epochs")
            print(f"Best validation accuracy: {best_acc:.2f}% (epoch {best_epoch})")
            print(f"{'='*120}{Style.RESET_ALL}\n")
            break
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    save_checkpoint(model, optimizer, args.epochs, metrics, final_path)
    
    # Save training log
    log_path = output_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    
    # Print final summary
    total_time = time.time() - training_start_time
    total_hours = total_time / 3600
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}{'='*120}")
    print(f"{'🎉 TRAINING COMPLETE! 🎉':^120}")
    print(f"{'='*120}{Style.RESET_ALL}\n")
    
    print(f"  {Fore.CYAN}{Style.BRIGHT}Results Summary:{Style.RESET_ALL}")
    print(f"  ├─ Best Val Accuracy:  {Fore.GREEN}{Style.BRIGHT}{best_acc:.2f}%{Style.RESET_ALL} (Epoch {best_epoch})")
    print(f"  ├─ Final Val Accuracy: {Fore.YELLOW}{val_acc:.2f}%{Style.RESET_ALL}")
    print(f"  ├─ Total Training Time: {Fore.CYAN}{total_hours:.2f}h{Style.RESET_ALL} ({total_time/60:.1f}min)")
    print(f"  └─ Average Time/Epoch: {Fore.CYAN}{total_time/args.epochs:.1f}s{Style.RESET_ALL}\n")
    
    print(f"  {Fore.MAGENTA}{Style.BRIGHT}Output Files:{Style.RESET_ALL}")
    print(f"  ├─ Directory: {Fore.BLUE}{output_dir}{Style.RESET_ALL}")
    print(f"  ├─ Best Model: {Fore.GREEN}{best_path.name}{Style.RESET_ALL}")
    print(f"  └─ Training Log: {Fore.YELLOW}{log_path.name}{Style.RESET_ALL}\n")
    
    print(f"{Fore.GREEN}{'='*120}{Style.RESET_ALL}\n")
    
    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train models for backdoor attack analysis')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                                'vit_base', 'vit_small', 'deit_base', 'deit_small'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pre-trained weights from ImageNet')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'gtsrb'],
                        help='Dataset name')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--augmentation', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false',
                        help='Disable data augmentation')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='multistep',
                        choices=['step', 'multistep', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr-decay-step', type=int, default=50,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.1,
                        help='Gamma for learning rate decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement, 0 to disable)')
    parser.add_argument('--max-grad-norm', type=float, default=0.0,
                        help='Max gradient norm for clipping (0 to disable)')
    
    # System arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (-1 for CPU)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (-1 for no seed)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results/models',
                        help='Output directory for models and logs')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='Use tensorboard logging')
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                        help='Disable tensorboard logging')
    
    args = parser.parse_args()
    
    # Adjust hyperparameters for pre-trained models
    if args.pretrained:
        print("\nUsing pre-trained model. Adjusting hyperparameters...")
        if args.lr == 0.1:  # Default LR for training from scratch
            args.lr = 0.01  # Lower LR for fine-tuning
            print(f"  Learning rate adjusted to {args.lr}")
        if args.epochs == 200:  # Default epochs for training from scratch
            args.epochs = 100  # Fewer epochs for fine-tuning
            print(f"  Epochs adjusted to {args.epochs}")
        print()
    
    train(args)


if __name__ == '__main__':
    main()
