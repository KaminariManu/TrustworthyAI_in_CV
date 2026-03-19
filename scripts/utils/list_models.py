"""
Utility to list and compare trained models.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd


def find_model_directories(base_dir: str) -> List[Path]:
    """Find all model directories in the base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    model_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it has a config.json (marker of model directory)
            if (item / 'config.json').exists():
                model_dirs.append(item)
    
    return sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def load_model_info(model_dir: Path) -> Dict:
    """Load information about a trained model."""
    info = {
        'directory': model_dir.name,
        'path': str(model_dir),
    }
    
    # Load config
    config_path = model_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        info.update({
            'model': config.get('model', 'unknown'),
            'dataset': config.get('dataset', 'unknown'),
            'pretrained': config.get('pretrained', False),
            'epochs': config.get('epochs', 0),
            'lr': config.get('lr', 0),
            'optimizer': config.get('optimizer', 'unknown'),
        })
    
    # Load training log
    log_path = model_dir / 'training_log.json'
    if log_path.exists():
        with open(log_path, 'r') as f:
            log = json.load(f)
        
        if log:
            # Get final metrics
            final = log[-1]
            info['final_train_acc'] = final.get('train_acc', 0)
            info['final_val_acc'] = final.get('val_acc', 0)
            info['final_train_loss'] = final.get('train_loss', 0)
            info['final_val_loss'] = final.get('val_loss', 0)
            
            # Get best metrics
            best_val_acc = max(log, key=lambda x: x.get('val_acc', 0))
            info['best_val_acc'] = best_val_acc.get('val_acc', 0)
            info['best_epoch'] = best_val_acc.get('epoch', 0)
    
    # Check for model files
    info['has_best_model'] = (model_dir / 'best_model.pth').exists()
    info['has_final_model'] = (model_dir / 'final_model.pth').exists()
    
    # Get creation time
    try:
        ctime = model_dir.stat().st_ctime
        info['created'] = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
    except:
        info['created'] = 'unknown'
    
    return info


def print_model_table(models: List[Dict], detailed: bool = False):
    """Print models in a formatted table."""
    if not models:
        print("No models found.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(models)
    
    # Select columns based on detail level
    if detailed:
        columns = ['directory', 'model', 'dataset', 'pretrained', 'epochs', 
                  'best_val_acc', 'final_val_acc', 'optimizer', 'lr', 'created']
    else:
        columns = ['model', 'dataset', 'pretrained', 'best_val_acc', 'final_val_acc', 'created']
    
    # Filter columns that exist
    columns = [col for col in columns if col in df.columns]
    
    if columns:
        display_df = df[columns].copy()
        
        # Format columns
        if 'best_val_acc' in display_df.columns:
            display_df['best_val_acc'] = display_df['best_val_acc'].apply(lambda x: f"{x:.2f}%" if x > 0 else "N/A")
        if 'final_val_acc' in display_df.columns:
            display_df['final_val_acc'] = display_df['final_val_acc'].apply(lambda x: f"{x:.2f}%" if x > 0 else "N/A")
        if 'lr' in display_df.columns:
            display_df['lr'] = display_df['lr'].apply(lambda x: f"{x:.4f}" if x > 0 else "N/A")
        
        print("\n" + "=" * 100)
        print("Trained Models")
        print("=" * 100)
        print(display_df.to_string(index=False))
        print("=" * 100)
        print(f"\nTotal models: {len(models)}")


def filter_models(models: List[Dict], args) -> List[Dict]:
    """Filter models based on command line arguments."""
    filtered = models
    
    if args.model:
        filtered = [m for m in filtered if m.get('model') == args.model]
    
    if args.dataset:
        filtered = [m for m in filtered if m.get('dataset') == args.dataset]
    
    if args.pretrained is not None:
        filtered = [m for m in filtered if m.get('pretrained') == args.pretrained]
    
    return filtered


def compare_models(models: List[Dict], metric: str = 'best_val_acc'):
    """Compare models and show rankings."""
    if not models:
        print("No models to compare.")
        return
    
    # Sort by metric
    sorted_models = sorted(models, key=lambda x: x.get(metric, 0), reverse=True)
    
    print("\n" + "=" * 100)
    print(f"Model Rankings by {metric}")
    print("=" * 100)
    
    for i, model in enumerate(sorted_models, 1):
        model_name = model.get('model', 'unknown')
        dataset = model.get('dataset', 'unknown')
        pretrained = 'pretrained' if model.get('pretrained', False) else 'scratch'
        value = model.get(metric, 0)
        
        print(f"{i:2d}. {model_name:12s} ({dataset:8s}, {pretrained:10s}): {value:.2f}%")
    
    print("=" * 100)


def generate_summary(models: List[Dict]):
    """Generate summary statistics."""
    if not models:
        print("No models to summarize.")
        return
    
    print("\n" + "=" * 100)
    print("Summary Statistics")
    print("=" * 100)
    
    # Group by dataset
    datasets = {}
    for model in models:
        ds = model.get('dataset', 'unknown')
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(model)
    
    for dataset, ds_models in datasets.items():
        print(f"\n{dataset.upper()}:")
        
        # Separate by training mode
        scratch = [m for m in ds_models if not m.get('pretrained', False)]
        pretrained = [m for m in ds_models if m.get('pretrained', False)]
        
        if scratch:
            accs = [m.get('best_val_acc', 0) for m in scratch]
            print(f"  From Scratch ({len(scratch)} models):")
            print(f"    Best: {max(accs):.2f}%")
            print(f"    Mean: {sum(accs)/len(accs):.2f}%")
            print(f"    Worst: {min(accs):.2f}%")
        
        if pretrained:
            accs = [m.get('best_val_acc', 0) for m in pretrained]
            print(f"  Fine-tuned ({len(pretrained)} models):")
            print(f"    Best: {max(accs):.2f}%")
            print(f"    Mean: {sum(accs)/len(accs):.2f}%")
            print(f"    Worst: {min(accs):.2f}%")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='List and compare trained models')
    
    parser.add_argument('--base-dir', type=str, default='./results/models',
                        help='Base directory containing model directories')
    
    # Filters
    parser.add_argument('--model', type=str,
                        choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                                'vit_base', 'vit_small', 'deit_base', 'deit_small'],
                        help='Filter by model architecture')
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'gtsrb'],
                        help='Filter by dataset')
    parser.add_argument('--pretrained', type=bool,
                        help='Filter by pretrained status')
    
    # Display options
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed information')
    parser.add_argument('--compare', action='store_true',
                        help='Show model rankings')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics')
    parser.add_argument('--export', type=str,
                        help='Export results to CSV file')
    
    args = parser.parse_args()
    
    # Find model directories
    model_dirs = find_model_directories(args.base_dir)
    
    if not model_dirs:
        print(f"No models found in {args.base_dir}")
        return
    
    print(f"Found {len(model_dirs)} model directories in {args.base_dir}")
    
    # Load model information
    models = []
    for model_dir in model_dirs:
        info = load_model_info(model_dir)
        models.append(info)
    
    # Filter models
    filtered_models = filter_models(models, args)
    
    if not filtered_models:
        print("No models match the specified filters.")
        return
    
    # Display results
    print_model_table(filtered_models, detailed=args.detailed)
    
    if args.compare:
        compare_models(filtered_models)
    
    if args.summary:
        generate_summary(filtered_models)
    
    # Export to CSV
    if args.export:
        df = pd.DataFrame(filtered_models)
        df.to_csv(args.export, index=False)
        print(f"\nResults exported to: {args.export}")


if __name__ == '__main__':
    main()
