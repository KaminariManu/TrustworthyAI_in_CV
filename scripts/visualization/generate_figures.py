"""
Generate visualizations from backdoor ATTACK results
Creates plots, charts including ASR (Attack Success Rate) metrics for attack scenarios.
For baseline clean models without attacks, use generate_baseline_figures.py instead.
"""
# pyright: reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trigger visualisation helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_attack_trigger_figures import make_trigger_figure

# Import visualization utilities
from utils.visualization_utils import (
    plot_metric_heatmap,
    plot_scatter_with_annotations,
    plot_training_history,
    plot_confusion_matrix as plot_confusion_matrix_util,
    set_plot_style
)

# Set style
set_plot_style('whitegrid', font_scale=1.0)
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def _is_vit_deit_scratch(row):
    """Return True for scratch-trained ViT/DeiT entries."""
    model = str(row.get('model', '')).lower()
    if not (model.startswith('vit') or model.startswith('deit')):
        return False

    pretrained = row.get('pretrained')
    if pretrained is False:
        return True

    training_style = str(row.get('training_style', '')).lower()
    return training_style in {'scratch', 'vit_scratch'}


def filter_vit_deit_scratch(results):
    """Drop ViT/DeiT scratch rows to keep scope aligned with baseline."""
    return [r for r in results if not _is_vit_deit_scratch(r)]


def plot_accuracy_comparison(results, save_path, attack_name="Backdoor"):
    """
    Plot Clean Accuracy comparison across models and datasets.
    Separates pretrained vs scratch-trained models when 'model_display' is available.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Group by dataset
    datasets = df['dataset'].unique()
    models = df[model_col].unique()
    x = np.arange(len(models))
    width = 0.35
    
    for i, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset].set_index(model_col).reindex(models).reset_index()
        offset = width * (i - len(datasets)/2 + 0.5)
        bars = ax.bar(x + offset, data['clean_accuracy'], width, 
               label=dataset.upper(), alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=14)
    
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    ax.set_ylabel('Clean Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_title(f'Clean Accuracy Comparison: {attack_name} Attack', fontsize=20, fontweight='bold', pad=40)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_asr_comparison(results, save_path, attack_name="Backdoor"):
    """
    Plot Attack Success Rate comparison across models and datasets.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    
    model_col = 'model_display' if 'model_display' in df.columns else 'model'

    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Group by dataset
    datasets = df['dataset'].unique()
    models = df[model_col].unique()
    x = np.arange(len(models))
    width = 0.35
    
    for i, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset].set_index(model_col).reindex(models).reset_index()
        offset = width * (i - len(datasets)/2 + 0.5)
        bars = ax.bar(x + offset, data['attack_success_rate'], width,
               label=dataset.upper(), alpha=0.8)
               
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=14)
    
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=18, fontweight='bold')
    ax.set_title(f'Attack Success Rate: {attack_name} Attack', fontsize=20, fontweight='bold', pad=40)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_heatmap_clean_accuracy(results, save_path, attack_name="Backdoor"):
    """
    Plot heatmap of Clean Accuracy using visualization utils.
    Separates pretrained vs scratch-trained models when 'model_display' is available.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    model_col = 'model_display' if 'model_display' in df.columns else 'model'

    plot_metric_heatmap(
        results_df=df,
        metric='clean_accuracy',
        row_var=model_col,
        col_var='dataset',
        save_path=save_path,
        title=f'Clean Accuracy Heatmap: {attack_name} Attack',
        cmap='YlGn',
        show=False
    )


def plot_heatmap_asr(results, save_path, attack_name="Backdoor"):
    """
    Plot heatmap of Attack Success Rate using visualization utils.
    Separates pretrained vs scratch-trained models when 'model_display' is available.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    model_col = 'model_display' if 'model_display' in df.columns else 'model'

    plot_metric_heatmap(
        results_df=df,
        metric='attack_success_rate',
        row_var=model_col,
        col_var='dataset',
        save_path=save_path,
        title=f'Attack Success Rate Heatmap: {attack_name} Attack',
        cmap='YlOrRd',
        show=False
    )


def plot_scatter_clean_vs_asr(results, save_path, attack_name="Backdoor"):
    """
    Scatter plot: Clean Accuracy vs ASR using visualization utils.
    Separates pretrained vs scratch-trained models when 'model_display' is available.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    model_col = 'model_display' if 'model_display' in df.columns else 'model'

    plot_scatter_with_annotations(
        results_df=df,
        x_metric='clean_accuracy',
        y_metric='attack_success_rate',
        label_col=model_col,
        hue_col='dataset',
        save_path=save_path,
        title=f'Clean Accuracy vs Attack Success Rate: {attack_name}',
        show=False
    )


def plot_pretrained_vs_scratch(results, save_path, attack_name="Backdoor"):
    """
    Compare pretrained vs scratch-trained models for both Clean Accuracy and ASR.
    Only generated when 'pretrained' field is present in results.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)

    if 'pretrained' not in df.columns:
        print("  Skipping pretrained comparison (not available)")
        return

    for col in ('clean_accuracy', 'attack_success_rate'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel, title_suffix in [
        (axes[0], 'clean_accuracy', 'Clean Accuracy (%)', 'Clean Accuracy'),
        (axes[1], 'attack_success_rate', 'Attack Success Rate (%)', 'ASR'),
    ]:
        pretrained_means, scratch_means = [], []
        for dataset in datasets:
            pt = df[(df['dataset'] == dataset) & (df['pretrained'] == True)][metric].mean()
            sc = df[(df['dataset'] == dataset) & (df['pretrained'] == False)][metric].mean()
            pretrained_means.append(pt if not pd.isna(pt) else 0)
            scratch_means.append(sc if not pd.isna(sc) else 0)

        bars_pt = ax.bar(x - width/2, pretrained_means, width, label='Pretrained', alpha=0.8)
        bars_sc = ax.bar(x + width/2, scratch_means, width, label='Scratch', alpha=0.8)

        ax.set_xlabel('Dataset', fontsize=18, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
        ax.set_title(f'{title_suffix}: Pretrained vs Scratch', fontsize=20, fontweight='bold', pad=40)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets], fontsize=16)
        ax.legend(fontsize=14)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        for i, (pt, sc) in enumerate(zip(pretrained_means, scratch_means)):
            if pt > 0:
                ax.text(i - width/2, pt + 1, f'{pt:.1f}%', ha='center', va='bottom', fontsize=14)
            if sc > 0:
                ax.text(i + width/2, sc + 1, f'{sc:.1f}%', ha='center', va='bottom', fontsize=14)

    plt.suptitle(f'{attack_name} Attack: Pretrained vs Scratch Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_accuracy_by_training_style(results, save_path, attack_name="Backdoor"):
    """
    Compare clean accuracy and ASR across different training styles.
    Only generated when 'training_style' field is present in results.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)

    if 'training_style' not in df.columns:
        print("  Skipping training style comparison (not available)")
        return

    for col in ('clean_accuracy', 'attack_success_rate'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    training_styles = df['training_style'].unique()
    datasets = df['dataset'].unique()
    x = np.arange(len(training_styles))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel, title_suffix in [
        (axes[0], 'clean_accuracy', 'Average Clean Accuracy (%)', 'Clean Accuracy'),
        (axes[1], 'attack_success_rate', 'Average ASR (%)', 'ASR'),
    ]:
        for i, dataset in enumerate(datasets):
            means = [
                df[(df['dataset'] == dataset) & (df['training_style'] == style)][metric].mean()
                for style in training_styles
            ]
            means = [m if not pd.isna(m) else 0 for m in means]
            offset = width * (i - len(datasets)/2 + 0.5)
            ax.bar(x + offset, means, width, label=dataset.upper(), alpha=0.8)

        ax.set_xlabel('Training Style')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_suffix} by Training Style', pad=40)
        ax.set_xticks(x)
        ax.set_xticklabels(training_styles, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle(f'{attack_name} Attack: Performance by Training Style',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_model_comparison_grouped(results, save_path, attack_name="Backdoor"):
    """
    Grouped line chart comparing Clean Acc and ASR for each model.
    Separates pretrained vs scratch-trained models when 'model_display' is available.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    
    # Group by model
    models = df[model_col].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Clean Accuracy
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].set_index(model_col).reindex(models).reset_index()
        x = np.arange(len(models))
        axes[0].plot(x, data['clean_accuracy'].values, 
                    marker='o', linewidth=2.5, markersize=10,
                    label=dataset.upper())
    
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Clean Accuracy (%)')
    axes[0].set_title('Clean Accuracy by Model', pad=40)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 100)
    
    # ASR
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].set_index(model_col).reindex(models).reset_index()
        x = np.arange(len(models))
        axes[1].plot(x, data['attack_success_rate'].values,
                    marker='s', linewidth=2.5, markersize=10,
                    label=dataset.upper())
    
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Attack Success Rate (%)')
    axes[1].set_title('Attack Success Rate by Model', pad=40)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 100)
    
    plt.suptitle(f'{attack_name} Attack: Model Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_trade_off_analysis(results, save_path, attack_name="Backdoor"):
    """
    Plot trade-off between clean accuracy and attack success rate.
    Shows the degradation in clean accuracy vs achieved ASR.
    Separates pretrained vs scratch-trained models when 'model_display' is available.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate accuracy degradation if baseline is available
    if 'baseline_accuracy' in df.columns:
        df['accuracy_drop'] = df['baseline_accuracy'] - df['clean_accuracy']
    else:
        df['accuracy_drop'] = 100 - df['clean_accuracy']  # Assume perfect baseline
    
    # Plot
    colors = sns.color_palette("Set2", n_colors=len(df['dataset'].unique()))
    for idx, dataset in enumerate(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        ax.scatter(data['attack_success_rate'], data['accuracy_drop'],
                  label=dataset.upper(), s=150, alpha=0.7,
                  color=colors[idx], edgecolors='black', linewidth=1.5)
        
        # Add model labels
        for _, row in data.iterrows():
            ax.annotate(row[model_col], 
                       (row['attack_success_rate'], row['accuracy_drop']),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Attack Success Rate (%)')
    ax.set_ylabel('Clean Accuracy Drop (%)')
    ax.set_title(f'Attack-Defense Trade-off: {attack_name}', pad=40)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(results, save_path, attack_name="Backdoor"):
    """
    Plot training curves if available in results.
    
    Args:
        results: List of experiment results (must contain 'training_history')
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    # Check if training history is available
    if not results or 'training_history' not in results[0]:
        print("  Skipping training curves (no history available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for result in results[:4]:  # Plot first 4 experiments
        history = result['training_history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        idx = results.index(result)
        ax_row = idx // 2
        ax_col = idx % 2
        
        ax1 = axes[ax_row, ax_col]
        ax2 = ax1.twinx()
        
        # Loss
        line1 = ax1.plot(epochs, history['train_loss'], 'b-', 
                        label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            line2 = ax1.plot(epochs, history['val_loss'], 'b--',
                           label='Val Loss', linewidth=2)
        
        # Accuracy
        line3 = ax2.plot(epochs, history['train_acc'], 'r-',
                        label='Train Acc', linewidth=2)
        if 'val_acc' in history:
            line4 = ax2.plot(epochs, history['val_acc'], 'r--',
                           label='Val Acc', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax2.set_ylabel('Accuracy (%)', color='r')
        ax1.set_title(f"{result['model']} - {result['dataset']}")
        
        # Combine legends
        lines = line1
        if 'val_loss' in history:
            lines += line2
        lines += line3
        if 'val_acc' in history:
            lines += line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        ax1.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves: {attack_name} Attack',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(results, save_path, attack_name="Backdoor"):
    """
    Plot confusion matrices for attack predictions if available.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    # Check if confusion matrix is available
    if not results or 'confusion_matrix' not in results[0]:
        print("  Skipping confusion matrix (not available)")
        return
    
    n_results = min(4, len(results))
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    for idx, result in enumerate(results[:n_results]):
        cm = np.array(result['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   ax=axes[idx], cbar_kws={'label': 'Count'})
        axes[idx].set_title(f"{result['model']} - {result['dataset']}")
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    
    plt.suptitle(f'Confusion Matrices: {attack_name} Attack',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_poisoning_rate_analysis(results, save_path, attack_name="Backdoor"):
    """
    Plot impact of poisoning rate on attack performance.
    
    Args:
        results: List of experiment results (must contain 'poisoning_rate')
        save_path: Path to save the figure
        attack_name: Name of the attack for the title
    """
    df = pd.DataFrame(results)
    
    if 'poisoning_rate' not in df.columns:
        print("  Skipping poisoning rate analysis (not available)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group by model and dataset
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            data = df[(df['model'] == model) & (df['dataset'] == dataset)]
            if len(data) > 1:
                data = data.sort_values('poisoning_rate')
                label = f"{model} - {dataset.upper()}"
                
                axes[0].plot(data['poisoning_rate'], data['clean_accuracy'],
                           marker='o', linewidth=2, markersize=8, label=label)
                axes[1].plot(data['poisoning_rate'], data['attack_success_rate'],
                           marker='s', linewidth=2, markersize=8, label=label)
    
    axes[0].set_xlabel('Poisoning Rate (%)')
    axes[0].set_ylabel('Clean Accuracy (%)')
    axes[0].set_title('Clean Accuracy vs Poisoning Rate', pad=40)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Poisoning Rate (%)')
    axes[1].set_ylabel('Attack Success Rate (%)')
    axes[1].set_title('ASR vs Poisoning Rate', pad=40)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Poisoning Rate Impact: {attack_name} Attack',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_visualizations(results_path, save_dir='./results/figures/attack',
                                attack_name="Backdoor", data_dir="./data",
                                poison_type=None, datasets=None):
    """
    Generate all visualizations from backdoor ATTACK results.
    Includes ASR (Attack Success Rate) metrics and attack-specific analyses.
    
    NOTE: For baseline clean models without attacks, use generate_baseline_figures.py
    
    Args:
        results_path: Path to JSON results file with attack metrics
        save_dir: Directory to save figures
        attack_name: Name of the attack (for titles and filenames)
    """
    # Infer poison_type from attack_name if not explicitly provided
    if poison_type is None:
        poison_type = attack_name.lower().replace('-', '')
    print("="*70)
    print(f"Generating Attack Visualizations from {attack_name} Attack Results")
    print("="*70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)
    print(f"Loaded {len(results)} experiment results")

    # Keep attack outputs aligned with baseline scope.
    before_filter = len(results)
    results = filter_vit_deit_scratch(results)
    removed = before_filter - len(results)
    if removed:
        print(f"Filtered out {removed} ViT/DeiT scratch entries to match baseline scope")

    # Coerce None ASR to NaN and aggregate duplicates: keep best clean_accuracy
    # per (model, dataset, pretrained, training_style) so bar charts have one bar per group.
    # The pretrained/training_style keys are preserved to enable the pretrained-vs-scratch split.
    import pandas as _pd
    _df = _pd.DataFrame(results)
    for _col in ('clean_accuracy', 'attack_success_rate', 'best_train_acc', 'best_val_loss'):
        if _col in _df.columns:
            _df[_col] = _pd.to_numeric(_df[_col], errors='coerce')

    _groupby_keys = ['model', 'dataset']
    if 'pretrained' in _df.columns:
        _groupby_keys.append('pretrained')
    if 'training_style' in _df.columns:
        _groupby_keys.append('training_style')

    _df = (
        _df.groupby(_groupby_keys, as_index=False)
           .agg({
               'clean_accuracy':      'max',
               'attack_success_rate': 'max',
               **{c: 'first' for c in _df.columns
                  if c not in _groupby_keys + ['clean_accuracy', 'attack_success_rate']}
           })
    )

    # Create display names that include pretrained/scratch suffix
    if 'pretrained' in _df.columns:
        _df['model_display'] = _df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        print(f"  Pretrained entries: {_df['pretrained'].sum()}, "
              f"Scratch entries: {(~_df['pretrained']).sum()}")
        print("  Figures will show '<model>_pretrained' and '<model>_scratch' separately")

    results = _df.to_dict(orient='records')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    attack_prefix = attack_name.lower().replace(' ', '_')
    
    print("\nGenerating plots...")
    
    # 1. Clean Accuracy Comparison
    print("  1. Clean Accuracy Bar Chart...")
    plot_accuracy_comparison(
        results,
        os.path.join(save_dir, f'{attack_prefix}_clean_accuracy_comparison_{timestamp}.png'),
        attack_name
    )
    
    # 2. ASR Comparison
    print("  2. ASR Bar Chart...")
    plot_asr_comparison(
        results,
        os.path.join(save_dir, f'{attack_prefix}_asr_comparison_{timestamp}.png'),
        attack_name
    )
    
    # 3. Clean Accuracy Heatmap
    print("  3. Clean Accuracy Heatmap...")
    plot_heatmap_clean_accuracy(
        results,
        os.path.join(save_dir, f'{attack_prefix}_clean_accuracy_heatmap_{timestamp}.png'),
        attack_name
    )
    
    # 4. ASR Heatmap
    print("  4. ASR Heatmap...")
    plot_heatmap_asr(
        results,
        os.path.join(save_dir, f'{attack_prefix}_asr_heatmap_{timestamp}.png'),
        attack_name
    )
    
    # 5. Scatter Plot
    print("  5. Clean Acc vs ASR Scatter Plot...")
    plot_scatter_clean_vs_asr(
        results,
        os.path.join(save_dir, f'{attack_prefix}_clean_vs_asr_scatter_{timestamp}.png'),
        attack_name
    )
    
    # 6. Model Comparison
    print("  6. Model Comparison Line Charts...")
    plot_model_comparison_grouped(
        results,
        os.path.join(save_dir, f'{attack_prefix}_model_comparison_{timestamp}.png'),
        attack_name
    )
    
    # 7. Trade-off Analysis
    print("  7. Attack-Defense Trade-off...")
    plot_trade_off_analysis(
        results,
        os.path.join(save_dir, f'{attack_prefix}_tradeoff_analysis_{timestamp}.png'),
        attack_name
    )
    
    # 8. Training Curves (if available)
    print("  8. Training Curves...")
    plot_training_curves(
        results,
        os.path.join(save_dir, f'{attack_prefix}_training_curves_{timestamp}.png'),
        attack_name
    )
    
    # 9. Confusion Matrix (if available)
    print("  9. Confusion Matrices...")
    plot_confusion_matrix(
        results,
        os.path.join(save_dir, f'{attack_prefix}_confusion_matrices_{timestamp}.png'),
        attack_name
    )
    
    # 10. Poisoning Rate Analysis (if available)
    print("  10. Poisoning Rate Analysis...")
    plot_poisoning_rate_analysis(
        results,
        os.path.join(save_dir, f'{attack_prefix}_poisoning_rate_{timestamp}.png'),
        attack_name
    )

    # 11. Pretrained vs Scratch Comparison
    print("  11. Pretrained vs Scratch Comparison...")
    plot_pretrained_vs_scratch(
        results,
        os.path.join(save_dir, f'{attack_prefix}_pretrained_vs_scratch_{timestamp}.png'),
        attack_name
    )

    # 12. Training Style Comparison
    print("  12. Training Style Comparison...")
    plot_accuracy_by_training_style(
        results,
        os.path.join(save_dir, f'{attack_prefix}_training_style_comparison_{timestamp}.png'),
        attack_name
    )

    # 13. Trigger examples — clean vs poisoned image grids
    print("  13. Trigger Examples (clean vs poisoned)...")
    trigger_dir = Path(save_dir) / 'attack_triggers'
    _datasets = datasets if datasets else ['cifar10', 'gtsrb']
    # Extract poison_rate / cover_rate from results for accurate subtitle
    _first = results[0] if results else {}
    _pr = _first.get('poison_rate', 0.1)
    _cr = _first.get('cover_rate', 0.1)
    for ds in _datasets:
        try:
            make_trigger_figure(
                dataset=ds,
                poison_type=poison_type,
                n_samples=8,
                data_dir=data_dir,
                output_dir=trigger_dir,
                poison_rate=_pr,
                cover_rate=_cr,
            )
        except Exception as e:
            print(f"    Warning: could not generate trigger figure for {ds}: {e}")
    
    print("\n" + "="*70)
    print("All visualizations generated successfully!")
    print(f"Saved to: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate visualizations from backdoor ATTACK results (includes ASR metrics)'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to attack results JSON file')
    parser.add_argument('--save-dir', type=str, default='./results/figures/attack',
                       help='Directory to save figures (use results/figures/attack/<type> for specific attacks)')
    parser.add_argument('--attack-name', type=str, default='Backdoor',
                       help='Name of the attack for titles and filenames')
    parser.add_argument('--poison-type', type=str, default=None,
                       help='Poison type for trigger figure (e.g. badnet, WaNet, blend). Inferred from --attack-name if omitted.')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Root data directory for trigger figure generation')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'gtsrb'],
                       choices=['cifar10', 'gtsrb'],
                       help='Datasets to include in trigger examples figure')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        exit(1)
    
    # Generate visualizations
    generate_all_visualizations(
        args.results, args.save_dir, args.attack_name,
        data_dir=args.data_dir,
        poison_type=args.poison_type,
        datasets=args.datasets,
    )
