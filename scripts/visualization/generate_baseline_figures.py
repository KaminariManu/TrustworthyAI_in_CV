"""
Generate visualizations from baseline clean model results
Creates plots and charts for clean model accuracy without attack metrics
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization utilities
from utils.visualization_utils import (
    plot_metric_heatmap,
    plot_training_history,
    set_plot_style
)

# Set style
set_plot_style('whitegrid', font_scale=1.0)
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def plot_accuracy_comparison(results, save_path):
    """
    Plot Clean Accuracy comparison across models and datasets.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Group by dataset
    datasets = df['dataset'].unique()
    models = df['model_display'].unique() if 'model_display' in df.columns else df['model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    
    for i, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset]
        # Ensure data is sorted by model order
        data = data.set_index(model_col).reindex(models).reset_index()
        offset = width * (i - len(datasets)/2 + 0.5)
        ax.bar(x + offset, data['clean_accuracy'], width, 
               label=dataset.upper(), alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('Baseline Model Performance: Clean Accuracy Comparison', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_heatmap_clean_accuracy(results, save_path):
    """
    Plot heatmap of Clean Accuracy using visualization utils.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    
    plot_metric_heatmap(
        results_df=df,
        metric='clean_accuracy',
        row_var=model_col,
        col_var='dataset',
        save_path=save_path,
        title='Baseline Model Performance: Clean Accuracy Heatmap',
        cmap='YlGn',
        show=False
    )


def plot_model_comparison_line(results, save_path):
    """
    Line chart comparing Clean Accuracy for each model across datasets.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    
    # Plot by dataset
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('clean_accuracy')  # type: ignore
        x = np.arange(len(data))
        ax.plot(x, data['clean_accuracy'].to_numpy(), 
                marker='o', linewidth=2.5, markersize=10,
                label=dataset.upper())
        
        # Add model labels
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(i, row['clean_accuracy'] + 0.5, row[model_col], 
                   rotation=45, ha='left', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Model (sorted by accuracy)')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('Baseline Model Performance by Dataset', pad=20)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_accuracy_by_training_style(results, save_path):
    """
    Compare accuracy across different training styles.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    # Check if training_style is available
    if 'training_style' not in df.columns:
        print("  Skipping training style comparison (not available)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    training_styles = df['training_style'].unique()
    datasets = df['dataset'].unique()
    
    x = np.arange(len(training_styles))
    width = 0.35
    
    for i, dataset in enumerate(datasets):
        means = []
        for style in training_styles:
            data = df[(df['dataset'] == dataset) & (df['training_style'] == style)]
            means.append(data['clean_accuracy'].mean() if len(data) > 0 else 0)
        
        offset = width * (i - len(datasets)/2 + 0.5)
        ax.bar(x + offset, means, width, label=dataset.upper(), alpha=0.8)
    
    ax.set_xlabel('Training Style')
    ax.set_ylabel('Average Clean Accuracy (%)')
    ax.set_title('Baseline Performance by Training Style', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(training_styles, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_pretrained_vs_scratch(results, save_path):
    """
    Compare pretrained vs scratch-trained models.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    # Check if pretrained column exists
    if 'pretrained' not in df.columns:
        print("  Skipping pretrained comparison (not available)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    pretrained_means = []
    scratch_means = []
    
    for dataset in datasets:
        pretrained_data = df[(df['dataset'] == dataset) & (df['pretrained'] == True)]
        scratch_data = df[(df['dataset'] == dataset) & (df['pretrained'] == False)]
        
        pretrained_means.append(pretrained_data['clean_accuracy'].mean() if len(pretrained_data) > 0 else 0)
        scratch_means.append(scratch_data['clean_accuracy'].mean() if len(scratch_data) > 0 else 0)
    
    ax.bar(x - width/2, pretrained_means, width, label='Pretrained', alpha=0.8)
    ax.bar(x + width/2, scratch_means, width, label='Scratch', alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Clean Accuracy (%)')
    ax.set_title('Pretrained vs Scratch-Trained Models', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, (pt, sc) in enumerate(zip(pretrained_means, scratch_means)):
        if pt > 0:
            ax.text(i - width/2, pt + 1, f'{pt:.1f}%', ha='center', va='bottom', fontsize=9)
        if sc > 0:
            ax.text(i + width/2, sc + 1, f'{sc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_efficiency(results, save_path):
    """
    Plot training efficiency: accuracy vs epochs.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    # Check if training_epochs is available
    if 'training_epochs' not in df.columns:
        print("  Skipping training efficiency plot (epochs not available)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    colors = sns.color_palette("Set2", n_colors=len(df['dataset'].unique()))
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        ax.scatter(data['training_epochs'], data['clean_accuracy'],
                  label=dataset.upper(), s=150, alpha=0.7,
                  color=colors[idx], edgecolors='black', linewidth=1.5)
        
        # Add model labels
        for _, row in data.iterrows():
            ax.annotate(row[model_col], 
                       (row['training_epochs'], row['clean_accuracy']),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('Training Efficiency: Accuracy vs Epochs', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_model_ranking(results, save_path):
    """
    Plot model ranking based on average accuracy across datasets.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    
    # Calculate average accuracy per model
    model_avg = df.groupby(model_col)['clean_accuracy'].mean().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(model_avg) * 0.5)))
    
    y_pos = np.arange(len(model_avg))
    bars = ax.barh(y_pos, model_avg.to_numpy(), alpha=0.8, 
                   color=sns.color_palette("viridis", len(model_avg)))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_avg.index)
    ax.set_xlabel('Average Clean Accuracy (%)')
    ax.set_title('Model Ranking (Average Across Datasets)', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, v in enumerate(model_avg.to_numpy()):
        ax.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_detailed_comparison_table(results, save_path):
    """
    Create a visual table with detailed model comparison.
    
    Args:
        results: List of experiment results
        save_path: Path to save the figure
    """
    df = pd.DataFrame(results)
    
    # Select key columns
    model_col = 'model_display' if 'model_display' in df.columns else 'model'
    display_cols = [model_col, 'dataset', 'clean_accuracy']
    if 'training_epochs' in df.columns:
        display_cols.append('training_epochs')
    if 'best_train_acc' in df.columns:
        display_cols.append('best_train_acc')
    
    df_display = df[display_cols].copy()
    df_display = df_display.sort_values(['dataset', 'clean_accuracy'], ascending=[True, False])  # type: ignore
    
    # Format values
    df_display['clean_accuracy'] = df_display['clean_accuracy'].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A')
    if 'best_train_acc' in df_display.columns:
        df_display['best_train_acc'] = df_display['best_train_acc'].apply(
            lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A'
        )
    
    # Rename columns for better display
    column_rename = {
        model_col: 'Model',
        'dataset': 'Dataset',
        'clean_accuracy': 'Clean Accuracy',
        'training_epochs': 'Epochs',
        'best_train_acc': 'Best Train Acc'
    }
    df_display = df_display.rename(columns={k: v for k, v in column_rename.items() if k in df_display.columns})
    
    # Create figure with better sizing
    fig, ax = plt.subplots(figsize=(16, max(10, len(df_display) * 0.35 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate column widths based on content
    col_widths = []
    for col in df_display.columns:
        if col == 'Model':
            col_widths.append(0.35)  # Wider for model names
        elif col == 'Dataset':
            col_widths.append(0.15)
        elif col == 'Epochs':
            col_widths.append(0.12)
        else:
            col_widths.append(0.18)
    
    table = ax.table(cellText=df_display.values,
                    colLabels=df_display.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Baseline Model Performance Summary', fontsize=16, fontweight='bold', pad=30)
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_baseline_visualizations(results_path, save_dir='./results/figures/baseline'):
    """
    Generate all visualizations from baseline clean model results.
    
    Args:
        results_path: Path to JSON results file
        save_dir: Directory to save figures
    """
    print("="*70)
    print("Generating Visualizations from Baseline Clean Model Results")
    print("="*70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)
    print(f"Loaded {len(results)} experiment results")

    # Filter out any attack results (where attack_success_rate is not None)
    results = [r for r in results if r.get('attack_success_rate') is None]
    print(f"Filtered to {len(results)} baseline results")

    # Aggregate duplicates: keep best clean_accuracy per (model, dataset, pretrained, training_style)
    # This ensures we don't mix metadata from pretrained and scratch models
    import pandas as _pd
    _df = _pd.DataFrame(results)
    for _col in ('clean_accuracy', 'best_train_acc', 'best_val_loss'):
        if _col in _df.columns:
            _df[_col] = _pd.to_numeric(_df[_col], errors='coerce')
    
    # Build groupby keys dynamically
    groupby_keys = ['model', 'dataset']
    if 'pretrained' in _df.columns:
        groupby_keys.append('pretrained')
    if 'training_style' in _df.columns:
        groupby_keys.append('training_style')
    
    _df = (
        _df.groupby(groupby_keys, as_index=False)
           .agg({
               'clean_accuracy': 'max',
               **{c: 'first' for c in _df.columns
                  if c not in groupby_keys + ['clean_accuracy']}
           })
    )
    results = _df.to_dict(orient='records')
    
    # Create descriptive model names that include training type
    for r in results:
        if 'pretrained' in r and 'training_style' in r:
            if r['pretrained']:
                r['model_display'] = f"{r['model']}_pretrained"
            else:
                r['model_display'] = f"{r['model']}_scratch"
        else:
            r['model_display'] = r['model']
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\nGenerating plots...")
    
    # 1. Clean Accuracy Bar Chart
    print("  1. Clean Accuracy Bar Chart...")
    plot_accuracy_comparison(
        results,
        os.path.join(save_dir, f'baseline_clean_accuracy_comparison_{timestamp}.png')
    )
    
    # 2. Clean Accuracy Heatmap
    print("  2. Clean Accuracy Heatmap...")
    plot_heatmap_clean_accuracy(
        results,
        os.path.join(save_dir, f'baseline_clean_accuracy_heatmap_{timestamp}.png')
    )
    
    # 3. Line Chart
    print("  3. Model Comparison Line Chart...")
    plot_model_comparison_line(
        results,
        os.path.join(save_dir, f'baseline_model_comparison_{timestamp}.png')
    )
    
    # 4. Training Style Comparison
    print("  4. Training Style Comparison...")
    plot_accuracy_by_training_style(
        results,
        os.path.join(save_dir, f'baseline_training_style_comparison_{timestamp}.png')
    )
    
    # 5. Pretrained vs Scratch
    print("  5. Pretrained vs Scratch Comparison...")
    plot_pretrained_vs_scratch(
        results,
        os.path.join(save_dir, f'baseline_pretrained_vs_scratch_{timestamp}.png')
    )
    
    # 6. Training Efficiency - DISABLED
    # print("  6. Training Efficiency...")
    # plot_training_efficiency(
    #     results,
    #     os.path.join(save_dir, f'baseline_training_efficiency_{timestamp}.png')
    # )
    
    # 7. Model Ranking
    print("  6. Model Ranking...")
    plot_model_ranking(
        results,
        os.path.join(save_dir, f'baseline_model_ranking_{timestamp}.png')
    )
    
    # 8. Detailed Table - DISABLED
    # print("  7. Detailed Comparison Table...")
    # plot_detailed_comparison_table(
    #     results,
    #     os.path.join(save_dir, f'baseline_comparison_table_{timestamp}.png')
    # )
    
    print("\n" + "="*70)
    print("All baseline visualizations generated successfully!")
    print(f"Saved to: {save_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate visualizations from baseline clean model results'
    )
    parser.add_argument('--results', type=str, default='./results/clean_results.json',
                       help='Path to baseline results JSON file')
    parser.add_argument('--save-dir', type=str, default='./results/figures/baseline',
                       help='Directory to save figures')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        exit(1)
    
    # Generate visualizations
    generate_all_baseline_visualizations(args.results, args.save_dir)
