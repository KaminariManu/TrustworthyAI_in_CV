"""
Visualization utilities for backdoor attack results and metrics.
Provides reusable functions for plotting training history, comparisons, and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import os
from datetime import datetime


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training History",
    show: bool = True
):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Acc', linewidth=2, marker='o')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Acc', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_attack_comparison(
    results_df: pd.DataFrame,
    metric: str = 'ASR',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comparison of different attacks.
    
    Args:
        results_df: DataFrame with columns ['Attack', 'Model', metric]
        metric: Metric to plot (e.g., 'ASR', 'Clean_Acc')
        save_path: Path to save figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    if 'Model' in results_df.columns:
        sns.barplot(data=results_df, x='Attack', y=metric, hue='Model', palette='Set2')
    else:
        sns.barplot(data=results_df, x='Attack', y=metric, palette='Set2')
    
    plt.xlabel('Attack Type')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison Across Attacks')
    plt.xticks(rotation=45, ha='right')
    
    if 'Model' in results_df.columns:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_defense_effectiveness(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot defense effectiveness (ASR reduction).
    
    Args:
        results_df: DataFrame with columns ['Defense', 'ASR_Before', 'ASR_After', 'Clean_Acc']
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ASR comparison
    x = np.arange(len(results_df))
    width = 0.35
    
    ax1.bar(x - width/2, results_df['ASR_Before'], width, 
            label='Before Defense', alpha=0.8, color='coral')
    ax1.bar(x + width/2, results_df['ASR_After'], width, 
            label='After Defense', alpha=0.8, color='lightblue')
    ax1.set_xlabel('Defense Method')
    ax1.set_ylabel('Attack Success Rate (%)')
    ax1.set_title('Defense Effectiveness on ASR')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Defense'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Clean accuracy impact
    ax2.bar(results_df['Defense'], results_df['Clean_Acc'], 
            alpha=0.8, color='green')
    ax2.set_xlabel('Defense Method')
    ax2.set_ylabel('Clean Accuracy (%)')
    ax2.set_title('Clean Accuracy After Defense')
    ax2.set_xticklabels(results_df['Defense'], rotation=45, ha='right')
    ax2.axhline(y=90, color='r', linestyle='--', linewidth=2, label='Target Threshold')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    show: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        show: Whether to display the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_trigger_visualization(
    clean_image: np.ndarray,
    poisoned_image: np.ndarray,
    trigger_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize trigger pattern on clean image.
    
    Args:
        clean_image: Original clean image (H, W, C)
        poisoned_image: Image with trigger (H, W, C)
        trigger_mask: Binary mask showing trigger location (H, W) or (H, W, C)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    n_plots = 3 if trigger_mask is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1], None]
    
    # Clean image
    axes[0].imshow(clean_image)
    axes[0].set_title('Clean Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Poisoned image
    axes[1].imshow(poisoned_image)
    axes[1].set_title('Poisoned Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Trigger mask
    if trigger_mask is not None:
        if len(trigger_mask.shape) == 3:
            trigger_mask = trigger_mask[:, :, 0]
        axes[2].imshow(trigger_mask, cmap='Reds', alpha=0.8)
        axes[2].set_title('Trigger Mask', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_heatmap(
    results_df: pd.DataFrame,
    metric: str,
    row_var: str = 'model',
    col_var: str = 'dataset',
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = 'YlOrRd',
    show: bool = True
):
    """
    Plot heatmap for a specific metric across different variables.
    
    Args:
        results_df: DataFrame with results
        metric: Metric column to visualize
        row_var: Variable for rows
        col_var: Variable for columns
        save_path: Path to save figure
        title: Plot title (default: uses metric name)
        cmap: Colormap name
        show: Whether to display the plot
    """
    pivot = results_df.pivot(index=row_var, columns=col_var, values=metric)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap,
                cbar_kws={'label': metric}, ax=ax,
                vmin=0, vmax=100)
    
    if title is None:
        title = f'{metric} Heatmap'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(col_var.capitalize())
    ax.set_ylabel(row_var.capitalize())
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_metric_comparison(
    results_df: pd.DataFrame,
    metrics: List[str],
    group_by: str = 'model',
    save_path: Optional[str] = None,
    title: str = "Metric Comparison",
    show: bool = True
):
    """
    Plot multiple metrics in subplots for comparison.
    
    Args:
        results_df: DataFrame with results
        metrics: List of metric columns to plot
        group_by: Column to group by (e.g., 'model', 'attack')
        save_path: Path to save figure
        title: Main plot title
        show: Whether to display the plot
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        if 'dataset' in results_df.columns:
            sns.barplot(data=results_df, x=group_by, y=metric, 
                       hue='dataset', ax=ax, palette='Set2')
        else:
            sns.barplot(data=results_df, x=group_by, y=metric, 
                       ax=ax, palette='Set2')
        
        ax.set_xlabel(group_by.capitalize())
        ax.set_ylabel(metric)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results_table(
    results_df: pd.DataFrame,
    save_path: str,
    float_format: str = '%.2f'
):
    """
    Save results to CSV and display as formatted table.
    
    Args:
        results_df: DataFrame with results
        save_path: Path to save CSV
        float_format: Format for floating point numbers
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(save_path, index=False, float_format=float_format)
    print(f"\nResults saved to {save_path}")
    
    # Display formatted table
    print("\nResults Summary:")
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


def create_results_summary(
    experiments: List[Dict],
    save_dir: str = './results/tables',
    filename_prefix: str = 'results_summary'
) -> pd.DataFrame:
    """
    Create summary table from multiple experiments.
    
    Args:
        experiments: List of experiment dictionaries
        save_dir: Directory to save results
        filename_prefix: Prefix for the output filename
        
    Returns:
        DataFrame with summary results
    """
    df = pd.DataFrame(experiments)
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'{filename_prefix}_{timestamp}.csv')
    
    save_results_table(df, save_path)
    
    return df


def plot_scatter_with_annotations(
    results_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    label_col: Optional[str] = None,
    hue_col: Optional[str] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True
):
    """
    Create scatter plot with optional annotations.
    
    Args:
        results_df: DataFrame with results
        x_metric: Column for x-axis
        y_metric: Column for y-axis
        label_col: Column to use for point labels
        hue_col: Column to use for color grouping
        save_path: Path to save figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if hue_col and hue_col in results_df.columns:
        for hue_val in results_df[hue_col].unique():
            data = results_df[results_df[hue_col] == hue_val]
            ax.scatter(data[x_metric], data[y_metric], 
                      label=hue_val, s=120, alpha=0.7, edgecolors='black')
    else:
        ax.scatter(results_df[x_metric], results_df[y_metric],
                  s=120, alpha=0.7, edgecolors='black')
    
    # Add labels
    if label_col and label_col in results_df.columns:
        for _, row in results_df.iterrows():
            ax.annotate(row[label_col],
                       (row[x_metric], row[y_metric]),
                       fontsize=9, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel(x_metric.replace('_', ' ').title())
    ax.set_ylabel(y_metric.replace('_', ' ').title())
    
    if title is None:
        title = f'{y_metric} vs {x_metric}'
    ax.set_title(title)
    
    if hue_col:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_sample_images_grid(
    images: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    n_cols: int = 4,
    save_path: Optional[str] = None,
    title: str = "Sample Images",
    show: bool = True
):
    """
    Plot a grid of sample images.
    
    Args:
        images: Array of images or list of images
        labels: Optional labels for each image
        n_cols: Number of columns in the grid
        save_path: Path to save figure
        title: Main plot title
        show: Whether to display the plot
    """
    if isinstance(images, np.ndarray):
        if len(images.shape) == 4:
            images = [images[i] for i in range(images.shape[0])]
    
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for idx, img in enumerate(images):
        ax = axes[idx]
        ax.imshow(img)
        
        if labels and idx < len(labels):
            ax.set_title(labels[idx])
        
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def set_plot_style(style: str = 'whitegrid', font_scale: float = 1.0):
    """
    Set the plotting style for all subsequent plots.
    
    Args:
        style: Seaborn style name ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        font_scale: Scale factor for fonts
    """
    sns.set_style(style)
    sns.set_context("notebook", font_scale=font_scale)
    plt.rcParams['font.size'] = 12 * font_scale
    plt.rcParams['axes.titlesize'] = 14 * font_scale
    plt.rcParams['axes.labelsize'] = 12 * font_scale


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_training_history()")
    print("  - plot_attack_comparison()")
    print("  - plot_defense_effectiveness()")
    print("  - plot_confusion_matrix()")
    print("  - plot_trigger_visualization()")
    print("  - plot_metric_heatmap()")
    print("  - plot_multi_metric_comparison()")
    print("  - plot_scatter_with_annotations()")
    print("  - plot_sample_images_grid()")
    print("  - save_results_table()")
    print("  - create_results_summary()")
    print("  - set_plot_style()")
