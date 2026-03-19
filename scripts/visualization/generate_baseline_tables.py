"""
Generate tables from baseline clean model results (without attack metrics)
Creates formatted tables with Clean Accuracy and training statistics
Supports CSV, LaTeX, and Markdown formats
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Dict, List, Optional


def load_results(results_path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create comprehensive summary table with baseline results.
    
    Args:
        results: List of experiment result dictionaries
        
    Returns:
        pandas DataFrame with formatted results
    """
    df = pd.DataFrame(results)
    
    # Filter out any attack results
    df = df[df['attack_success_rate'].isna()]
    
    # Identify available columns (no ASR for baseline)
    base_columns = ['dataset', 'model', 'clean_accuracy']
    optional_columns = ['pretrained', 'training_style', 'training_epochs', 
                       'best_train_acc', 'best_val_loss']
    
    # Select columns that exist in the data
    columns = base_columns.copy()
    for col in optional_columns:
        if col in df.columns:
            columns.append(col)
    
    df = df[columns]
    
    # Round numeric values
    numeric_cols = ['clean_accuracy', 'best_train_acc', 'best_val_loss']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    
    # Rename columns for better readability
    rename_map = {
        'dataset': 'Dataset',
        'model': 'Model',
        'clean_accuracy': 'Clean Acc (%)',
        'pretrained': 'Pretrained',
        'training_style': 'Training Style',
        'training_epochs': 'Epochs',
        'best_train_acc': 'Best Train Acc (%)',
        'best_val_loss': 'Best Val Loss'
    }
    
    df = df.rename(columns=rename_map)
    
    return df


def create_pivot_table(results: List[Dict], metric: str = 'clean_accuracy') -> pd.DataFrame:
    """
    Create pivot table with models as rows and datasets as columns.
    
    Args:
        results: List of result dictionaries
        metric: Metric to pivot (default: 'clean_accuracy')
        
    Returns:
        pandas DataFrame pivot table
    """
    df = pd.DataFrame(results)
    
    # Filter baseline results only
    df = df[df['attack_success_rate'].isna()]
    
    # Create model display names if pretrained/training_style available
    if 'pretrained' in df.columns and 'training_style' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        index_col = 'model_display'
    else:
        index_col = 'model'
    
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Use pivot_table with max to handle multiple runs per (model, dataset) pair
    pivot = df.pivot_table(index=index_col, columns='dataset', values=metric, aggfunc='max')
    
    # Round values
    pivot = pivot.round(2)
    
    # Sort by model name
    pivot = pivot.sort_index()
    
    return pivot


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create comprehensive comparison table with clean accuracy across models and datasets.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        pandas DataFrame comparison table
    """
    df = pd.DataFrame(results)
    
    # Filter baseline results only
    df = df[df['attack_success_rate'].isna()]
    
    # Coerce numeric columns
    df['clean_accuracy'] = pd.to_numeric(df['clean_accuracy'], errors='coerce')
    
    # Create model display names
    if 'pretrained' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        index_col = 'model_display'
    else:
        index_col = 'model'
    
    # Create pivot table
    clean_acc_pivot = df.pivot_table(
        index=index_col, 
        columns='dataset', 
        values='clean_accuracy', 
        aggfunc='max'
    )
    
    # Calculate average across datasets
    clean_acc_pivot['Average'] = clean_acc_pivot.mean(axis=1)
    
    # Round all values
    clean_acc_pivot = clean_acc_pivot.round(2)
    
    # Sort by average (descending)
    clean_acc_pivot = clean_acc_pivot.sort_values('Average', ascending=False)
    
    return clean_acc_pivot


def create_ranking_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create model ranking based on average clean accuracy.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        pandas DataFrame with model rankings
    """
    df = pd.DataFrame(results)
    
    # Filter baseline results only
    df = df[df['attack_success_rate'].isna()]
    
    # Create model display names
    if 'pretrained' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        group_col = 'model_display'
    else:
        group_col = 'model'
    
    # Coerce numeric
    df['clean_accuracy'] = pd.to_numeric(df['clean_accuracy'], errors='coerce')
    
    # Calculate average clean accuracy per model
    ranking = df.groupby(group_col).agg({
        'clean_accuracy': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    ranking.columns = ['Avg Clean Acc (%)', 'Std Dev', 'Min', 'Max', 'Runs']
    ranking = ranking.sort_values('Avg Clean Acc (%)', ascending=False)
    ranking.index.name = 'Model'
    
    return ranking


def create_statistics_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create overall statistics table.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        pandas DataFrame with statistics
    """
    df = pd.DataFrame(results)
    
    # Filter baseline results only
    df = df[df['attack_success_rate'].isna()]
    
    df['clean_accuracy'] = pd.to_numeric(df['clean_accuracy'], errors='coerce')
    
    stats = pd.DataFrame({
        'Metric': [
            'Number of Models',
            'Number of Datasets',
            'Total Experiments',
            'Avg Clean Accuracy (%)',
            'Std Clean Accuracy',
            'Min Clean Accuracy (%)',
            'Max Clean Accuracy (%)'
        ],
        'Value': [
            df['model'].nunique(),
            df['dataset'].nunique(),
            len(df),
            round(df['clean_accuracy'].mean(), 2),
            round(df['clean_accuracy'].std(), 2),
            round(df['clean_accuracy'].min(), 2),
            round(df['clean_accuracy'].max(), 2)
        ]
    })
    
    return stats


def create_dataset_stats(results: List[Dict]) -> pd.DataFrame:
    """Create statistics grouped by dataset."""
    df = pd.DataFrame(results)
    df = df[df['attack_success_rate'].isna()]
    df['clean_accuracy'] = pd.to_numeric(df['clean_accuracy'], errors='coerce')
    
    stats = df.groupby('dataset').agg({
        'clean_accuracy': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    stats.columns = ['Avg Clean Acc (%)', 'Std Dev', 'Min', 'Max', 'Models']
    stats.index.name = 'Dataset'
    
    return stats


def create_model_stats(results: List[Dict]) -> pd.DataFrame:
    """Create statistics grouped by model architecture."""
    df = pd.DataFrame(results)
    df = df[df['attack_success_rate'].isna()]
    df['clean_accuracy'] = pd.to_numeric(df['clean_accuracy'], errors='coerce')
    
    # Create model display names
    if 'pretrained' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        group_col = 'model_display'
    else:
        group_col = 'model'
    
    stats = df.groupby(group_col).agg({
        'clean_accuracy': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    stats.columns = ['Avg Clean Acc (%)', 'Std Dev', 'Min', 'Max', 'Datasets']
    stats.index.name = 'Model'
    
    return stats


def print_latex_table(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert DataFrame to LaTeX table format."""
    latex = df.to_latex(
        float_format="%.2f",
        na_rep='-',
        caption=caption,
        label=label,
        escape=False
    )
    return latex


def print_markdown_table(df: pd.DataFrame, title: str = "") -> str:
    """Convert DataFrame to Markdown table format."""
    md = f"\n## {title}\n\n" if title else "\n"
    md += df.to_markdown(floatfmt=".2f")
    md += "\n"
    return md


def generate_all_baseline_tables(
    results_path: str,
    save_dir: str = './results/tables/baseline',
    experiment_name: str = 'baseline'
) -> Dict[str, pd.DataFrame]:
    """
    Generate all baseline tables from results.
    
    Args:
        results_path: Path to JSON results file
        save_dir: Directory to save tables
        experiment_name: Name for the experiment (for captions and filenames)
        
    Returns:
        Dictionary containing all generated tables
    """
    print("="*70)
    print("Generating Baseline Tables")
    print("="*70)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)
    print(f"Loaded {len(results)} experiment results")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_prefix = experiment_name.lower().replace(' ', '_')
    
    # 1. Summary Table
    print("\n" + "-"*70)
    print("1. Summary Table")
    print("-"*70)
    summary_df = create_summary_table(results)
    summary_path = os.path.join(save_dir, f'{exp_prefix}_summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")
    print(summary_df.head(10))
    
    # 2. Clean Accuracy Pivot Table
    print("\n" + "-"*70)
    print("2. Clean Accuracy Pivot Table")
    print("-"*70)
    clean_acc_pivot = create_pivot_table(results, 'clean_accuracy')
    clean_pivot_path = os.path.join(save_dir, f'{exp_prefix}_clean_accuracy_pivot_{timestamp}.csv')
    clean_acc_pivot.to_csv(clean_pivot_path)
    print(f"Clean accuracy pivot saved to: {clean_pivot_path}")
    print(clean_acc_pivot)
    
    # 3. Comparison Table
    print("\n" + "-"*70)
    print("3. Comparison Table")
    print("-"*70)
    comparison_df = create_comparison_table(results)
    comp_path = os.path.join(save_dir, f'{exp_prefix}_comparison_table_{timestamp}.csv')
    comparison_df.to_csv(comp_path)
    print(f"Comparison table saved to: {comp_path}")
    print(comparison_df)
    
    # 4. Model Ranking
    print("\n" + "-"*70)
    print("4. Model Ranking")
    print("-"*70)
    ranking = create_ranking_table(results)
    ranking_path = os.path.join(save_dir, f'{exp_prefix}_ranking_{timestamp}.csv')
    ranking.to_csv(ranking_path)
    print(f"Model ranking saved to: {ranking_path}")
    print(ranking)
    
    # 5. Overall Statistics
    print("\n" + "-"*70)
    print("5. Overall Statistics")
    print("-"*70)
    stats = create_statistics_table(results)
    stats_path = os.path.join(save_dir, f'{exp_prefix}_statistics_{timestamp}.csv')
    stats.to_csv(stats_path, index=False)
    print(f"Statistics saved to: {stats_path}")
    print(stats)
    
    # 6. Dataset Statistics
    print("\n" + "-"*70)
    print("6. Dataset Statistics")
    print("-"*70)
    dataset_stats = create_dataset_stats(results)
    dataset_stats_path = os.path.join(save_dir, f'{exp_prefix}_dataset_stats_{timestamp}.csv')
    dataset_stats.to_csv(dataset_stats_path)
    print(f"Dataset statistics saved to: {dataset_stats_path}")
    print(dataset_stats)
    
    # 7. Model Statistics
    print("\n" + "-"*70)
    print("7. Model Statistics")
    print("-"*70)
    model_stats = create_model_stats(results)
    model_stats_path = os.path.join(save_dir, f'{exp_prefix}_model_stats_{timestamp}.csv')
    model_stats.to_csv(model_stats_path)
    print(f"Model statistics saved to: {model_stats_path}")
    print(model_stats)
    
    # 8. LaTeX Format
    print("\n" + "-"*70)
    print("8. LaTeX Format Tables")
    print("-"*70)
    
    latex_dir = os.path.join(save_dir, 'latex')
    os.makedirs(latex_dir, exist_ok=True)
    
    # Clean Accuracy LaTeX
    latex_clean = print_latex_table(
        clean_acc_pivot,
        caption=f"Baseline Clean Accuracy (\\%) - {experiment_name.title()}",
        label=f"tab:{exp_prefix}_clean_acc"
    )
    latex_clean_path = os.path.join(latex_dir, f'{exp_prefix}_clean_accuracy_{timestamp}.tex')
    with open(latex_clean_path, 'w') as f:
        f.write(latex_clean)
    print(f"Clean Accuracy LaTeX saved to: {latex_clean_path}")
    
    # Comparison LaTeX
    latex_comparison = print_latex_table(
        comparison_df,
        caption=f"Baseline Comparison Table - {experiment_name.title()}",
        label=f"tab:{exp_prefix}_comparison"
    )
    latex_comparison_path = os.path.join(latex_dir, f'{exp_prefix}_comparison_{timestamp}.tex')
    with open(latex_comparison_path, 'w') as f:
        f.write(latex_comparison)
    print(f"Comparison LaTeX saved to: {latex_comparison_path}")
    
    # 9. Markdown Tables
    print("\n" + "-"*70)
    print("9. Markdown Format Tables")
    print("-"*70)
    
    markdown_dir = os.path.join(save_dir, 'markdown')
    os.makedirs(markdown_dir, exist_ok=True)
    
    md_content = f"# {experiment_name.title()} Baseline Results\n\n"
    md_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += print_markdown_table(clean_acc_pivot, "Clean Accuracy (%)")
    md_content += print_markdown_table(comparison_df, "Comparison Table")
    md_content += print_markdown_table(ranking, "Model Ranking")
    
    md_path = os.path.join(markdown_dir, f'{exp_prefix}_results_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Markdown tables saved to: {md_path}")
    
    print("\n" + "="*70)
    print("All baseline tables generated successfully!")
    print(f"Tables saved to: {save_dir}")
    print("="*70)
    
    return {
        'summary': summary_df,
        'clean_accuracy_pivot': clean_acc_pivot,
        'comparison': comparison_df,
        'ranking': ranking,
        'statistics': stats,
        'dataset_stats': dataset_stats,
        'model_stats': model_stats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate tables from baseline clean model results (no attack metrics)'
    )
    parser.add_argument('--results', type=str, default='./results/clean_results.json',
                       help='Path to baseline results JSON file')
    parser.add_argument('--save-dir', type=str, default='./results/tables/baseline',
                       help='Directory to save tables')
    parser.add_argument('--experiment-name', type=str, default='baseline',
                       help='Name of the experiment for captions and filenames')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        exit(1)
    
    # Generate tables
    tables = generate_all_baseline_tables(args.results, args.save_dir, args.experiment_name)
