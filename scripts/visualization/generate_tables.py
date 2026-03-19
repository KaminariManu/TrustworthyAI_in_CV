"""
Generate tables from backdoor ATTACK results (includes ASR metrics)
Creates formatted tables with metrics like Clean Accuracy, ASR, and statistics
For baseline clean models without attacks, use generate_baseline_tables.py instead.
Supports CSV, LaTeX, and Markdown formats
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Union


def load_results(results_path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def _is_vit_deit_scratch(row: Dict) -> bool:
    """Return True for scratch-trained ViT/DeiT entries."""
    model = str(row.get('model', '')).lower()
    if not (model.startswith('vit') or model.startswith('deit')):
        return False

    pretrained = row.get('pretrained')
    if pretrained is False:
        return True

    training_style = str(row.get('training_style', '')).lower()
    return training_style in {'scratch', 'vit_scratch'}


def filter_vit_deit_scratch(results: List[Dict]) -> List[Dict]:
    """Drop ViT/DeiT scratch rows to keep scope aligned with baseline."""
    return [r for r in results if not _is_vit_deit_scratch(r)]


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create comprehensive summary table with all results.
    
    Args:
        results: List of experiment result dictionaries
        
    Returns:
        pandas DataFrame with formatted results
    """
    df = pd.DataFrame(results)
    
    # Identify available columns
    base_columns = ['dataset', 'model', 'clean_accuracy', 'attack_success_rate']
    optional_columns = ['pretrained', 'training_style', 'poison_ratio', 'num_poisoned',
                       'training_epochs', 'poisoning_rate', 'target_label', 'trigger_size']
    
    # Select columns that exist in the data
    columns = base_columns.copy()
    for col in optional_columns:
        if col in df.columns:
            columns.append(col)
    
    if 'error' in df.columns:
        columns.append('error')
    
    df = df[columns]
    
    # Round numeric values (coerce None/null to NaN first so .round() works)
    if 'clean_accuracy' in df.columns:
        df['clean_accuracy'] = pd.to_numeric(df['clean_accuracy'], errors='coerce').round(2)
    if 'attack_success_rate' in df.columns:
        df['attack_success_rate'] = pd.to_numeric(df['attack_success_rate'], errors='coerce').round(2)
    
    # Rename columns for better readability
    rename_map = {
        'dataset': 'Dataset',
        'model': 'Model',
        'pretrained': 'Pretrained',
        'training_style': 'Training Style',
        'clean_accuracy': 'Clean Acc (%)',
        'attack_success_rate': 'ASR (%)',
        'poison_ratio': 'Poison Ratio',
        'poisoning_rate': 'Poison Rate',
        'num_poisoned': 'Poisoned Samples',
        'training_epochs': 'Epochs',
        'target_label': 'Target Label',
        'trigger_size': 'Trigger Size'
    }
    
    df = df.rename(columns=rename_map)
    
    return df


def create_pivot_table(results: List[Dict], metric: str = 'clean_accuracy') -> pd.DataFrame:
    """
    Create pivot table with models as rows and datasets as columns.
    Separates pretrained vs scratch-trained models when 'pretrained' field is available.
    
    Args:
        results: List of result dictionaries
        metric: Metric to pivot (e.g., 'clean_accuracy', 'attack_success_rate')
        
    Returns:
        pandas DataFrame pivot table
    """
    df = pd.DataFrame(results)
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Create model display names including pretrained/scratch info
    if 'pretrained' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        index_col = 'model_display'
    else:
        index_col = 'model'

    # Use pivot_table with max to handle multiple runs per (model, dataset) pair
    pivot = df.pivot_table(index=index_col, columns='dataset', values=metric, aggfunc='max')
    pivot.index.name = 'Model'

    # Round values
    pivot = pivot.round(2)

    # Sort by model name
    pivot = pivot.sort_index()

    return pivot


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create side-by-side comparison table with multiple metrics.
    Separates pretrained vs scratch-trained models when 'pretrained' field is available.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        pandas DataFrame with multi-level columns
    """
    df = pd.DataFrame(results)
    for col in ('clean_accuracy', 'attack_success_rate'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create model display names including pretrained/scratch info
    if 'pretrained' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        index_col = 'model_display'
    else:
        index_col = 'model'

    # Use pivot_table with max to handle multiple runs per (model, dataset) pair
    clean_acc_pivot = df.pivot_table(index=index_col, columns='dataset', values='clean_accuracy', aggfunc='max')
    asr_pivot = df.pivot_table(index=index_col, columns='dataset', values='attack_success_rate', aggfunc='max')

    # Round values
    clean_acc_pivot = clean_acc_pivot.round(2)
    asr_pivot = asr_pivot.round(2)
    
    # Combine into multi-level columns
    combined = pd.concat([clean_acc_pivot, asr_pivot], axis=1, 
                        keys=['Clean Accuracy (%)', 'ASR (%)'])
    
    # Reorder columns
    combined = combined.swaplevel(axis=1).sort_index(axis=1)
    
    return combined


def create_ranking_table(results: List[Dict], metric: str = 'attack_success_rate', 
                        ascending: bool = False) -> pd.DataFrame:
    """
    Create ranking table sorted by a specific metric.
    Includes pretrained/scratch distinction when 'pretrained' field is available.
    
    Args:
        results: List of result dictionaries
        metric: Metric to rank by
        ascending: Sort order (False for descending)
        
    Returns:
        pandas DataFrame with ranking
    """
    df = pd.DataFrame(results)

    # Create model display names including pretrained/scratch info
    if 'pretrained' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        model_col = 'model_display'
    else:
        model_col = 'model'

    # Select relevant columns
    columns = [model_col, 'dataset', 'clean_accuracy', 'attack_success_rate']
    if 'pretrained' in df.columns:
        columns.insert(1, 'pretrained')
    if 'training_style' in df.columns:
        columns.insert(2, 'training_style')
    df_rank = df[columns].copy()

    # Coerce numeric values
    df_rank['clean_accuracy'] = pd.to_numeric(df_rank['clean_accuracy'], errors='coerce')
    df_rank['attack_success_rate'] = pd.to_numeric(df_rank['attack_success_rate'], errors='coerce')

    # Sort by metric
    df_rank = df_rank.sort_values(by=metric, ascending=ascending)

    # Add rank column
    df_rank.insert(0, 'Rank', range(1, len(df_rank) + 1))

    # Round numeric values
    df_rank['clean_accuracy'] = df_rank['clean_accuracy'].round(2)
    df_rank['attack_success_rate'] = df_rank['attack_success_rate'].round(2)

    # Rename columns
    rename_map = {
        model_col: 'Model',
        'dataset': 'Dataset',
        'pretrained': 'Pretrained',
        'training_style': 'Training Style',
        'clean_accuracy': 'Clean Acc (%)',
        'attack_success_rate': 'ASR (%)'
    }
    df_rank = df_rank.rename(columns={k: v for k, v in rename_map.items() if k in df_rank.columns})

    return df_rank


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics across all experiments.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with statistics
    """
    df = pd.DataFrame(results)
    asr = pd.to_numeric(df['attack_success_rate'], errors='coerce')
    asr_available = asr.notna().any()

    stats = {
        'Total Experiments': len(df),
        'Number of Models': df['model'].nunique(),
        'Number of Datasets': df['dataset'].nunique(),
        'Average Clean Accuracy': df['clean_accuracy'].mean(),
        'Std Clean Accuracy': df['clean_accuracy'].std(),
        'Min Clean Accuracy': df['clean_accuracy'].min(),
        'Max Clean Accuracy': df['clean_accuracy'].max(),
        'Average ASR': asr.mean() if asr_available else 'N/A',
        'Std ASR':     asr.std()  if asr_available else 'N/A',
        'Min ASR':     asr.min()  if asr_available else 'N/A',
        'Max ASR':     asr.max()  if asr_available else 'N/A',
        'Best Clean Accuracy':   df['clean_accuracy'].max(),
        'Best Clean Acc Model':  df.loc[df['clean_accuracy'].idxmax(), 'model'],
        'Best Clean Acc Dataset': df.loc[df['clean_accuracy'].idxmax(), 'dataset'],
        'Highest ASR':       asr.max()                          if asr_available else 'N/A',
        'Highest ASR Model': df.loc[asr.idxmax(), 'model']      if asr_available else 'N/A',
        'Highest ASR Dataset': df.loc[asr.idxmax(), 'dataset']  if asr_available else 'N/A',
        'Lowest ASR':        asr.min()                          if asr_available else 'N/A',
        'Lowest ASR Model':  df.loc[asr.idxmin(), 'model']      if asr_available else 'N/A',
        'Lowest ASR Dataset': df.loc[asr.idxmin(), 'dataset']   if asr_available else 'N/A',
    }

    return stats


def create_per_dataset_stats(results: List[Dict]) -> pd.DataFrame:
    """
    Create statistics table grouped by dataset.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        pandas DataFrame with per-dataset statistics
    """
    df = pd.DataFrame(results)
    
    stats = df.groupby('dataset').agg({
        'clean_accuracy': ['mean', 'std', 'min', 'max'],
        'attack_success_rate': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.rename(columns={
        'clean_accuracy_mean': 'Clean Acc Mean',
        'clean_accuracy_std': 'Clean Acc Std',
        'clean_accuracy_min': 'Clean Acc Min',
        'clean_accuracy_max': 'Clean Acc Max',
        'attack_success_rate_mean': 'ASR Mean',
        'attack_success_rate_std': 'ASR Std',
        'attack_success_rate_min': 'ASR Min',
        'attack_success_rate_max': 'ASR Max'
    })
    
    return stats


def create_per_model_stats(results: List[Dict]) -> pd.DataFrame:
    """
    Create statistics table grouped by model, separating pretrained vs scratch.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        pandas DataFrame with per-model statistics
    """
    df = pd.DataFrame(results)

    # Create model display names including pretrained/scratch info
    if 'pretrained' in df.columns:
        df['model_display'] = df.apply(
            lambda row: f"{row['model']}_{'pretrained' if row['pretrained'] else 'scratch'}",
            axis=1
        )
        group_col = 'model_display'
    else:
        group_col = 'model'

    for col in ('clean_accuracy', 'attack_success_rate'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = df.groupby(group_col).agg({
        'clean_accuracy': ['mean', 'std', 'min', 'max'],
        'attack_success_rate': ['mean', 'std', 'min', 'max']
    }).round(2)

    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.rename(columns={
        'clean_accuracy_mean': 'Clean Acc Mean',
        'clean_accuracy_std': 'Clean Acc Std',
        'clean_accuracy_min': 'Clean Acc Min',
        'clean_accuracy_max': 'Clean Acc Max',
        'attack_success_rate_mean': 'ASR Mean',
        'attack_success_rate_std': 'ASR Std',
        'attack_success_rate_min': 'ASR Min',
        'attack_success_rate_max': 'ASR Max'
    })
    stats.index.name = 'Model'

    return stats


def print_latex_table(df: pd.DataFrame, caption: str = "Attack Results",
                     label: str = "tab:results") -> str:
    """
    Convert DataFrame to LaTeX table format.
    
    Args:
        df: DataFrame to convert
        caption: Table caption
        label: LaTeX label
        
    Returns:
        LaTeX table string
    """
    latex_str = df.to_latex(
        index=True,
        float_format="%.2f",
        caption=caption,
        label=label,
        position='htbp',
        column_format='l' + 'c' * len(df.columns),
        escape=False
    )
    return latex_str


def print_markdown_table(df: pd.DataFrame, title: Optional[str] = None) -> str:
    """
    Convert DataFrame to Markdown table format.
    
    Args:
        df: DataFrame to convert
        title: Optional title for the table
        
    Returns:
        Markdown table string
    """
    md_str = ""
    if title:
        md_str += f"## {title}\n\n"
    
    md_str += df.to_markdown(index=True, floatfmt=".2f")
    md_str += "\n\n"
    
    return md_str


def generate_all_tables(results_path: str, save_dir: str = './results/tables',
                       attack_name: str = 'backdoor') -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Generate all tables from backdoor attack results.
    
    Args:
        results_path: Path to JSON results file
        save_dir: Directory to save tables
        attack_name: Name of the attack (for filenames and captions)
        
    Returns:
        Dictionary containing all generated tables
    """
    print("="*70)
    print(f"Generating Tables from {attack_name.title()} Attack Results")
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

    # Coerce numeric fields stored as None/null to proper numbers (NaN)
    _numeric_fields = ('clean_accuracy', 'attack_success_rate',
                       'training_epochs', 'best_train_acc', 'best_val_loss',
                       'poison_rate', 'num_poisoned')
    for entry in results:
        for field in _numeric_fields:
            if field in entry and entry[field] is None:
                entry[field] = float('nan')

    # Log pretrained/scratch breakdown
    import pandas as _pd
    _df_info = _pd.DataFrame(results)
    if 'pretrained' in _df_info.columns:
        _pt_count = _df_info['pretrained'].sum()
        _sc_count = (~_df_info['pretrained']).sum()
        print(f"  Pretrained models: {_pt_count} entries, Scratch models: {_sc_count} entries")
        print("  Note: tables will show '<model>_pretrained' and '<model>_scratch' rows separately")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    attack_prefix = attack_name.lower().replace(' ', '_')
    
    # 1. Summary Table
    print("\n" + "-"*70)
    print("1. Summary Table")
    print("-"*70)
    summary_df = create_summary_table(results)
    print(summary_df.to_string(index=False))
    
    summary_path = os.path.join(save_dir, f'{attack_prefix}_summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved to: {summary_path}")
    
    # 2. Clean Accuracy Pivot Table
    print("\n" + "-"*70)
    print("2. Clean Accuracy by Model and Dataset")
    print("-"*70)
    clean_acc_pivot = create_pivot_table(results, metric='clean_accuracy')
    print(clean_acc_pivot.to_string())
    
    clean_acc_path = os.path.join(save_dir, f'{attack_prefix}_clean_accuracy_pivot_{timestamp}.csv')
    clean_acc_pivot.to_csv(clean_acc_path)
    print(f"\nSaved to: {clean_acc_path}")
    
    # 3. ASR Pivot Table
    print("\n" + "-"*70)
    print("3. Attack Success Rate by Model and Dataset")
    print("-"*70)
    asr_pivot = create_pivot_table(results, metric='attack_success_rate')
    print(asr_pivot.to_string())
    
    asr_path = os.path.join(save_dir, f'{attack_prefix}_asr_pivot_{timestamp}.csv')
    asr_pivot.to_csv(asr_path)
    print(f"\nSaved to: {asr_path}")
    
    # 4. Comparison Table
    print("\n" + "-"*70)
    print("4. Combined Comparison Table")
    print("-"*70)
    comparison_df = create_comparison_table(results)
    print(comparison_df.to_string())
    
    comparison_path = os.path.join(save_dir, f'{attack_prefix}_comparison_table_{timestamp}.csv')
    comparison_df.to_csv(comparison_path)
    print(f"\nSaved to: {comparison_path}")
    
    # 5. Ranking Tables
    print("\n" + "-"*70)
    print("5. Ranking by ASR (Descending)")
    print("-"*70)
    ranking_asr = create_ranking_table(results, metric='attack_success_rate', ascending=False)
    print(ranking_asr.to_string(index=False))
    
    ranking_asr_path = os.path.join(save_dir, f'{attack_prefix}_ranking_asr_{timestamp}.csv')
    ranking_asr.to_csv(ranking_asr_path, index=False)
    print(f"\nSaved to: {ranking_asr_path}")
    
    # 6. Statistics
    print("\n" + "-"*70)
    print("6. Overall Statistics")
    print("-"*70)
    stats = calculate_statistics(results)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    stats_df = pd.DataFrame([stats]).T
    stats_df.columns = ['Value']
    stats_path = os.path.join(save_dir, f'{attack_prefix}_statistics_{timestamp}.csv')
    stats_df.to_csv(stats_path)
    print(f"\nSaved to: {stats_path}")
    
    # 7. Per-Dataset Statistics
    print("\n" + "-"*70)
    print("7. Per-Dataset Statistics")
    print("-"*70)
    dataset_stats = create_per_dataset_stats(results)
    print(dataset_stats.to_string())
    
    dataset_stats_path = os.path.join(save_dir, f'{attack_prefix}_dataset_stats_{timestamp}.csv')
    dataset_stats.to_csv(dataset_stats_path)
    print(f"\nSaved to: {dataset_stats_path}")
    
    # 8. Per-Model Statistics
    print("\n" + "-"*70)
    print("8. Per-Model Statistics")
    print("-"*70)
    model_stats = create_per_model_stats(results)
    print(model_stats.to_string())
    
    model_stats_path = os.path.join(save_dir, f'{attack_prefix}_model_stats_{timestamp}.csv')
    model_stats.to_csv(model_stats_path)
    print(f"\nSaved to: {model_stats_path}")
    
    # 9. LaTeX Tables
    print("\n" + "-"*70)
    print("9. LaTeX Format Tables")
    print("-"*70)
    
    latex_dir = os.path.join(save_dir, 'latex')
    os.makedirs(latex_dir, exist_ok=True)
    
    # Clean Accuracy LaTeX
    latex_clean = print_latex_table(
        clean_acc_pivot, 
        caption=f"Clean Accuracy (\\%) - {attack_name.title()} Attack",
        label=f"tab:{attack_prefix}_clean_acc"
    )
    latex_clean_path = os.path.join(latex_dir, f'{attack_prefix}_clean_accuracy_{timestamp}.tex')
    with open(latex_clean_path, 'w') as f:
        f.write(latex_clean)
    print(f"Clean Accuracy LaTeX saved to: {latex_clean_path}")
    
    # ASR LaTeX
    latex_asr = print_latex_table(
        asr_pivot,
        caption=f"Attack Success Rate (\\%) - {attack_name.title()} Attack",
        label=f"tab:{attack_prefix}_asr"
    )
    latex_asr_path = os.path.join(latex_dir, f'{attack_prefix}_asr_{timestamp}.tex')
    with open(latex_asr_path, 'w') as f:
        f.write(latex_asr)
    print(f"ASR LaTeX saved to: {latex_asr_path}")
    
    # Comparison LaTeX
    latex_comparison = print_latex_table(
        comparison_df,
        caption=f"Comparison Table - {attack_name.title()} Attack",
        label=f"tab:{attack_prefix}_comparison"
    )
    latex_comparison_path = os.path.join(latex_dir, f'{attack_prefix}_comparison_{timestamp}.tex')
    with open(latex_comparison_path, 'w') as f:
        f.write(latex_comparison)
    print(f"Comparison LaTeX saved to: {latex_comparison_path}")
    
    # 10. Markdown Tables
    print("\n" + "-"*70)
    print("10. Markdown Format Tables")
    print("-"*70)
    
    markdown_dir = os.path.join(save_dir, 'markdown')
    os.makedirs(markdown_dir, exist_ok=True)
    
    md_content = f"# {attack_name.title()} Attack Results\n\n"
    md_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += print_markdown_table(clean_acc_pivot, "Clean Accuracy (%)")
    md_content += print_markdown_table(asr_pivot, "Attack Success Rate (%)")
    md_content += print_markdown_table(comparison_df, "Comparison Table")
    
    md_path = os.path.join(markdown_dir, f'{attack_prefix}_results_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Markdown tables saved to: {md_path}")
    
    print("\n" + "="*70)
    print("All tables generated successfully!")
    print(f"Tables saved to: {save_dir}")
    print("="*70)
    
    return {
        'summary': summary_df,
        'clean_accuracy_pivot': clean_acc_pivot,
        'asr_pivot': asr_pivot,
        'comparison': comparison_df,
        'ranking_asr': ranking_asr,
        'statistics': stats,
        'dataset_stats': dataset_stats,
        'model_stats': model_stats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate tables from backdoor ATTACK results (includes ASR metrics)'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to attack results JSON file')
    parser.add_argument('--save-dir', type=str, default='./results/tables/attack',
                       help='Directory to save tables (use results/tables/attack/<attack_type> for specific attacks)')
    parser.add_argument('--attack-name', type=str, default='backdoor',
                       help='Name of the attack for captions and filenames')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        exit(1)
    
    # Generate tables
    tables = generate_all_tables(args.results, args.save_dir, args.attack_name)
