"""
Generate tables from backdoor defense results (STRIP and Neural Cleanse).

Reads the per-model JSON files produced by run_defense.py / batch_run_defense.py
and produces formatted tables in CSV, LaTeX, and Markdown.

Output tables
-------------
For STRIP:
  - Summary: model × dataset  × attack  →  TPR, FPR, AUC, ASR-after-defense
  - Pivot (TPR):  model × dataset
  - Pivot (AUC):  model × dataset

For NC:
  - Summary: model × dataset × attack → detected, suspect class, correct identification
  - Pivot (detected %): model × dataset

If results from both STRIP and NC are mixed in one file, tables are separated
by defense type automatically.

Usage examples
--------------
# All BadNet defense results (STRIP + NC merged)
python scripts/visualization/generate_defense_tables.py \\
    --results results/defense/all_defenses_results.json \\
    --save-dir results/tables/defense

# STRIP results only for BadNet
python scripts/visualization/generate_defense_tables.py \\
    --results results/defense/badnet_strip_results.json \\
    --save-dir results/tables/defense/badnet \\
    --defense STRIP --attack badnet

# NC results for all attacks
python scripts/visualization/generate_defense_tables.py \\
    --results results/defense/all_defenses_results.json \\
    --save-dir results/tables/defense \\
    --defense NC
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    # Accept both a list of records and a single record
    if isinstance(data, dict):
        data = [data]
    return data


def _is_vit_deit_scratch(row: dict) -> bool:
    """Return True for scratch-trained ViT/DeiT entries."""
    model = str(row.get('model', '')).lower()
    if not (model.startswith('vit') or model.startswith('deit')):
        return False

    pretrained = row.get('pretrained')
    if pretrained is False:
        return True

    training_style = str(row.get('training_style', '')).lower()
    return training_style in {'scratch', 'vit_scratch'}


def filter_vit_deit_scratch(results: list[dict]) -> list[dict]:
    """Drop ViT/DeiT scratch rows to keep scope aligned with baseline."""
    return [r for r in results if not _is_vit_deit_scratch(r)]


def filter_by_defense(results: list[dict], defense: Optional[str]) -> list[dict]:
    if defense is None:
        return results
    return [r for r in results if r.get('defense', '').upper() == defense.upper()]


def filter_by_attack(results: list[dict], attack: Optional[str]) -> list[dict]:
    if attack is None:
        return results
    return [r for r in results if r.get('poison_type', '').lower() == attack.lower()]


# ══════════════════════════════════════════════════════════════════════════════
# STRIP tables
# ══════════════════════════════════════════════════════════════════════════════

def _strip_summary(results: list[dict]) -> pd.DataFrame:
    """
    Full row-per-model summary for STRIP.

    Columns are named to convey the mechanism:
    - Detection Rate / TPR: % of backdoored inputs caught by the entropy filter
    - False Alarm Rate / FPR: % of clean inputs wrongly rejected (utility cost)
    - Clean / Poison H Median: median normalised prediction entropy under superimposition
    - Entropy Gap: H(clean) - H(poison); quantifies how well STRIP separates the two
      distributions (larger gap = more reliable detection)
    - Threshold (H_low): entropy below which a sample is flagged as poisoned
    - Missed Backdoors: 100 - TPR; % of poisoned inputs that still reach the victim
    - ASR after STRIP: attack success rate among the inputs that were NOT caught
    - Clean Acc after STRIP: clean accuracy after the entropy-filtered inference pipeline
    """
    rows = []
    for r in results:
        clean_H  = r.get('clean_entropy_median',  float('nan'))
        poison_H = r.get('poison_entropy_median', float('nan'))
        tpr      = r.get('tpr', float('nan'))
        try:
            entropy_gap = round(float(clean_H) - float(poison_H), 4)
        except (TypeError, ValueError):
            entropy_gap = float('nan')
        try:
            missed_pct = round(100.0 - float(tpr), 2)
        except (TypeError, ValueError):
            missed_pct = float('nan')
        rows.append({
            'Dataset':                    r.get('dataset', ''),
            'Model':                      r.get('model', ''),
            'Training Style':             r.get('training_style', ''),
            'Attack':                     r.get('poison_type', ''),
            'Poison Rate':                r.get('poison_rate', float('nan')),
            'Detection Rate / TPR (%)':   tpr,
            'False Alarm Rate / FPR (%)': r.get('fpr', float('nan')),
            'AUC':                        r.get('auc', float('nan')),
            'Clean H Median':             clean_H,
            'Poison H Median':            poison_H,
            'Entropy Gap (H_c−H_p)':      entropy_gap,
            'Threshold H_low':            r.get('threshold_low', float('nan')),
            'Missed Backdoors (%)':       missed_pct,
            'ASR after STRIP (%)':        _pct(r.get('asr_after_defense')),
            'Clean Acc after STRIP (%)':  _pct(r.get('clean_accuracy_after_defense')),
        })
    df = pd.DataFrame(rows)
    _round_numeric(df, 3)
    return df


def _pct(v) -> float:
    """Convert [0,1] fraction or percentage to percentage, handling both."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float('nan')
    return round(float(v) * 100, 2) if float(v) <= 1.0 else round(float(v), 2)


def _strip_pivot(results: list[dict], metric: str, as_percentage: bool = False) -> pd.DataFrame:
    """Pivot table: model_style × (dataset, attack) → metric."""
    df = pd.DataFrame(results)
    if metric not in df.columns:
        return pd.DataFrame()
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    if as_percentage:
        df[metric] = df[metric].where(df[metric] > 1.0, df[metric] * 100.0)
    df['model_style'] = df.apply(
        lambda r: f"{r.get('model', '')}_{r.get('training_style', '')}", axis=1)
    pivot = df.pivot_table(
        index='model_style', columns=['dataset', 'poison_type'],
        values=metric, aggfunc='mean')
    pivot.columns = [f'{d}/{a}' for d, a in pivot.columns]
    pivot.index.name = 'Model'
    pivot = pivot.round(3)
    return pivot


# ══════════════════════════════════════════════════════════════════════════════
# NC tables
# ══════════════════════════════════════════════════════════════════════════════

def _nc_summary(results: list[dict]) -> pd.DataFrame:
    """
    Full row-per-model summary for Neural Cleanse.

    NC reconstructs a minimal trigger mask for each class and flags the class
    whose mask is an outlier (measured by the anomaly index = ratio of min norm
    to median of all other norms).  Key columns:
    - Detected: whether NC's anomaly index exceeded its threshold
    - Suspect Class: class NC flagged as backdoored
    - Correct ID: whether the suspect class matches the actual target class
    - Min Norm Class: class that required the smallest trigger mask (lower = more anomalous)
    """
    rows = []
    for r in results:
        detected = r.get('nc_detected', None)
        row = {
            'Dataset':              r.get('dataset', ''),
            'Model':                r.get('model', ''),
            'Training Style':       r.get('training_style', ''),
            'Attack':               r.get('poison_type', ''),
            'Poison Rate':          r.get('poison_rate', float('nan')),
            'Target Class':         r.get('target_class', ''),
            'Backdoor Detected':    'Yes' if detected else ('No' if detected is False else 'N/A'),
            'Suspect Class (NC)':   r.get('suspect_class', r.get('suspect_classes', '')),
            'Correct Target ID':    _bool_str(r.get('correctly_identified_target')),
            'Min Mask Norm Class':  r.get('min_norm_class', ''),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _bool_str(v) -> str:
    if v is None: return 'N/A'
    return 'Yes' if v else 'No'


def _nc_detection_pivot(results: list[dict]) -> pd.DataFrame:
    """Pivot: model_style × (dataset, attack) → detection rate (0/1 → 0% or 100%)."""
    df = pd.DataFrame(results)
    df['detected_num'] = df['nc_detected'].apply(
        lambda x: 1 if x is True else (0 if x is False else float('nan')))
    df['model_style'] = df.apply(
        lambda r: f"{r.get('model', '')}_{r.get('training_style', '')}", axis=1)
    if 'detected_num' not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index='model_style', columns=['dataset', 'poison_type'],
        values='detected_num', aggfunc='mean')
    pivot = pivot * 100  # to percentage
    pivot.columns = [f'{d}/{a}' for d, a in pivot.columns]
    pivot.index.name = 'Model'
    return pivot.round(0)


def _nc_correct_id_pivot(results: list[dict]) -> pd.DataFrame:
    """Pivot: model_style × (dataset, attack) → correct identification %."""
    df = pd.DataFrame(results)
    df['correct_num'] = df.get('correctly_identified_target', pd.Series(dtype=float)).apply(
        lambda x: 1 if x is True else (0 if x is False else float('nan')))
    df['model_style'] = df.apply(
        lambda r: f"{r.get('model', '')}_{r.get('training_style', '')}", axis=1)
    if 'correct_num' not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index='model_style', columns=['dataset', 'poison_type'],
        values='correct_num', aggfunc='mean')
    pivot = pivot * 100
    pivot.columns = [f'{d}/{a}' for d, a in pivot.columns]
    pivot.index.name = 'Model'
    return pivot.round(0)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _round_numeric(df: pd.DataFrame, decimals: int = 3) -> None:
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].round(decimals)


def _replace_missing_for_export(df: pd.DataFrame, missing_label: str = 'N/A') -> pd.DataFrame:
    export_df = df.copy()
    return export_df.astype(object).where(pd.notna(export_df), missing_label)


def save_table(df: pd.DataFrame, base_path: Path, fmt: str, title: str = '') -> None:
    if df.empty:
        print(f'  SKIP (empty): {base_path.name}')
        return
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'csv':
        _replace_missing_for_export(df).to_csv(str(base_path) + '.csv')
        print(f'  Saved: {base_path}.csv')
    elif fmt == 'latex':
        try:
            # Styler.to_latex() has native hrules support (\toprule/\midrule/\bottomrule)
            # hrules is a Styler parameter — NOT DataFrame.to_latex() — so use style API
            latex_str = df.style.format(precision=3, na_rep='N/A').to_latex(
                hrules=True,
                caption=title,
                label=f'tab:{base_path.stem}',
            )
        except Exception:
            # Fallback for very old pandas without Styler.to_latex()
            latex_str = df.to_latex(index=True, float_format='%.3f', na_rep='N/A',
                                     caption=title, label=f'tab:{base_path.stem}')
        with open(str(base_path) + '.tex', 'w', encoding='utf-8') as f:
            f.write(latex_str)
        print(f'  Saved: {base_path}.tex')
    elif fmt == 'markdown':
        with open(str(base_path) + '.md', 'w', encoding='utf-8') as f:
            if title:
                f.write(f'## {title}\n\n')
            f.write(_replace_missing_for_export(df).to_markdown(index=True))
            f.write('\n')
        print(f'  Saved: {base_path}.md')


def save_all_formats(df: pd.DataFrame, base_path: Path,
                     formats: list[str], title: str = '') -> None:
    for fmt in formats:
        save_table(df, base_path, fmt, title)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate tables from backdoor defense results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--results', required=True,
                        help='Path to the defense results JSON file')
    parser.add_argument('--save-dir', default='results/tables/defense',
                        help='Output directory (default: results/tables/defense)')
    parser.add_argument('--defense', default=None, choices=['STRIP', 'NC'],
                        help='Filter results to this defense only')
    parser.add_argument('--attack',  default=None,
                        choices=['badnet', 'WaNet', 'wanet', 'refool'],
                        help='Filter results to this attack only')
    parser.add_argument('--formats', nargs='+',
                        default=['csv', 'latex', 'markdown'],
                        choices=['csv', 'latex', 'markdown'],
                        help='Output formats (default: csv latex markdown)')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    all_results = load_results(args.results)

    before_filter = len(all_results)
    all_results = filter_vit_deit_scratch(all_results)
    removed = before_filter - len(all_results)
    if removed:
        print(f'Filtered out {removed} ViT/DeiT scratch entries to match baseline scope')

    if not all_results:
        sys.exit('No results found in the JSON file.')

    print(f'Loaded {len(all_results)} records from {args.results}')

    # ── normalise attack name casing ───────────────────────────────────────────
    for r in all_results:
        if r.get('poison_type', '').lower() == 'wanet':
            r['poison_type'] = 'WaNet'

    attack_filter  = args.attack
    defense_filter = args.defense

    # ── determine which defenses are present ───────────────────────────────────
    defenses_present = {r.get('defense', '').upper() for r in all_results}
    defenses_to_process = ([defense_filter.upper()] if defense_filter
                           else sorted(defenses_present & {'STRIP', 'NC'}))

    for defense in defenses_to_process:
        subset = filter_by_defense(all_results, defense)
        if attack_filter:
            subset = filter_by_attack(subset, attack_filter)
        if not subset:
            print(f'No records for defense={defense}' +
                  (f', attack={attack_filter}' if attack_filter else ''))
            continue

        print(f'\n── {defense} ({len(subset)} records) ──')
        tag = f'{attack_filter}_' if attack_filter else ''

        if defense == 'STRIP':
            # Summary table
            df_summary = _strip_summary(subset)
            save_all_formats(df_summary,
                             save_dir / f'{tag}strip_summary',
                             args.formats,
                             title=f'STRIP Defense Results — {attack_filter or "All Attacks"}')

            # Pivot: TPR
            df_tpr = _strip_pivot(subset, 'tpr', as_percentage=True)
            save_all_formats(df_tpr,
                             save_dir / f'{tag}strip_pivot_tpr',
                             args.formats,
                             title='STRIP — TPR (%) by Model and Attack')

            # Pivot: AUC
            df_auc = _strip_pivot(subset, 'auc')
            save_all_formats(df_auc,
                             save_dir / f'{tag}strip_pivot_auc',
                             args.formats,
                             title='STRIP — AUC by Model and Attack')

            # Pivot: ASR after defense
            df_asr = _strip_pivot(subset, 'asr_after_defense', as_percentage=True)
            save_all_formats(df_asr,
                             save_dir / f'{tag}strip_pivot_asr_after',
                             args.formats,
                             title='STRIP — ASR After Defense (%) by Model and Attack')

        else:  # NC
            # Summary table
            df_nc = _nc_summary(subset)
            save_all_formats(df_nc,
                             save_dir / f'{tag}nc_summary',
                             args.formats,
                             title=f'Neural Cleanse Defense Results — {attack_filter or "All Attacks"}')

            # Pivot: detection rate
            df_det = _nc_detection_pivot(subset)
            save_all_formats(df_det,
                             save_dir / f'{tag}nc_pivot_detection_rate',
                             args.formats,
                             title='NC — Detection Rate (%) by Model and Attack')

            # Pivot: correct identification rate
            df_cid = _nc_correct_id_pivot(subset)
            save_all_formats(df_cid,
                             save_dir / f'{tag}nc_pivot_correct_id',
                             args.formats,
                             title='NC — Correct Backdoor Class Identified (%) by Model and Attack')

    print(f'\nAll tables saved to: {save_dir}')


if __name__ == '__main__':
    main()
