"""
Generate figures from backdoor defense results (STRIP and Neural Cleanse).

Reads the defense results JSON produced by run_defense.py / batch_run_defense.py
and generates a comprehensive set of publication-quality plots.

Each figure is designed to be self-explanatory about the defense mechanism:
titles and axis labels describe *what the metric means mechanistically*, not
just its name, so readers can understand the plot without external context.

Figures produced
----------------
STRIP (entropy-based online detection)
  STRIP superimposes N random clean images on each test input and measures the
  prediction entropy.  Backdoored inputs stably predict the target class regardless
  of superimposition (low entropy); clean inputs vary (high entropy).  Any input
  with H < threshold_low is rejected as poisoned.

  1. Detection rate / TPR bar chart  — per model, grouped by dataset
  2. False alarm rate / FPR bar chart — per model, grouped by dataset
  3. Entropy separability AUC bar chart — per model, grouped by dataset
  4. Detection quality scatter (TPR vs FPR) — one point per run, ideal-region shaded
  5. Entropy comparison with threshold — clean H / poison H / H_low per model
  6. Entropy gap bar chart  — H(clean)−H(poison) per model, colour-coded by strength
  7. Detection breakdown  — stacked bar: caught vs missed backdoors per model
  8. ASR after STRIP heatmap  — model × dataset
  9. Detection rate heatmap (TPR) — model × dataset
 10. AUC heatmap — model × dataset
 11. Attack comparison: TPR across BadNet / WaNet / Refool (if mixed results)
 12. ROC operating-point scatter by attack type, one point per run

Neural Cleanse (trigger inversion)
  NC reverse-engineers the minimal pixel mask that redirects all inputs to each
  class.  A backdoored class requires an abnormally small mask (the trigger is
  already embedded); the anomaly index detects this outlier.

 13. Detection rate bar chart — per model × dataset
 14. Correct identification rate — per model × dataset
 15. Minimum trigger mask norm — box plot per model (lower = stronger anomaly signal)
 16. Detection rate heatmap: model × dataset

Usage examples
--------------
# All results (STRIP + NC together)
python scripts/visualization/generate_defense_figures.py \\
    --results results/defense/all_defenses_results.json \\
    --save-dir results/figures/defense

# STRIP only
python scripts/visualization/generate_defense_figures.py \\
    --results results/defense/badnet_strip_results.json \\
    --save-dir results/figures/defense/badnet \\
    --defense STRIP --attack badnet

# NC only, different colour palette
python scripts/visualization/generate_defense_figures.py \\
    --results results/defense/all_defenses_results.json \\
    --save-dir results/figures/defense \\
    --defense NC --palette Set2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns


ATTACK_PALETTE = sns.color_palette('tab10', 10)
DATASET_COLORS = {
    'cifar10': ATTACK_PALETTE[0],
    'gtsrb': ATTACK_PALETTE[1],
}
STRIP_COLORS = {
    'clean': ATTACK_PALETTE[0],
    'poison': ATTACK_PALETTE[1],
    'threshold': ATTACK_PALETTE[2],
    'detected': ATTACK_PALETTE[0],
    'missed': ATTACK_PALETTE[1],
}


# ══════════════════════════════════════════════════════════════════════════════
# Style helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_style(palette: str = 'tab10') -> None:
    sns.set_theme(style='whitegrid', font_scale=1.1, palette=palette)
    sns.set_palette(palette)
    plt.rcParams.update({
        'figure.dpi':    150,
        'savefig.dpi':   300,
        'figure.figsize': (14, 6),
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })


def save_fig(fig: Figure, path: Path, title: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f'  Saved: {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


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


def to_df(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    # Normalise poison_type casing
    if 'poison_type' in df.columns:
        df['poison_type'] = df['poison_type'].replace({'WaNet': 'WaNet', 'wanet': 'WaNet'})
    # Create combined model label
    if 'model' in df.columns and 'training_style' in df.columns:
        df['model_label'] = df['model'] + '\n(' + df['training_style'] + ')'
    elif 'model' in df.columns:
        df['model_label'] = df['model']
    return df


def split_by_defense(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strip_df = df[df['defense'].str.upper() == 'STRIP'].copy() if 'defense' in df.columns else pd.DataFrame()
    nc_df    = df[df['defense'].str.upper() == 'NC'].copy()    if 'defense' in df.columns else pd.DataFrame()
    return strip_df, nc_df


def _pct(series: pd.Series) -> pd.Series:
    """Convert values ≤1 from fraction to percentage, leave ≥1 as-is."""
    return series.where(series > 1.0, series * 100.0)


def _get_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Safe column getter — returns an empty float Series when the column is absent."""
    return df[col] if col in df.columns else pd.Series(dtype=float, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# STRIP figures
# ══════════════════════════════════════════════════════════════════════════════

def _grouped_bar(df: pd.DataFrame, metric: str, ylabel: str,
                 title: str, save_path: Path,
                 hue_col: str = 'dataset', ylim: tuple = (0, 100),
                 x_col: str = 'model_label') -> None:
    """Generic grouped bar chart with Wilson confidence intervals for proportions."""
    if metric not in df.columns or df[metric].isna().all():
        print(f'  SKIP (no data for {metric})')
        return
    col = pd.to_numeric(df[metric], errors='coerce')
    if col.max(skipna=True) <= 1.0:
        df = df.copy()
        df[metric] = col * 100.0
    else:
        df = df.copy()
        df[metric] = col

    def _wilson_from_samples(samples: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
        vals = np.asarray(samples, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return np.nan, np.nan, np.nan
        props = vals / 100.0
        props = np.clip(props, 0.0, 1.0)
        n = props.size
        p_hat = float(props.mean())
        z2 = z ** 2
        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2.0 * n)) / denom
        margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
        low = max(0.0, center - margin)
        high = min(1.0, center + margin)
        return p_hat * 100.0, low * 100.0, high * 100.0

    fig, ax = plt.subplots(figsize=(16, 6))
    x_order = list(df[x_col].dropna().unique()) if x_col in df.columns else []
    hue_order = None
    palette = None
    if hue_col in df.columns:
        hue_order = list(df[hue_col].dropna().unique())
        if hue_col == 'dataset':
            palette = [DATASET_COLORS.get(v, ATTACK_PALETTE[i % len(ATTACK_PALETTE)])
                       for i, v in enumerate(hue_order)]

    if hue_order is None:
        hue_order = ['all']
        df['_single_hue'] = 'all'
        hue_col = '_single_hue'

    if palette is None:
        palette = sns.color_palette('tab10', len(hue_order))

    n_hue = max(1, len(hue_order))
    total_width = 0.82
    bar_width = total_width / n_hue
    x_idx = np.arange(len(x_order))

    for h_i, h_val in enumerate(hue_order):
        means = []
        low_err = []
        high_err = []
        positions = x_idx - (total_width / 2) + (h_i + 0.5) * bar_width
        for x_val in x_order:
            group = df[(df[x_col] == x_val) & (df[hue_col] == h_val)][metric]
            mean, low, high = _wilson_from_samples(group.to_numpy(dtype=float))
            means.append(mean)
            low_err.append(mean - low if not np.isnan(mean) else np.nan)
            high_err.append(high - mean if not np.isnan(mean) else np.nan)

        bars = ax.bar(
            positions,
            means,
            width=bar_width,
            color=palette[h_i % len(palette)],
            alpha=0.85,
            label=h_val,
            edgecolor='white',
            linewidth=0.6,
        )

        for b_i, bar in enumerate(bars):
            m = means[b_i]
            if np.isnan(m):
                continue
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                m,
                yerr=np.array([[low_err[b_i]], [high_err[b_i]]]),
                fmt='none',
                ecolor='black',
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                zorder=4,
            )

    ax.set_xlabel('Model')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(v) for v in x_order], rotation=30, ha='right')
    ax.legend(title=hue_col.replace('_', ' ').title())
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, save_path, title)


def _heatmap(pivot: pd.DataFrame, title: str, save_path: Path,
             fmt: str = '.1f', cmap: str = 'YlOrRd',
             vmin: float = 0, vmax: float = 100) -> None:
    if pivot.empty:
        print(f'  SKIP (empty pivot): {save_path.name}')
        return
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 2), max(5, pivot.shape[0] * 0.7)))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax,
                cbar_kws={'label': title})
    ax.set_title(title, fontsize=13, pad=14)
    plt.tight_layout()
    save_fig(fig, save_path)


def _make_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()
    col = pd.to_numeric(df[metric], errors='coerce')
    df = df.copy()
    df[metric] = col.where(col > 1.0, col * 100.0)
    pivot = df.pivot_table(index='model_label',
                           columns=['dataset', 'poison_type'],
                           values=metric, aggfunc='mean')
    pivot.columns = [f'{d}/{a}' for d, a in pivot.columns]
    return pivot.round(1)


def generate_strip_figures(strip_df: pd.DataFrame, save_dir: Path,
                           attack_filter: Optional[str]) -> None:
    tag = f'{attack_filter}_' if attack_filter else ''

    # 1. TPR bar chart — primary detection metric
    _grouped_bar(strip_df, 'tpr', 'Detection Rate / TPR (%)',
                 'STRIP - Backdoor Detection Rate (TPR) per Model',
                 save_dir / f'{tag}strip_tpr_bar.png')

    # 2. FPR bar chart — utility cost of defense
    _grouped_bar(strip_df, 'fpr', 'False Alarm Rate / FPR (%)',
                 'STRIP - False Alarm Rate (FPR) per Model',
                 save_dir / f'{tag}strip_fpr_bar.png',
                 ylim=(0, 30))

    # 3. AUC bar chart — entropy separability quality
    _grouped_bar(strip_df.assign(
                     auc_pct=pd.to_numeric(_get_col(strip_df, 'auc'), errors='coerce') * 100),
                 'auc_pct', 'AUC (× 100)',
                 'STRIP - Entropy Separability AUC per Model',
                 save_dir / f'{tag}strip_auc_bar.png')

    # 4. TPR vs FPR scatter — detection quality trade-off
    _strip_tpr_fpr_scatter(strip_df, save_dir / f'{tag}strip_tpr_fpr_scatter.png')

    # 5. Entropy comparison with detection threshold — core mechanism
    _strip_entropy_comparison(strip_df, save_dir / f'{tag}strip_entropy_comparison.png')

    # 6. Entropy gap bar chart — separation strength per model (new)
    _strip_entropy_gap_bar(strip_df, save_dir / f'{tag}strip_entropy_gap.png')

    # 7. Detection breakdown: caught vs missed backdoors (new)
    _strip_detection_breakdown(strip_df, save_dir / f'{tag}strip_detection_breakdown.png')

    # 8. ASR after STRIP heatmap
    _strip_asr_comparison_heatmap(strip_df, save_dir / f'{tag}strip_asr_comparison.png')

    # 9. TPR heatmap
    _heatmap(_make_pivot(strip_df, 'tpr'),
             'STRIP - Detection Rate / TPR (%) by Model and Attack',
             save_dir / f'{tag}strip_tpr_heatmap.png',
             cmap='RdYlGn')

    # 10. AUC heatmap
    auc_pivot = _make_pivot(
        strip_df.assign(auc_pct=pd.to_numeric(_get_col(strip_df, 'auc'), errors='coerce') * 100),
        'auc_pct')
    _heatmap(auc_pivot,
             'STRIP - Entropy Separability AUC by Model and Attack',
             save_dir / f'{tag}strip_auc_heatmap.png',
             cmap='RdYlGn', vmin=50, vmax=100)

    # 11. Attack comparison (only when results span multiple attacks)
    if strip_df.get('poison_type') is not None:
        attacks = strip_df['poison_type'].dropna().unique()
        if len(attacks) > 1:
            _strip_attack_comparison(strip_df, save_dir / f'{tag}strip_attack_comparison.png')

    # 12. ROC operating-point scatter
    _strip_roc_scatter(strip_df, save_dir / f'{tag}strip_roc_scatter.png')


def _strip_tpr_fpr_scatter(df: pd.DataFrame, path: Path) -> None:
    """
    Scatter of STRIP operating points (FPR, TPR) per model × dataset run.
    Top-left quadrant = ideal: many backdoors caught, few clean inputs rejected.
    The diagonal = random-classifier baseline (no separability).
    """
    tpr = pd.to_numeric(_get_col(df, 'tpr'), errors='coerce')
    fpr = pd.to_numeric(_get_col(df, 'fpr'), errors='coerce')
    if tpr.isna().all() or fpr.isna().all():
        return
    df = df.copy()
    df['tpr_v'] = tpr.where(tpr > 1.0, tpr * 100.0)
    df['fpr_v'] = fpr.where(fpr > 1.0, fpr * 100.0)

    fig, ax = plt.subplots(figsize=(8, 8))

    datasets = df['dataset'].unique() if 'dataset' in df.columns else ['all']
    palette = [DATASET_COLORS.get(d, ATTACK_PALETTE[i % len(ATTACK_PALETTE)])
               for i, d in enumerate(datasets)]
    colour_map = dict(zip(datasets, palette))

    for _, row in df.iterrows():
        c = colour_map.get(row.get('dataset', 'all'), 'grey')
        ax.scatter(row['fpr_v'], row['tpr_v'], color=c, s=90, alpha=0.85, zorder=3,
                   edgecolors='white', linewidths=0.7)
        ax.annotate(row.get('model', ''), (row['fpr_v'], row['tpr_v']),
                    fontsize=7.5, xytext=(5, 4), textcoords='offset points', color='dimgray')

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.25, linewidth=1.2,
            label='Random baseline (no entropy separation)')
    ax.set_xlabel('False Alarm Rate / FPR (%) — clean inputs wrongly rejected', fontsize=10)
    ax.set_ylabel('Detection Rate / TPR (%) — backdoored inputs caught', fontsize=10)
    ax.set_title('STRIP - Detection Quality: Backdoor Caught vs False Alarms', fontsize=11)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    patches = [mpatches.Patch(color=c, label=d) for d, c in colour_map.items()]
    ax.legend(handles=patches, title='Dataset')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_fig(fig, path)


def _strip_entropy_comparison(df: pd.DataFrame, path: Path) -> None:
    """
    STRIP core mechanism figure.

    STRIP works by superimposing N random images on each test input and measuring
    prediction entropy.  Backdoored inputs produce consistently low entropy
    (the trigger overwhelms the superimposed content → stuck prediction).
    Clean inputs produce high entropy (predictions vary with different images).
    Any input with H < threshold_low is flagged as poisoned.

    This figure shows the median entropy of clean and poison inputs together
    with the detection threshold, making the separation (or lack thereof) visible.
    """
    clean  = pd.to_numeric(_get_col(df, 'clean_entropy_median'), errors='coerce')
    poison = pd.to_numeric(_get_col(df, 'poison_entropy_median'), errors='coerce')
    thresh = pd.to_numeric(_get_col(df, 'threshold_low'),         errors='coerce')
    if clean.isna().all() and poison.isna().all():
        return
    df = df.copy()
    df['clean_H']  = clean
    df['poison_H'] = poison
    df['thresh']   = thresh
    labels = df['model_label'].tolist() if 'model_label' in df.columns else list(range(len(df)))
    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 6))

    bars_c = ax.bar(x - w/2, df['clean_H'].fillna(0),  w,
                    label='Clean H median  (→ high = hard to flag)',
                    alpha=0.85, color=STRIP_COLORS['clean'], edgecolor='white')
    bars_p = ax.bar(x + w/2, df['poison_H'].fillna(0), w,
                    label='Poison H median  (→ low = easy to flag)',
                    alpha=0.85, color=STRIP_COLORS['poison'], edgecolor='white')

    # Threshold markers and gap annotations
    if not thresh.isna().all():
        ax.scatter(x, df['thresh'], marker='_', color=STRIP_COLORS['threshold'], s=500, zorder=5,
                   linewidths=3.0, label='Detection threshold H_low\n(samples below → flagged as poisoned)')
        for xi, (t, p, c) in enumerate(
                zip(df['thresh'].fillna(0), df['poison_H'].fillna(0), df['clean_H'].fillna(0))):
            # Gap label between clean and poison bars
            gap = c - p
            if gap > 0.05:
                ax.text(xi, max(c, t) + 0.02, f'Δ={gap:.2f}',
                        ha='center', fontsize=7.5, color='dimgray', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Normalised Prediction Entropy H(·)')
    ax.set_title('STRIP - Entropy Separation: Clean vs Backdoored Inputs')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, path)


def _strip_entropy_gap_bar(df: pd.DataFrame, path: Path) -> None:
    """
    Bar chart of the entropy gap H(clean) − H(poison) per model.

    This single number captures how well STRIP separates the two distributions.
    A large positive gap means the threshold can sit comfortably between them →
    high TPR and low FPR.  A gap near zero means the distributions overlap and
    STRIP struggles to discriminate (low AUC, degraded TPR/FPR trade-off).
    """
    clean  = pd.to_numeric(_get_col(df, 'clean_entropy_median'), errors='coerce')
    poison = pd.to_numeric(_get_col(df, 'poison_entropy_median'), errors='coerce')
    if clean.isna().all() or poison.isna().all():
        return
    df = df.copy()
    df['entropy_gap'] = clean - poison
    labels = df['model_label'].tolist() if 'model_label' in df.columns else list(range(len(df)))
    x = np.arange(len(labels))

    colours = [
        ATTACK_PALETTE[2] if g > 0.3 else (ATTACK_PALETTE[0] if g > 0.1 else ATTACK_PALETTE[1])
        for g in df['entropy_gap'].fillna(0)
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 5))
    ax.bar(x, df['entropy_gap'].fillna(0), color=colours, alpha=0.87, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Entropy Gap  H(clean) − H(poison)')
    ax.set_title('STRIP - Entropy Separation Gap: H(clean) - H(poison) per Model')
    ax.axhline(0,   color='black',       linewidth=0.9, linestyle='-',  alpha=0.5)
    ax.axhline(0.3, color=ATTACK_PALETTE[2], linewidth=1.0, linestyle='--', alpha=0.45,
               label='Strong separation threshold (0.3)')
    ax.axhline(0.1, color=ATTACK_PALETTE[0], linewidth=1.0, linestyle='--', alpha=0.45,
               label='Moderate separation threshold (0.1)')
    legend_handles = [
        mpatches.Patch(color=ATTACK_PALETTE[2], label='Gap > 0.3  (strong — reliable detection)'),
        mpatches.Patch(color=ATTACK_PALETTE[0], label='Gap 0.1–0.3  (moderate)'),
        mpatches.Patch(color=ATTACK_PALETTE[1], label='Gap < 0.1  (weak — distributions overlap)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, path)


def _strip_detection_breakdown(df: pd.DataFrame, path: Path) -> None:
    """
    Stacked bar chart showing what fraction of backdoored inputs STRIP catches
    vs what slips through, per model.

    This directly answers the attacker's question: "How many of my poisoned
    inputs still reach the victim after STRIP is deployed?"
    Green = detected and blocked | Red = missed — still reach the victim.
    """
    tpr = pd.to_numeric(_get_col(df, 'tpr'), errors='coerce')
    if tpr.isna().all():
        return

    df = df.copy()
    df['tpr_v'] = tpr.where(tpr > 1.0, tpr * 100.0)
    df['missed_v'] = 100.0 - df['tpr_v']
    labels = df['model_label'].tolist() if 'model_label' in df.columns else list(range(len(df)))
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 5))
    ax.bar(
        x,
        df['tpr_v'],
        color=STRIP_COLORS['detected'],
        alpha=0.87,
        edgecolor='white',
        label='Detected & blocked by STRIP (TPR)',
    )
    ax.bar(
        x,
        df['missed_v'],
        bottom=df['tpr_v'],
        color=STRIP_COLORS['missed'],
        alpha=0.80,
        edgecolor='white',
        label='Missed - still reach victim (100 - TPR)',
    )

    for xi, (det, miss) in enumerate(zip(df['tpr_v'].fillna(0), df['missed_v'].fillna(0))):
        if det > 8:
            ax.text(
                xi,
                det / 2,
                f'{det:.1f}%',
                ha='center',
                va='center',
                fontsize=8,
                color='white',
                fontweight='bold',
            )
        if miss > 8:
            ax.text(
                xi,
                det + miss / 2,
                f'{miss:.1f}%',
                ha='center',
                va='center',
                fontsize=8,
                color='white',
                fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Backdoored Inputs (%)')
    ax.set_ylim(0, 110)
    ax.set_title('STRIP - Backdoor Detection Breakdown per Model')
    ax.axhline(50, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, path)


def _strip_asr_comparison_heatmap(df: pd.DataFrame, path: Path) -> None:
    """
    Heatmap of Attack Success Rate remaining after STRIP filters out detected inputs.
    Low values are desirable: STRIP has neutralised the backdoor attack.
    """
    if 'asr_after_defense' not in df.columns:
        return
    df2 = df.copy()
    df2['asr_after'] = pd.to_numeric(df2['asr_after_defense'], errors='coerce')
    df2['asr_after'] = df2['asr_after'].where(df2['asr_after'] > 1.0, df2['asr_after'] * 100.0)
    if df2['asr_after'].isna().all():
        return
    pivot = df2.pivot_table(index='model_label', columns=['dataset', 'poison_type'],
                            values='asr_after', aggfunc='mean')
    pivot.columns = [f'{d}/{a}' for d, a in pivot.columns]
    _heatmap(pivot.round(1),
             'STRIP - Attack Success Rate After Defense (%)',
             path, cmap='YlOrRd', vmin=0, vmax=100)


def _strip_attack_comparison(df: pd.DataFrame, path: Path) -> None:
    """
    Bar chart comparing STRIP detection rate (TPR) across different attack types.
    Shows whether STRIP's entropy-based mechanism generalises to attacks other than
    BadNet (e.g. WaNet's warping may produce different entropy profiles).
    """
    df2 = df.copy()
    tpr = pd.to_numeric(_get_col(df2, 'tpr'), errors='coerce')
    df2['tpr_pct'] = tpr.where(tpr > 1.0, tpr * 100.0)
    _grouped_bar(
        df2,
        'tpr_pct',
        'Detection Rate / TPR (%)',
        'STRIP - Detection Rate (TPR) Across Attack Types',
        path,
        hue_col='dataset',
        ylim=(0, 110),
        x_col='poison_type',
    )


def _strip_roc_scatter(df: pd.DataFrame, path: Path) -> None:
    """
    Scatter of STRIP operating points (FPR, TPR), coloured by attack type and
    shaped by dataset.  Each point represents one model × dataset × attack run.
    The dotted green path marks the ideal classifier (top-left corner).
    Points clustered near the top-left indicate that STRIP works well for that
    attack; clusters near the diagonal indicate near-random entropy separation.
    """
    tpr = pd.to_numeric(_get_col(df, 'tpr'), errors='coerce')
    fpr = pd.to_numeric(_get_col(df, 'fpr'), errors='coerce')
    if tpr.isna().all() or fpr.isna().all():
        return
    df = df.copy()
    df['tpr_v'] = tpr.where(tpr > 1.0, tpr * 100.0)
    df['fpr_v'] = fpr.where(fpr > 1.0, fpr * 100.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    attacks  = df['poison_type'].dropna().unique() if 'poison_type' in df.columns else ['all']
    palette  = sns.color_palette('tab10', len(attacks))
    colours  = dict(zip(attacks, palette))
    markers  = dict(zip(df['dataset'].unique() if 'dataset' in df.columns else ['all'],
                        ['o', 's', '^', 'D', 'v']))

    for _, row in df.iterrows():
        c = colours.get(row.get('poison_type', 'all'), 'grey')
        m = markers.get(row.get('dataset', 'all'), 'o')
        ax.scatter(row['fpr_v'], row['tpr_v'], color=c, marker=m, s=100, alpha=0.85, zorder=3,
                   edgecolors='white', linewidths=0.6)

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.25, linewidth=1.2,
            label='Random baseline — no entropy separation')
    ax.plot([0, 0, 100], [0, 100, 100], 'g:', alpha=0.35, linewidth=1.5,
            label='Perfect detector')
    ax.set_xlabel('False Alarm Rate / FPR (%) — clean inputs wrongly rejected', fontsize=10)
    ax.set_ylabel('Detection Rate / TPR (%) — backdoored inputs caught', fontsize=10)
    ax.set_title('STRIP - Operating Points by Attack Type (FPR vs TPR)', fontsize=11)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    legend_patches = [mpatches.Patch(color=c, label=a) for a, c in colours.items()]
    ax.legend(handles=legend_patches, title='Attack Type')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_fig(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# NC figures
# ══════════════════════════════════════════════════════════════════════════════

def generate_nc_figures(nc_df: pd.DataFrame, save_dir: Path,
                        attack_filter: Optional[str]) -> None:
    tag = f'{attack_filter}_' if attack_filter else ''

    # 11. Detection rate bar chart
    _nc_detection_bar(nc_df, save_dir / f'{tag}nc_detection_rate_bar.png')

    # 12. Correct identification rate
    _nc_correct_id_bar(nc_df, save_dir / f'{tag}nc_correct_id_bar.png')

    # 13. Mask norm distribution (if available)
    _nc_mask_norm_comparison(nc_df, save_dir / f'{tag}nc_mask_norm_comparison.png')

    # 14. NC detection heatmap
    _nc_detection_heatmap(nc_df, save_dir / f'{tag}nc_detection_heatmap.png')


def _nc_detection_bar(df: pd.DataFrame, path: Path) -> None:
    df2 = df.copy()
    df2['detected_num'] = df2['nc_detected'].apply(
        lambda x: 100.0 if x is True else (0.0 if x is False else float('nan')))
    _grouped_bar(df2, 'detected_num', 'Detection Rate (%)',
                 'Neural Cleanse — Backdoor Detection Rate\n'
                 '(NC reverse-engineers triggers for each class; anomaly index flags the backdoor class;\n'
                 '100% = anomaly detected, 0% = NC found no anomalous class)',
                 path, ylim=(0, 110))


def _nc_correct_id_bar(df: pd.DataFrame, path: Path) -> None:
    if 'correctly_identified_target' not in df.columns:
        return
    df2 = df.copy()
    df2['cid_num'] = df2['correctly_identified_target'].apply(
        lambda x: 100.0 if x is True else (0.0 if x is False else float('nan')))
    _grouped_bar(df2, 'cid_num', 'Correct ID Rate (%)',
                 'Neural Cleanse — Correct Backdoor Class Identification\n'
                 '(of models where NC triggered, what fraction correctly identified the actual target class)',
                 path, ylim=(0, 110))


def _nc_mask_norm_comparison(df: pd.DataFrame, path: Path) -> None:
    """
    Box / strip plot of minimum mask norm per model (lower = more anomalous).
    Neural Cleanse reconstructs a minimal mask that fools the network into the
    target class for each candidate class.  For a backdoored class, the required
    mask is abnormally small (the backdoor pattern already exists).  The anomaly
    index compares this minimum norm to the median of all other classes — a large
    ratio flags a backdoor.  This chart shows the raw minimum norms across models.
    """
    if 'mask_norms' not in df.columns:
        return
    rows = []
    for _, r in df.iterrows():
        norms = r.get('mask_norms')
        if not isinstance(norms, list) or len(norms) == 0:
            continue
        min_norm = min(norms)
        rows.append({
            'model_label': r.get('model_label', r.get('model', '')),
            'dataset':     r.get('dataset', ''),
            'poison_type': r.get('poison_type', ''),
            'min_mask_norm': min_norm,
        })
    if not rows:
        return
    df2 = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df2, x='model_label', y='min_mask_norm', hue='dataset',
                ax=ax, palette='Set2')
    ax.set_xlabel('Model')
    ax.set_ylabel('Minimum Mask Norm (L₁)')
    ax.set_title('Neural Cleanse — Minimum Trigger Mask Norm per Model\n'
                 'NC finds the smallest mask that redirects all inputs to each class — '
                 'a backdoored class needs an abnormally small mask (the trigger is already embedded)\n'
                 'Lower value = stronger anomaly signal = stronger evidence of backdoor')
    ax.tick_params(axis='x', rotation=35)
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, path)


def _nc_detection_heatmap(df: pd.DataFrame, path: Path) -> None:
    df2 = df.copy()
    df2['detected_num'] = df2['nc_detected'].apply(
        lambda x: 100.0 if x is True else (0.0 if x is False else float('nan')))
    if 'model_label' not in df2.columns:
        return
    pivot = df2.pivot_table(index='model_label',
                            columns=['dataset', 'poison_type'],
                            values='detected_num', aggfunc='mean')
    pivot.columns = [f'{d}/{a}' for d, a in pivot.columns]
    _heatmap(pivot.round(0),
             'Neural Cleanse — Backdoor Detection Rate (%) by Model and Attack\n'
             '(100 = NC\'s anomaly index flagged a suspicious class; 0 = no anomaly found; ↑ better)',
             path, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=100)


# ══════════════════════════════════════════════════════════════════════════════
# Combined figures
# ══════════════════════════════════════════════════════════════════════════════

def generate_combined_figures(strip_df: pd.DataFrame, nc_df: pd.DataFrame,
                              save_dir: Path) -> None:
    """Cross-defense comparison figures (only when both STRIP and NC results exist)."""
    if strip_df.empty or nc_df.empty:
        return

    # Side-by-side detection performance
    _combined_detection_bar(strip_df, nc_df, save_dir / 'combined_detection_comparison.png')


def _combined_detection_bar(strip_df: pd.DataFrame, nc_df: pd.DataFrame,
                              path: Path) -> None:
    """Compare STRIP TPR vs NC detection rate side by side."""
    tpr_col  = pd.to_numeric(_get_col(strip_df, 'tpr'), errors='coerce')
    s_data   = strip_df.assign(
        metric=tpr_col.where(tpr_col > 1.0, tpr_col * 100.0),
        defense='STRIP — entropy test\n(flags low-H inputs)',
    )[['model_label', 'dataset', 'poison_type', 'metric', 'defense']]

    nc_data2 = nc_df.copy()
    nc_data2['metric'] = nc_data2['nc_detected'].apply(
        lambda x: 100.0 if x is True else (0.0 if x is False else float('nan')))
    nc_data2['defense'] = 'Neural Cleanse — mask anomaly\n(inverts triggers, flags small masks)'
    n_data = nc_data2[['model_label', 'dataset', 'poison_type', 'metric', 'defense']]

    combined = pd.concat([s_data, n_data], ignore_index=True)
    _grouped_bar(
        combined,
        'metric',
        'Detection Rate / TPR (%)',
        'Defense Comparison - STRIP vs Neural Cleanse',
        path,
        hue_col='defense',
        ylim=(0, 110),
        x_col='model_label',
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate figures from backdoor defense results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--results', required=True,
                        help='Path to the defense results JSON')
    parser.add_argument('--save-dir', default='results/figures/defense',
                        help='Output directory for figures (default: results/figures/defense)')
    parser.add_argument('--defense', default=None, choices=['STRIP', 'NC'],
                        help='Generate figures for this defense only')
    parser.add_argument('--attack',  default=None,
                        choices=['badnet', 'WaNet', 'wanet', 'refool'],
                        help='Filter results to this attack only')
    parser.add_argument('--palette', default='tab10',
                        help='Seaborn / matplotlib colour palette (default: tab10)')
    args = parser.parse_args()

    set_style(args.palette)

    save_dir = Path(args.save_dir)
    all_results = load_results(args.results)
    if not all_results:
        sys.exit('No results found.')

    print(f'Loaded {len(all_results)} records from {args.results}')

    # Filter
    if args.attack:
        low = args.attack.lower()
        all_results = [r for r in all_results
                       if r.get('poison_type', '').lower() == low]

    before_filter = len(all_results)
    all_results = filter_vit_deit_scratch(all_results)
    removed = before_filter - len(all_results)
    if removed:
        print(f'Filtered out {removed} ViT/DeiT scratch entries to match baseline scope')

    df = to_df(all_results)
    strip_df, nc_df = split_by_defense(df)

    if args.defense:
        if args.defense.upper() == 'STRIP':
            nc_df = pd.DataFrame()
        else:
            strip_df = pd.DataFrame()

    print(f'STRIP records: {len(strip_df)}  |  NC records: {len(nc_df)}')

    if not strip_df.empty:
        print('\n── Generating STRIP figures ──')
        generate_strip_figures(strip_df, save_dir, args.attack)

    if not nc_df.empty:
        print('\n── Generating NC figures ──')
        generate_nc_figures(nc_df, save_dir, args.attack)

    if not strip_df.empty and not nc_df.empty:
        print('\n── Generating combined figures ──')
        generate_combined_figures(strip_df, nc_df, save_dir)

    print(f'\nAll figures saved to: {save_dir}')


if __name__ == '__main__':
    main()
