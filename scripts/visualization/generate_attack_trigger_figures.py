"""
Visualise BadNet (and other) backdoor triggers.

For each dataset (CIFAR-10, GTSRB) produces a figure with a grid of sample
images shown in two rows:
  Row 1 – clean images (no trigger)
  Row 2 – the same images with the backdoor trigger applied

Output: results/figures/attack_triggers/trigger_examples_<dataset>_<attack>.png
"""

import os
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from torchvision import transforms

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLBOX_DIR  = PROJECT_ROOT / 'backdoor-toolbox-main'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(TOOLBOX_DIR))
sys.path.insert(0, str(TOOLBOX_DIR / 'utils'))

from datasets import get_dataloader

_supervisor  = importlib.import_module('supervisor')
_tb_config   = importlib.import_module('config')
# triggers_dir must be absolute — supervisor.get_poison_transform resolves trigger
# PNGs via config.triggers_dir, which defaults to './triggers' (relative to cwd).
setattr(_tb_config, 'triggers_dir', str(TOOLBOX_DIR / 'triggers'))

# ── normalisation stats ────────────────────────────────────────────────────────
STATS = {
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std':  [0.247,  0.243,  0.261],
        'size': 32,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'],
    },
    'gtsrb': {
        'mean': [0.3337, 0.3064, 0.3171],
        'std':  [0.2672, 0.2564, 0.2629],
        'size': 32,
        'classes': None,   # 43 classes — show numeric label
    },
}

DATASET_LABELS = {
    'cifar10': 'CIFAR-10',
    'gtsrb':   'GTSRB',
}


# ── helpers ────────────────────────────────────────────────────────────────────
def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Convert a normalised CHW tensor → HWC uint8 numpy array."""
    t = tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_transforms(dataset: str):
    s = STATS[dataset]
    return transforms.Compose([
        transforms.Resize((s['size'], s['size'])),
        transforms.ToTensor(),
        transforms.Normalize(s['mean'], s['std']),
    ])


def get_class_name(dataset: str, label: int) -> str:
    classes = STATS[dataset]['classes']
    if classes is not None:
        return classes[label]
    return f'class {label}'


# ── figure builder ─────────────────────────────────────────────────────────────
def make_trigger_figure(dataset: str, poison_type: str, n_samples: int,
                        data_dir: str, output_dir: Path, seed: int = 0,
                        poison_rate: float = 0.05, cover_rate: float = 0.1):
    """
    Build and save a side-by-side clean vs. triggered image grid.
    Supports BadNet-style patch attacks and WaNet warp-based attacks.
    """
    print(f"\n[{DATASET_LABELS[dataset]}]  attack={poison_type}  n={n_samples}")

    target_class = _tb_config.target_class[dataset]

    # ── poison transform ───────────────────────────────────────────────────────
    trigger_transform = get_transforms(dataset)
    try:
        if poison_type.lower() == 'wanet':
            # WaNet loads identity_grid/noise_grid via relative paths from TOOLBOX_DIR
            import argparse as _argparse
            _tb_args = _argparse.Namespace(
                poison_type='WaNet',
                poison_rate=poison_rate,
                cover_rate=cover_rate,
                dataset=dataset,
                trigger='none',
                alpha=0.2,
            )
            _orig_dir = os.getcwd()
            try:
                os.chdir(str(TOOLBOX_DIR))
                poison_transform = _supervisor.get_poison_transform(
                    poison_type='WaNet',
                    dataset_name=dataset,
                    target_class=target_class,
                    trigger_transform=trigger_transform,
                    is_normalized_input=True,
                    args=_tb_args,
                )
            finally:
                os.chdir(_orig_dir)
        elif poison_type.lower() == 'refool':
            # refool uses a cwd-relative path for VOC reflection images
            _orig_dir = os.getcwd()
            try:
                os.chdir(str(TOOLBOX_DIR))
                poison_transform = _supervisor.get_poison_transform(
                    poison_type='refool',
                    dataset_name=dataset,
                    target_class=target_class,
                    trigger_transform=trigger_transform,
                    is_normalized_input=True,
                )
            finally:
                os.chdir(_orig_dir)
        else:
            poison_transform = _supervisor.get_poison_transform(
                poison_type=poison_type,
                dataset_name=dataset,
                target_class=target_class,
                trigger_transform=trigger_transform,
                is_normalized_input=True,
            )
    except Exception as e:
        print(f"  Could not build poison_transform: {e}")
        print("  Skipping.")
        return

    # ── data ───────────────────────────────────────────────────────────────────
    loader = get_dataloader(
        dataset_name=dataset,
        data_dir=data_dir,
        batch_size=n_samples * 4,   # grab a bigger batch to find diverse classes
        train=False,
        shuffle=True,
        num_workers=0,
        download=False,
        augmentation=False,
    )

    torch.manual_seed(seed)
    images_all, labels_all = next(iter(loader))

    # try to pick one image per class for variety
    selected_idx = []
    seen_classes = set()
    for i in range(len(labels_all)):
        c = labels_all[i].item()
        if c not in seen_classes and c != target_class:
            selected_idx.append(i)
            seen_classes.add(c)
        if len(selected_idx) == n_samples:
            break
    # fall back to first n_samples if not enough classes
    while len(selected_idx) < n_samples:
        selected_idx.append(len(selected_idx))

    images = images_all[selected_idx]   # (N, C, H, W)
    labels = labels_all[selected_idx]

    # ── apply triggers ─────────────────────────────────────────────────────────
    poisoned_images, _ = poison_transform.transform(
        images.clone(), labels.clone()
    )

    # ── build figure ───────────────────────────────────────────────────────────
    mean = STATS[dataset]['mean']
    std  = STATS[dataset]['std']

    fig = plt.figure(figsize=(n_samples * 2.2, 5.5))
    fig.patch.set_facecolor('#1a1a2e')

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        hspace=0.05,
        top=0.80, bottom=0.04,
    )

    row_labels   = ['Clean', f'Poisoned\n(target → {get_class_name(dataset, target_class)})']
    row_colors   = ['#4CAF50', '#F44336']
    row_images   = [images, poisoned_images]

    for row in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_samples,
            subplot_spec=outer[row],
            wspace=0.04,
        )
        for col in range(n_samples):
            ax = fig.add_subplot(inner[col])
            img = denormalize(row_images[row][col], mean, std)
            ax.imshow(img, interpolation='nearest')
            ax.axis('off')

            if row == 0:
                class_name = get_class_name(dataset, labels[col].item())
                ax.set_title(class_name, fontsize=7.5, color='white',
                             pad=3, fontweight='bold')

        # row label on the left
        ax0 = fig.add_subplot(outer[row])
        ax0.axis('off')
        ax0.text(
            -0.01, 0.5, row_labels[row],
            transform=ax0.transAxes,
            va='center', ha='right',
            fontsize=9, color=row_colors[row],
            fontweight='bold',
        )

    # ── title / subtitle ───────────────────────────────────────────────────────
    fig.text(
        0.5, 0.93,
        f'Backdoor Trigger Visualisation — {DATASET_LABELS[dataset]}',
        ha='center', va='center',
        fontsize=14, fontweight='bold', color='white',
    )
    _pr_str = f'{poison_rate*100:.0f}%'
    _cr_str = f'  cover rate: {cover_rate*100:.0f}%' if poison_type.lower() == 'wanet' else ''
    fig.text(
        0.5, 0.865,
        f'Attack: {poison_type}  |  Poison rate: {_pr_str}{_cr_str}  |  Target class: {get_class_name(dataset, target_class)} ({target_class})',
        ha='center', va='center',
        fontsize=9.5, color='#aaaacc',
    )

    # dividing line between rows
    fig.add_artist(
        Line2D([0.05, 0.95], [0.49, 0.49],
               transform=fig.transFigure,
               color='#555577', linewidth=0.8, linestyle='--')
    )

    # ── save ───────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f'trigger_examples_{dataset}_{poison_type}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved → {fname}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate clean vs. triggered image grids for backdoor attacks.'
    )
    parser.add_argument('--poison-type', default='badnet',
                        help='Attack type (e.g. badnet, blend, trojan, SIG, WaNet)')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'gtsrb'],
                        choices=['cifar10', 'gtsrb'],
                        help='Datasets to visualise')
    parser.add_argument('--n-samples', type=int, default=8,
                        help='Number of image columns in the grid')
    parser.add_argument('--data-dir', default='./data',
                        help='Root data directory')
    parser.add_argument('--output-dir', default='./results/figures/attack_triggers',
                        help='Where to save the figures')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for dataset in args.datasets:
        make_trigger_figure(
            dataset=dataset,
            poison_type=args.poison_type,
            n_samples=args.n_samples,
            data_dir=args.data_dir,
            output_dir=output_dir,
            seed=args.seed,
        )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
