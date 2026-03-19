"""
Run STRIP or Neural Cleanse on a single trained model checkpoint.

Bridges our model training pipeline (src/models.py architectures + our checkpoint format)
with the backdoor-toolbox defense implementations.  All STRIP / NC logic is executed
in-process; no subprocess needed.

Key design notes
----------------
* The toolbox's BackdoorDefense.__init__ normally both constructs the model and loads
  its weights.  We monkey-patch config.arch[dataset] with a factory that creates the
  correct architecture (VGG16/ResNet18/ViT …) before instantiation, then pass a
  temporary state-dict file containing just the weights of our checkpoint so the
  toolbox can call model.load_state_dict() as usual.
* config.data_dir is re-pointed to TOOLBOX_DIR/data/ so the toolbox's data loaders
  find CIFAR-10 / GTSRB in the expected location.
* The working directory is temporarily set to TOOLBOX_DIR so relative paths inside
  the toolbox (histogram output, NC artefact folders, etc.) resolve correctly.
* All toolbox stdout is captured, then re-printed, and key numerical metrics are
  extracted via regex into the output JSON.

Usage examples
--------------
# STRIP on a BadNet-poisoned ResNet18
python scripts/defense/run_defense.py \\
    --checkpoint results/models/attack/badnet/<run>/best_model.pth \\
    --defense STRIP

# STRIP with explicit model / dataset (overrides config.json)
python scripts/defense/run_defense.py \\
    --checkpoint results/models/attack/wanet/<run>/best_model.pth \\
    --model vgg16 --dataset cifar10 \\
    --poison-type WaNet --poison-rate 0.05 --cover-rate 0.1 \\
    --defense STRIP

# NC on a Refool model (slow on GTSRB — use --nc-epochs 10 for a quick test)
python scripts/defense/run_defense.py \\
    --checkpoint results/models/attack/refool/<run>/best_model.pth \\
    --defense NC --nc-epochs 10
"""

import argparse
import contextlib
import importlib
import io
import json
import os
import re
import sys
import tempfile
import types
from pathlib import Path

import torch
import numpy as np

# ── project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLBOX_DIR  = PROJECT_ROOT / 'backdoor-toolbox-main'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(TOOLBOX_DIR))
sys.path.insert(0, str(TOOLBOX_DIR / 'utils'))

from models import get_model
from datasets import get_num_classes

# ── constants ──────────────────────────────────────────────────────────────────
_NUM_CLASSES = {'cifar10': 10, 'gtsrb': 43}
_TARGET_CLASS = {'cifar10': 0, 'gtsrb': 2}


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _cwd(path: Path):
    """Temporarily change the working directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _import_toolbox():
    """Import toolbox modules and patch absolute data / trigger paths."""
    supervisor = importlib.import_module('supervisor')
    tb_config  = importlib.import_module('config')

    # Use absolute paths so toolbox internals work regardless of cwd
    setattr(tb_config, 'data_dir',     str(TOOLBOX_DIR / 'data'))
    setattr(tb_config, 'triggers_dir', str(TOOLBOX_DIR / 'triggers'))
    return supervisor, tb_config


def _read_config_json(checkpoint: Path) -> dict:
    """Read config.json from the run directory next to the checkpoint file."""
    cfg_path = checkpoint.parent / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def _infer_run_metadata(checkpoint: Path) -> dict:
    """Infer metadata from run-folder naming when config.json is incomplete."""
    run_name = checkpoint.parent.name
    # Expected examples:
    #   deit_small_cifar10_pretrained_WaNet_pr0.050_cr0.100_20260303_235124
    #   resnet18_gtsrb_scratch_badnet_pr0.100_20260225_190040
    m = re.search(
        r'^(?P<model>.+?)_(?P<dataset>cifar10|gtsrb)_(?P<style>pretrained|scratch)_(?P<poison>[^_]+)_pr(?P<pr>[0-9.]+)(?:_cr(?P<cr>[0-9.]+))?',
        run_name,
    )
    inferred: dict = {}
    if m:
        inferred['model'] = m.group('model')
        inferred['dataset'] = m.group('dataset')
        inferred['pretrained'] = (m.group('style') == 'pretrained')
        inferred['poison_type'] = m.group('poison')
        try:
            inferred['poison_rate'] = float(m.group('pr'))
        except (TypeError, ValueError):
            pass
        cr = m.group('cr')
        if cr is not None:
            try:
                inferred['cover_rate'] = float(cr)
            except (TypeError, ValueError):
                pass

    # Secondary fallback: attack directory usually encodes poison type
    if 'poison_type' not in inferred:
        attack_dir_name = checkpoint.parent.parent.name
        if attack_dir_name.lower() == 'wanet':
            inferred['poison_type'] = 'WaNet'
        elif attack_dir_name:
            inferred['poison_type'] = attack_dir_name

    return inferred


def _load_our_model(checkpoint: Path, model_name: str, dataset: str,
                    pretrained: bool) -> torch.nn.Module:
    """
    Reconstruct the model architecture and load weights from our checkpoint.
    Handles both plain state-dict files and our {'model_state_dict': …} format.
    """
    num_classes = _NUM_CLASSES[dataset]
    ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Strip DataParallel prefix if present
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}

    model = get_model(model_name, num_classes=num_classes, pretrained=pretrained,
                      dataset=dataset)
    model.load_state_dict(state_dict)
    return model


def _write_plain_state_dict(model: torch.nn.Module) -> str:
    """
    Save model.state_dict() (no outer wrapper dict) to a temp file.
    The toolbox calls torch.load(path) then model.load_state_dict(…) directly.
    """
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    torch.save(model.state_dict(), tmp.name)
    tmp.close()
    return tmp.name


def _make_arch_factory(model_name: str, dataset: str, pretrained: bool):
    """
    Return a callable(num_classes) → nn.Module that creates our architecture.
    Used to monkey-patch config.arch[dataset] so the toolbox constructs the
    correct model type before loading our state dict.
    """
    def factory(num_classes: int):
        return get_model(model_name, num_classes=num_classes,
                         pretrained=pretrained, dataset=dataset)
    return factory


def _build_toolbox_args(dataset: str, poison_type: str, poison_rate: float,
                         cover_rate: float, alpha: float, model_path: str,
                         defense: str, gpu: int, tb_config) -> types.SimpleNamespace:
    """Build a minimal toolbox-compatible args namespace."""
    trigger = tb_config.trigger_default.get(dataset, {}).get(poison_type, 'none')
    return types.SimpleNamespace(
        dataset=dataset,
        poison_type=poison_type,
        poison_rate=poison_rate,
        cover_rate=cover_rate,
        alpha=alpha,
        test_alpha=None,
        trigger=trigger,
        no_aug=False,
        no_normalize=False,
        model=None,
        model_path=model_path,
        defense=defense,
        devices=str(gpu),
        seed=2333,
        noisy_test=False,
        log=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Output parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_strip_metrics(output: str) -> dict:
    """Extract STRIP metrics from captured stdout."""
    patterns = {
        'clean_entropy_median':         r'Entropy Clean\s+Median:\s+([\d.eE+-]+)',
        'poison_entropy_median':        r'Entropy Poison Median:\s+([\d.eE+-]+)',
        'threshold_low':                r'thresholds \(([\d.eE+-]+),',
        'tpr':                          r'TPR:\s+([\d.]+)',
        'fpr':                          r'FPR:\s+([\d.]+)',
        'auc':                          r'AUC:\s+([\d.]+)',
        'clean_accuracy_after_defense': r'Clean Accuracy:\s+\d+/\d+\s+=\s+([\d.eE+-]+)',
        'asr_after_defense':            r'ASR:\s+\d+/\d+\s+=\s+([\d.eE+-]+)',
    }
    metrics: dict = {}
    for key, pat in patterns.items():
        m = re.search(pat, output)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def _parse_nc_metrics(output: str, output_dir: Path) -> dict:
    """Extract NC metrics from captured stdout and saved .npz artefact."""
    metrics: dict = {}

    # Suspect class identified and unlearning triggered
    m = re.search(r'Unlearning with reversed trigger from class (\d+)', output)
    if m:
        metrics['suspect_class'] = int(m.group(1))
        metrics['nc_detected']   = True
    else:
        m2 = re.search(r'Suspect Classes:\s+\[([^\]]*)\]', output)
        if m2 and m2.group(1).strip():
            classes = [int(x.strip()) for x in m2.group(1).split(',') if x.strip()]
            metrics['suspect_classes'] = classes
            metrics['nc_detected']     = len(classes) > 0
        else:
            metrics['nc_detected'] = False

    # Load mask norms from the saved .npz file
    npz_files = sorted(output_dir.glob('neural_cleanse_*.npz'))
    if npz_files:
        try:
            f = np.load(str(npz_files[0]), allow_pickle=True)
            mask_list  = torch.tensor(np.array(f['mask_list']))
            mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1).tolist()
            loss_list  = f['loss_list'].tolist() if 'loss_list' in f else []
            metrics['mask_norms']       = mask_norms
            metrics['loss_list']        = loss_list
            metrics['min_norm_class']   = int(np.argmin(mask_norms))
        except Exception as exc:
            metrics['npz_parse_error'] = str(exc)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Defense runners
# ══════════════════════════════════════════════════════════════════════════════

def run_strip(tb_args, model_name: str, dataset: str, pretrained: bool,
              strip_alpha: float, n_samples: int, defense_fpr: float,
              output_dir: Path, tb_config) -> dict:
    """
    Set up and run STRIP detection. Returns a metrics dict.

    tb_args.model_path must already point to a plain state-dict temp file
    (written by the caller in main()).
    """
    from other_defenses_tool_box.strip import STRIP

    output_dir.mkdir(parents=True, exist_ok=True)

    captured_buf = io.StringIO()
    orig_arch = tb_config.arch.get(dataset)
    tb_config.arch[dataset] = _make_arch_factory(model_name, dataset, pretrained)
    try:
        with _cwd(TOOLBOX_DIR), contextlib.redirect_stdout(captured_buf):
            # Reduce batch_size from 128 to 16 to prevent OOM on large models (DeiT, ViT)
            # Each test sample processes N × batch_size images through the model
            defense = STRIP(tb_args, strip_alpha=strip_alpha,
                            N=n_samples, defense_fpr=defense_fpr, batch_size=16)
            # After weights are loaded, clear model_path so the temp-file path
            # does not get embedded in the histogram filename via get_model_name().
            tb_args.model_path = None
            tb_args.model      = model_name
            # Override folder_path so the histogram goes to our output dir
            defense.folder_path = str(output_dir)
            defense.detect(inspect_correct_predition_only=True, noisy_test=False)
    finally:
        if orig_arch is not None:
            tb_config.arch[dataset] = orig_arch

    captured = captured_buf.getvalue()
    sys.stdout.write(captured)  # echo to real stdout
    return _parse_strip_metrics(captured)


def run_nc(tb_args, model_name: str, dataset: str, pretrained: bool,
           nc_epochs: int, nc_batch_size: int,
           output_dir: Path, tb_config) -> dict:
    """Set up and run Neural Cleanse. Returns a metrics dict."""
    from other_defenses_tool_box.neural_cleanse import NC

    output_dir.mkdir(parents=True, exist_ok=True)

    captured_buf = io.StringIO()
    orig_arch = tb_config.arch.get(dataset)
    tb_config.arch[dataset] = _make_arch_factory(model_name, dataset, pretrained)
    try:
        with _cwd(TOOLBOX_DIR), contextlib.redirect_stdout(captured_buf):
            defense = NC(tb_args, epoch=nc_epochs, batch_size=nc_batch_size,
                         init_cost=1e-3, patience=5,
                         attack_succ_threshold=0.99, oracle=False)
            # Clear temp-file path so it doesn't appear in NC artefact filenames
            tb_args.model_path = None
            tb_args.model      = model_name
            defense.folder_path = str(output_dir)
            defense.detect()
    finally:
        if orig_arch is not None:
            tb_config.arch[dataset] = orig_arch

    captured = captured_buf.getvalue()
    sys.stdout.write(captured)

    return _parse_nc_metrics(captured, output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Run STRIP or NC defense on a single trained model checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pth (our checkpoint format)')
    parser.add_argument('--defense', required=True, choices=['STRIP', 'NC'],
                        help='Defense algorithm to apply')

    # ── model / data (auto-filled from config.json when omitted) ──────────────
    parser.add_argument('--model',       default=None,
                        help='Architecture name (vgg16, resnet18, …). '
                             'Auto-read from config.json if omitted.')
    parser.add_argument('--dataset',     default=None, choices=['cifar10', 'gtsrb'],
                        help='Dataset. Auto-read from config.json if omitted.')
    parser.add_argument('--poison-type', default=None,
                        help='Poison type (badnet, WaNet, refool, …). '
                             'Auto-read from config.json if omitted.')
    parser.add_argument('--poison-rate', type=float, default=None,
                        help='Poison rate. Auto-read from config.json if omitted.')
    parser.add_argument('--cover-rate',  type=float, default=None,
                        help='Cover rate (WaNet). Auto-read from config.json if omitted.')
    parser.add_argument('--pretrained',  action='store_true', default=None,
                        help='Model was trained with pretrained backbone.')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')

    # ── STRIP hyper-parameters ────────────────────────────────────────────────
    parser.add_argument('--strip-alpha',  type=float, default=1.0,
                        help='STRIP superimposition factor (default: 1.0)')
    parser.add_argument('--strip-n',      type=int,   default=100,
                        help='Number of random images superimposed per test input (default: 100)')
    parser.add_argument('--strip-fpr',    type=float, default=0.1,
                        help='Target false-positive rate for entropy threshold (default: 0.1)')

    # ── NC hyper-parameters ───────────────────────────────────────────────────
    parser.add_argument('--nc-epochs',      type=int, default=30,
                        help='Optimisation epochs per class for NC (default: 30)')
    parser.add_argument('--nc-batch-size',  type=int, default=32,
                        help='Batch size for NC optimisation (default: 32)')

    # ── system / output ───────────────────────────────────────────────────────
    parser.add_argument('--gpu',        type=int,  default=0)
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save artefacts (histograms, masks, JSON). '
                             'Defaults to results/defense/<defense_lower>/<attack>/')
    parser.add_argument('--output-json', default=None,
                        help='Path for the result JSON file. '
                             'Defaults to <output-dir>/<stem>.json')

    args = parser.parse_args()

    # ── resolve paths ──────────────────────────────────────────────────────────
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        sys.exit(f'Checkpoint not found: {checkpoint}')

    # ── fill missing args from config.json ────────────────────────────────────
    run_cfg = _read_config_json(checkpoint)
    inferred_cfg = _infer_run_metadata(checkpoint)

    model_name = args.model or run_cfg.get('model') or inferred_cfg.get('model')
    dataset = args.dataset or run_cfg.get('dataset') or inferred_cfg.get('dataset')
    poison_type = args.poison_type or run_cfg.get('poison_type') or inferred_cfg.get('poison_type')

    if args.poison_rate is not None:
        poison_rate = args.poison_rate
    elif (_pr := run_cfg.get('poison_rate')) is not None:
        poison_rate = float(_pr)
    else:
        poison_rate = float(inferred_cfg.get('poison_rate', 0.1))

    if args.cover_rate is not None:
        cover_rate = args.cover_rate
    elif (_cr := run_cfg.get('cover_rate')) is not None:
        cover_rate = float(_cr)
    else:
        cover_rate = float(inferred_cfg.get('cover_rate', 0.0))

    if args.pretrained is not None:
        pretrained = args.pretrained
    elif run_cfg.get('pretrained') is not None:
        pretrained = bool(run_cfg.get('pretrained'))
    else:
        pretrained = bool(inferred_cfg.get('pretrained', False))
    alpha       = float(run_cfg.get('alpha', 0.2))

    if not model_name or not dataset or not poison_type:
        sys.exit('--model, --dataset and --poison-type are required '
                 '(or must be present in config.json next to the checkpoint).')

    # ── output directory & JSON path ──────────────────────────────────────────
    if args.output_dir is None:
        output_dir = (PROJECT_ROOT / 'results' / 'defense'
                      / args.defense.lower() / poison_type)
    else:
        output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()

    stem = (f"{model_name}_{dataset}"
            f"_{'pretrained' if pretrained else 'scratch'}"
            f"_{poison_type}_pr{poison_rate:.3f}")
    json_path = Path(args.output_json).resolve() if args.output_json else (output_dir / f'{stem}.json')

    print(f"\n{'='*70}")
    print(f"  Defense:    {args.defense}")
    print(f"  Model:      {model_name}  ({'pretrained' if pretrained else 'scratch'})")
    print(f"  Dataset:    {dataset}")
    print(f"  Attack:     {poison_type}  pr={poison_rate:.3f}  cr={cover_rate:.3f}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*70}\n")

    # ── import toolbox and monkey-patch paths ──────────────────────────────────
    supervisor, tb_config = _import_toolbox()

    # ── write temp state-dict file ────────────────────────────────────────────
    model = _load_our_model(checkpoint, model_name, dataset, pretrained)
    tmp_sd_path = _write_plain_state_dict(model)

    try:
        # ── build toolbox args ─────────────────────────────────────────────────
        tb_args = _build_toolbox_args(
            dataset=dataset,
            poison_type=poison_type,
            poison_rate=poison_rate,
            cover_rate=cover_rate,
            alpha=alpha,
            model_path=tmp_sd_path,
            defense=args.defense,
            gpu=args.gpu,
            tb_config=tb_config,
        )

        # ── run chosen defense ─────────────────────────────────────────────────
        if args.defense == 'STRIP':
            defense_metrics = run_strip(
                tb_args=tb_args,
                model_name=model_name, dataset=dataset, pretrained=pretrained,
                strip_alpha=args.strip_alpha,
                n_samples=args.strip_n,
                defense_fpr=args.strip_fpr,
                output_dir=output_dir,
                tb_config=tb_config,
            )
        else:  # NC
            defense_metrics = run_nc(
                tb_args=tb_args,
                model_name=model_name, dataset=dataset, pretrained=pretrained,
                nc_epochs=args.nc_epochs,
                nc_batch_size=args.nc_batch_size,
                output_dir=output_dir,
                tb_config=tb_config,
            )
        
        # Clear GPU cache after defense completes to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:
        if os.path.exists(tmp_sd_path):
            os.unlink(tmp_sd_path)
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── build result record ────────────────────────────────────────────────────
    training_style = ('vit_finetune' if any(v in model_name for v in ['vit', 'deit']) and pretrained
                      else ('finetune' if pretrained else 'scratch'))

    result = {
        'model':          model_name,
        'dataset':        dataset,
        'poison_type':    poison_type,
        'poison_rate':    poison_rate,
        'cover_rate':     cover_rate,
        'pretrained':     pretrained,
        'training_style': training_style,
        'defense':        args.defense,
        'checkpoint':     str(checkpoint),
        'target_class':   _TARGET_CLASS.get(dataset, 0),
        **defense_metrics,
    }

    # NC-specific: flag whether the target class was correctly identified
    if args.defense == 'NC':
        target = _TARGET_CLASS.get(dataset, 0)
        if 'suspect_class' in result:
            result['correctly_identified_target'] = (result['suspect_class'] == target)
        elif 'suspect_classes' in result:
            result['correctly_identified_target'] = (target in result['suspect_classes'])

    # ── save JSON ──────────────────────────────────────────────────────────────
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nResult saved to: {json_path}")

    # ── pretty-print summary ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  {args.defense} SUMMARY  |  {model_name} / {dataset} / {poison_type}")
    print(f"{'─'*50}")
    if args.defense == 'STRIP':
        for key in ['tpr', 'fpr', 'auc', 'clean_entropy_median',
                    'poison_entropy_median', 'asr_after_defense']:
            if key in result:
                print(f"  {key:<30}: {result[key]}")
    else:
        for key in ['nc_detected', 'suspect_class', 'suspect_classes',
                    'min_norm_class', 'correctly_identified_target']:
            if key in result:
                print(f"  {key:<30}: {result[key]}")
    print(f"{'─'*50}\n")

    return result


if __name__ == '__main__':
    main()
