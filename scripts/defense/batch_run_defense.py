"""
Batch defense runner.

Scans a directory of attack model checkpoints and runs STRIP or NC (or both)
on every model found.  Each run produces a per-model JSON result; after all
runs the individual results are merged into a single aggregated JSON file.

Expected attack model directory layout
---------------------------------------
results/models/attack/<attack_type>/
  <model>_<dataset>_<style>_<attack>_pr<rate>_<timestamp>/
    config.json          ← written by train_*_attack.py
    best_model.pth       ← checkpoint we evaluate
    training_log.json

Usage examples
--------------
# STRIP on all BadNet models (default output: results/defense/strip/badnet/)
python scripts/defense/batch_run_defense.py \\
    --attack-dir results/models/attack/badnet \\
    --defense STRIP

# NC on all WaNet models, on GPU 1
python scripts/defense/batch_run_defense.py \\
    --attack-dir results/models/attack/wanet  \\
    --defense NC --gpu 1 --nc-epochs 20

# Both STRIP and NC on Refool models
python scripts/defense/batch_run_defense.py \\
    --attack-dir results/models/attack/refool \\
    --defense both

# Dry-run: print what would be run without executing
python scripts/defense/batch_run_defense.py \\
    --attack-dir results/models/attack/badnet \\
    --defense STRIP --dry-run
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFENSE_SCRIPT = Path(__file__).resolve().parent / 'run_defense.py'


def _infer_cfg_from_ckpt(ckpt: Path) -> dict:
    """Infer run metadata from checkpoint path when config fields are missing."""
    run_name = ckpt.parent.name
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

    if 'poison_type' not in inferred:
        attack_dir_name = ckpt.parent.parent.name
        inferred['poison_type'] = 'WaNet' if attack_dir_name.lower() == 'wanet' else attack_dir_name

    return inferred


def find_attack_runs(attack_dir: Path) -> list[tuple[Path, Path]]:
    """Return paths to best_model.pth for every valid run subdirectory."""
    runs = []
    for run_dir in sorted(attack_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt       = run_dir / 'best_model.pth'
        cfg_path   = run_dir / 'config.json'
        log_path   = run_dir / 'training_log.json'
        if ckpt.exists() and cfg_path.exists():
            runs.append((ckpt, cfg_path))
        elif not ckpt.exists():
            print(f'  SKIP (no best_model.pth): {run_dir.name}')
        elif not cfg_path.exists():
            print(f'  SKIP (no config.json): {run_dir.name}')
    return runs


def run_single_defense(ckpt: Path, defense: str, output_dir: Path,
                       extra_args: list[str], dry_run: bool,
                       skip_existing: bool = False) -> tuple[bool, dict]:
    """
    Invoke run_defense.py for one checkpoint.
    Returns (success, result_dict).
    The result JSON written by run_defense.py is read back and returned.
    """
    # Build stem for the per-model JSON
    cfg_path = ckpt.parent / 'config.json'
    with open(cfg_path) as f:
        cfg = json.load(f)
    inferred = _infer_cfg_from_ckpt(ckpt)
    model_name = cfg.get('model') or inferred.get('model', 'unknown')
    dataset = cfg.get('dataset') or inferred.get('dataset', 'unknown')
    pretrained = (cfg.get('pretrained') if cfg.get('pretrained') is not None
                  else inferred.get('pretrained', False))
    poison_type = cfg.get('poison_type') or inferred.get('poison_type', 'unknown')
    poison_rate = (float(cfg.get('poison_rate')) if cfg.get('poison_rate') is not None
                   else float(inferred.get('poison_rate', 0.0)))

    stem = (f"{model_name}_{dataset}"
            f"_{'pretrained' if pretrained else 'scratch'}"
            f"_{poison_type}_pr{poison_rate:.3f}")
    json_out = output_dir / defense.lower() / f'{stem}.json'
    json_out.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and json_out.exists():
        print(f'  SKIP (already done): {stem}')
        with open(json_out) as f:
            return True, json.load(f)

    cmd = [
        'python', str(DEFENSE_SCRIPT),
        '--checkpoint',  str(ckpt),
        '--defense',     defense,
        '--output-dir',  str(output_dir / defense.lower()),
        '--output-json', str(json_out),
    ] + extra_args

    if dry_run:
        print(f'  [DRY-RUN] {" ".join(cmd)}')
        return True, {}

    print(f'\n  Running {defense} on {stem} …')
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    ok = result.returncode == 0
    print(f'  {"OK" if ok else "FAILED"} ({elapsed/60:.1f} min)')

    parsed: dict = {}
    if json_out.exists():
        with open(json_out) as f:
            parsed = json.load(f)

    return ok, parsed


def aggregate_and_save(results: list[dict], output_path: Path) -> None:
    """Save list of result dicts to an aggregated JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAggregated results saved to: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run STRIP / NC on all models in an attack output directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────────
    parser.add_argument('--attack-dir', required=True,
                        help='Path to the attack model directory '
                             '(e.g. results/models/attack/badnet)')
    parser.add_argument('--defense', required=True,
                        choices=['STRIP', 'NC', 'both'],
                        help='Which defense(s) to run')

    # ── output ────────────────────────────────────────────────────────────────
    parser.add_argument('--output-dir', default=None,
                        help='Root output directory for per-model artefacts '
                             '(default: results/defense/<attack_name>/)')
    parser.add_argument('--aggregated-json', default=None,
                        help='Path for merged results JSON '
                             '(default: results/defense/<attack>_<defense>_results.json)')

    # ── STRIP hyper-parameters ────────────────────────────────────────────────
    parser.add_argument('--strip-alpha', type=float, default=1.0)
    parser.add_argument('--strip-n',     type=int,   default=100)
    parser.add_argument('--strip-fpr',   type=float, default=0.1)

    # ── NC hyper-parameters ───────────────────────────────────────────────────
    parser.add_argument('--nc-epochs',     type=int, default=30)
    parser.add_argument('--nc-batch-size', type=int, default=32)

    # ── filters ───────────────────────────────────────────────────────────────
    parser.add_argument('--models',   nargs='+', default=None,
                        help='Restrict to these model names')
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=['cifar10', 'gtsrb'],
                        help='Restrict to these datasets')
    parser.add_argument('--skip-pretrained', action='store_true',
                        help='Skip pretrained (fine-tuned) checkpoints')
    parser.add_argument('--skip-scratch',    action='store_true',
                        help='Skip from-scratch checkpoints')

    # ── execution ─────────────────────────────────────────────────────────────
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--dry-run',    action='store_true')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip models whose output JSON already exists (allows resuming)')
    parser.add_argument('--skip-failed', action='store_true',
                        help='Continue even if a single model fails')

    args = parser.parse_args()

    attack_dir = Path(args.attack_dir).resolve()
    if not attack_dir.is_dir():
        sys.exit(f'Attack directory not found: {attack_dir}')

    attack_name = attack_dir.name  # e.g. 'badnet'

    if args.output_dir is None:
        output_dir = (PROJECT_ROOT / 'results' / 'defense' / attack_name).resolve()
    else:
        output_dir = Path(args.output_dir).resolve()

    # ── discover runs ──────────────────────────────────────────────────────────
    all_runs = find_attack_runs(attack_dir)
    if not all_runs:
        sys.exit(f'No valid runs found under {attack_dir}')

    # ── apply filters ──────────────────────────────────────────────────────────
    filtered_runs: list[tuple[Path, dict]] = []
    for ckpt, cfg_path in all_runs:
        with open(cfg_path) as f:
            cfg = json.load(f)
        model   = cfg.get('model', '')
        dataset = cfg.get('dataset', '')
        pretrained = cfg.get('pretrained', False)
        if args.models   and model   not in args.models:   continue
        if args.datasets and dataset not in args.datasets: continue
        if args.skip_pretrained and pretrained:  continue
        if args.skip_scratch    and not pretrained: continue
        filtered_runs.append((ckpt, cfg))

    print(f'\nFound {len(filtered_runs)} run(s) to defend in {attack_dir}')

    defenses = ['STRIP', 'NC'] if args.defense == 'both' else [args.defense]

    # ── extra args forwarded to run_defense.py ─────────────────────────────────
    extra = [
        '--gpu',        str(args.gpu),
        '--strip-alpha', str(args.strip_alpha),
        '--strip-n',     str(args.strip_n),
        '--strip-fpr',   str(args.strip_fpr),
        '--nc-epochs',   str(args.nc_epochs),
        '--nc-batch-size', str(args.nc_batch_size),
    ]

    bar = '=' * 70

    for defense in defenses:
        all_results: list[dict] = []
        failed: list[str]       = []
        t_total = time.time()

        print(f'\n{bar}')
        print(f'  Defense: {defense}  |  {len(filtered_runs)} models')
        print(f'{bar}')

        for ckpt, cfg in filtered_runs:
            stem = f"{cfg.get('model')}_{cfg.get('dataset')}_{cfg.get('poison_type')}"
            success, parsed = run_single_defense(
                ckpt=ckpt,
                defense=defense,
                output_dir=output_dir,
                extra_args=extra,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            if parsed:
                all_results.append(parsed)
            if not success:
                failed.append(stem)
                if not args.skip_failed:
                    print('Stopping due to failure. Use --skip-failed to continue.')
                    sys.exit(1)

        # ── save aggregated results ────────────────────────────────────────────
        if not args.dry_run and all_results:
            agg_path = (Path(args.aggregated_json).resolve() if args.aggregated_json
                        else PROJECT_ROOT / 'results' / 'defense'
                            / f'{attack_name}_{defense.lower()}_results.json')
            aggregate_and_save(all_results, agg_path)

        # ── summary ────────────────────────────────────────────────────────────
        elapsed = time.time() - t_total
        print(f'\n{bar}')
        print(f'  {defense} COMPLETE  |  '
              f'{len(filtered_runs)-len(failed)}/{len(filtered_runs)} succeeded  |  '
              f'{elapsed/60:.1f} min')
        if failed:
            print(f'  Failed: {", ".join(failed)}')
        print(f'{bar}\n')


if __name__ == '__main__':
    main()
