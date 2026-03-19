"""
Master orchestrator for all backdoor defense experiments.

Delegates to batch_run_defense.py for each (attack × defense) combination,
then produces a merged summary JSON across all attacks.

Modes
-----
  1 — all 3 attacks × both defenses   (STRIP + NC on badnet, wanet, refool)
  2 — all 3 attacks × STRIP only
  3 — all 3 attacks × NC only
  4 — BadNet only  × both defenses
  5 — WaNet only   × both defenses
  6 — Refool only  × both defenses

Attack model directories are expected at:
  results/models/attack/{badnet,wanet,refool}/

Aggregated defense results are written to:
  results/defense/{attack}_{defense}_results.json   (per attack × defense)
  results/defense/all_defenses_results.json          (master summary)

Usage examples
--------------
# Run everything (all attacks, both defenses)
conda run -n backdoor-toolbox python scripts/defense/run_all_defenses.py --mode 1

# STRIP only on all attacks
conda run -n backdoor-toolbox python scripts/defense/run_all_defenses.py --mode 2

# NC on BadNet models only, 15 optimization epochs
conda run -n backdoor-toolbox python scripts/defense/run_all_defenses.py \\
    --mode 4 --defense NC --nc-epochs 15

# Dry-run to preview all commands
conda run -n backdoor-toolbox python scripts/defense/run_all_defenses.py \\
    --mode 1 --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
BATCH_SCRIPT  = Path(__file__).resolve().parent / 'batch_run_defense.py'

# ── default attack → model directory mapping ──────────────────────────────────
_ATTACK_DIRS = {
    'badnet': PROJECT_ROOT / 'results' / 'models' / 'attack' / 'badnet',
    'wanet':  PROJECT_ROOT / 'results' / 'models' / 'attack' / 'WaNet',
    'refool': PROJECT_ROOT / 'results' / 'models' / 'attack' / 'refool',
}

# ── mode → list of (attack, defense) pairs ────────────────────────────────────
_ALL_ATTACKS  = ['badnet', 'wanet', 'refool']
_ALL_DEFENSES = ['STRIP', 'NC']

_MODE_MAP: dict[int, tuple[list[str], list[str]]] = {
    1: (_ALL_ATTACKS,  _ALL_DEFENSES),
    2: (_ALL_ATTACKS,  ['STRIP']),
    3: (_ALL_ATTACKS,  ['NC']),
    4: (['badnet'],    _ALL_DEFENSES),
    5: (['wanet'],     _ALL_DEFENSES),
    6: (['refool'],    _ALL_DEFENSES),
}


def check_attack_dirs(attacks: list[str]) -> list[str]:
    """Warn about missing attack directories; return only available ones."""
    available = []
    for a in attacks:
        d = _ATTACK_DIRS[a]
        if d.is_dir() and any(d.iterdir()):
            available.append(a)
        else:
            print(f'  WARNING: attack directory not found or empty: {d}')
    return available


def run_batch(attack: str, defense: str, args: argparse.Namespace) -> tuple[bool, Path | None]:
    """Call batch_run_defense.py for one (attack, defense) pair."""
    attack_dir = _ATTACK_DIRS[attack]
    out_root   = (Path(args.output_dir) / attack if args.output_dir
                  else PROJECT_ROOT / 'results' / 'defense' / attack)
    agg_json   = (PROJECT_ROOT / 'results' / 'defense'
                  / f'{attack}_{defense.lower()}_results.json')

    cmd = [
        'python', str(BATCH_SCRIPT),
        '--attack-dir',       str(attack_dir),
        '--defense',          defense,
        '--output-dir',       str(out_root),
        '--aggregated-json',  str(agg_json),
        '--gpu',              str(args.gpu),
        '--strip-alpha',      str(args.strip_alpha),
        '--strip-n',          str(args.strip_n),
        '--strip-fpr',        str(args.strip_fpr),
        '--nc-epochs',        str(args.nc_epochs),
        '--nc-batch-size',    str(args.nc_batch_size),
    ]
    if args.skip_failed:
        cmd.append('--skip-failed')
    if args.dry_run:
        cmd.append('--dry-run')
    if getattr(args, 'skip_existing', False):
        cmd.append('--skip-existing')
    if args.models:
        cmd += ['--models'] + args.models
    if args.datasets:
        cmd += ['--datasets'] + args.datasets

    bar = '=' * 72
    print(f'\n{bar}')
    print(f'  STARTING: {defense} on {attack}')
    print(f'{bar}\n')

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    ok = result.returncode == 0
    status = 'SUCCESS' if ok else f'FAILED (code {result.returncode})'
    print(f'\n{bar}')
    print(f'  {defense} / {attack}  —  {status}  |  {elapsed/60:.1f} min')
    print(f'{bar}\n')

    return ok, (agg_json if agg_json.exists() else None)


def merge_all_results(result_files: list[Path], output_path: Path) -> None:
    """Read all per-(attack×defense) JSONs and merge into one."""
    merged: list[dict] = []
    for p in result_files:
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                merged.extend(data)
            elif isinstance(data, dict):
                merged.append(data)
        except Exception as exc:
            print(f'  WARNING: could not read {p}: {exc}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'\nMaster summary ({len(merged)} records) saved to: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run all backdoor defenses (STRIP / NC) across all attack types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  1  All 3 attacks × STRIP + NC  (default)
  2  All 3 attacks × STRIP only
  3  All 3 attacks × NC only
  4  BadNet × STRIP + NC
  5  WaNet  × STRIP + NC
  6  Refool × STRIP + NC

Examples:
  python scripts/defense/run_all_defenses.py --mode 1
  python scripts/defense/run_all_defenses.py --mode 2 --strip-n 64
  python scripts/defense/run_all_defenses.py --mode 3 --nc-epochs 15
""",
    )

    parser.add_argument('--mode', type=int, default=1, choices=list(_MODE_MAP.keys()),
                        help='Experiment scope (see above)')
    parser.add_argument('--defense', default=None, choices=['STRIP', 'NC'],
                        help='Override mode: run only this defense for all attacks')

    # ── STRIP hyper-parameters ────────────────────────────────────────────────
    parser.add_argument('--strip-alpha', type=float, default=1.0)
    parser.add_argument('--strip-n',     type=int,   default=100,
                        help='Perturbations per test image (default: 100)')
    parser.add_argument('--strip-fpr',   type=float, default=0.1,
                        help='Target false-positive rate for STRIP threshold (default: 0.1)')

    # ── NC hyper-parameters ───────────────────────────────────────────────────
    parser.add_argument('--nc-epochs',     type=int, default=30,
                        help='Optimisation epochs per class in NC (default: 30)')
    parser.add_argument('--nc-batch-size', type=int, default=32)

    # ── filters ───────────────────────────────────────────────────────────────
    parser.add_argument('--models',   nargs='+', default=None,
                        help='Restrict to these model names')
    parser.add_argument('--datasets', nargs='+', choices=['cifar10', 'gtsrb'], default=None,
                        help='Restrict to these datasets')

    # ── paths ─────────────────────────────────────────────────────────────────
    parser.add_argument('--output-dir', default=None,
                        help='Root output directory (default: results/defense/<attack>/)')

    # ── execution ─────────────────────────────────────────────────────────────
    parser.add_argument('--gpu',         type=int,  default=0)
    parser.add_argument('--skip-failed', action='store_true',
                        help='Continue to the next combination if a batch fails')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip models whose output JSON already exists (allows resuming)')
    parser.add_argument('--dry-run',     action='store_true',
                        help='Print commands without executing them')

    args = parser.parse_args()

    attacks, defenses = _MODE_MAP[args.mode]
    if args.defense is not None:
        defenses = [args.defense]

    # Remove attacks with missing directories
    attacks = check_attack_dirs(attacks)
    if not attacks:
        sys.exit('No attack model directories available.')

    bar = '=' * 72
    combos = [(a, d) for a in attacks for d in defenses]

    print(f'\n{bar}')
    print(f'  Defense Master Orchestrator')
    print(f'  mode       : {args.mode}')
    print(f'  attacks    : {", ".join(attacks)}')
    print(f'  defenses   : {", ".join(defenses)}')
    print(f'  combinations: {len(combos)}')
    print(f'{bar}\n')

    failed:       list[str]  = []
    result_files: list[Path] = []
    t_total = time.time()

    for attack, defense in combos:
        ok, agg_json = run_batch(attack, defense, args)
        if agg_json:
            result_files.append(agg_json)
        if not ok:
            failed.append(f'{defense}/{attack}')
            if not args.skip_failed:
                print('Stopping due to failure. Use --skip-failed to continue.')
                sys.exit(1)

    # ── merge everything into a single master JSON ─────────────────────────────
    if not args.dry_run and result_files:
        master_json = PROJECT_ROOT / 'results' / 'defense' / 'all_defenses_results.json'
        merge_all_results(result_files, master_json)

    # ── final summary ──────────────────────────────────────────────────────────
    total_time = time.time() - t_total
    print(f'\n{bar}')
    print(f'  ALL DONE  |  '
          f'{len(combos)-len(failed)}/{len(combos)} combinations succeeded  |  '
          f'{total_time/3600:.2f}h')
    if failed:
        print(f'  Failed: {", ".join(failed)}')
    print(f'{bar}\n')

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
