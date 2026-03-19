"""
Compile training logs from any set of model runs into a single results JSON.

Reads config.json + training_log.json from each subdirectory of the given
models directory and produces a flat list consumable by generate_figures.py
and generate_tables.py.

Works for both clean runs and attack runs — any extra fields present in
config.json (e.g. poison_type, poison_rate) are automatically included.

Usage:
    # clean models
    python scripts/evaluation/compile_results.py \
        --models-dir results/models/clean \
        --output     results/clean_results.json

    # badnet attack models
    python scripts/evaluation/compile_results.py \
        --models-dir results/models/attack/badnet \
        --output     results/attack_badnet_results.json

    # any other directory
    python scripts/evaluation/compile_results.py \
        --models-dir path/to/runs \
        --output     path/to/out.json
"""

import argparse
import json
from pathlib import Path


# Fields that are standard in every run's config.json
_CORE_CONFIG_FIELDS = {
    "model", "dataset", "pretrained", "optimizer",
    "lr", "epochs", "batch_size", "weight_decay",
}

# Extra fields that are meaningful if present (attack runs, etc.)
_EXTRA_CONFIG_FIELDS = [
    "poison_type",
    "poison_rate",
    "target_class",
    "trigger_size",
    "trigger_alpha",
    "num_poisoned",
]


def _training_style(cfg: dict) -> str:
    """Infer training style from config fields."""
    optimizer = cfg.get("optimizer", "sgd").lower()
    pretrained = cfg.get("pretrained", False)
    if optimizer == "adamw":
        return "vit_finetune"
    if pretrained:
        return "finetune"
    return "scratch"


def compile_results(models_dir: Path, output: Path) -> list:
    results = []

    run_dirs = sorted(d for d in models_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"  No subdirectories found in {models_dir}")
        return results

    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        log_path    = run_dir / "training_log.json"

        if not config_path.exists() or not log_path.exists():
            missing = []
            if not config_path.exists():
                missing.append("config.json")
            if not log_path.exists():
                missing.append("training_log.json")
            print(f"  SKIP ({', '.join(missing)} missing): {run_dir.name}")
            continue

        cfg = json.loads(config_path.read_text())
        log = json.loads(log_path.read_text())

        if not log:
            print(f"  SKIP (empty training log): {run_dir.name}")
            continue

        best = max(log, key=lambda e: e.get("val_acc", 0))

        # For attack runs: ASR at best-CA epoch may not equal peak ASR
        # (especially for WaNet where cover samples cause per-epoch oscillation).
        # Report both: asr at best CA epoch AND best ASR seen across all epochs.
        asr_values = [e["asr"] for e in log if e.get("asr") is not None]
        best_asr = round(max(asr_values), 3) if asr_values else None
        best_asr_epoch = next(
            (e["epoch"] for e in log if e.get("asr") == max(asr_values, default=None)),
            None
        ) if asr_values else None

        entry = {
            "run_dir":             run_dir.name,
            "model":               cfg.get("model", "unknown"),
            "dataset":             cfg.get("dataset", "unknown"),
            "pretrained":          cfg.get("pretrained", False),
            "training_style":      _training_style(cfg),
            "clean_accuracy":      round(best.get("val_acc", 0.0), 2),
            "attack_success_rate": best_asr,          # peak ASR across all epochs
            "asr_at_best_ca":      best.get("asr", None),  # ASR at best-CA epoch
            "best_asr_epoch":      best_asr_epoch,
            "training_epochs":     best.get("epoch"),
            "best_train_acc":      round(best.get("train_acc", 0.0), 2),
            "best_val_loss":       round(best.get("val_loss", 0.0), 4),
        }

        # Attach any extra config fields that are present (attack metadata etc.)
        for field in _EXTRA_CONFIG_FIELDS:
            if field in cfg:
                entry[field] = cfg[field]

        results.append(entry)
        asr_str = f"  best_asr={best_asr:.2f}" if best_asr is not None else ""
        ca_asr = best.get('asr')
        ca_asr_str = f" (at best-CA ep: {ca_asr:.2f})" if ca_asr is not None and ca_asr != best_asr else ""
        print(f"  {run_dir.name}  ->  val_acc={best.get('val_acc', 0):.2f}  "
              f"epoch={best.get('epoch')}{asr_str}{ca_asr_str}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nCompiled {len(results)} entries  ->  {output}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compile model-run results into a single JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Directory whose subdirectories are individual run folders "
             "(each must contain config.json and training_log.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path.  Defaults to <models-dir>/../../<models-dir-name>_results.json "
             "(i.e. results/<name>_results.json when models-dir is under results/models/)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        raise SystemExit(f"ERROR: --models-dir '{models_dir}' does not exist or is not a directory")

    if args.output:
        output = Path(args.output)
    else:
        # Sensible default: place JSON next to (or two levels up from) the models dir
        output = models_dir.parent.parent / f"{models_dir.name}_results.json"

    print(f"Compiling results from: {models_dir}")
    print(f"Output               : {output}\n")
    compile_results(models_dir, output)


if __name__ == "__main__":
    main()
