# Backdoor Analysis Framework

Unified framework for training, attacking, defending, and reporting backdoor experiments on CIFAR-10 and GTSRB with CNNs and Transformers.

## What this repository contains

- End-to-end training pipelines for clean and poisoned models
- Three attack pipelines: BadNet, WaNet, Refool
- One defense: STRIP
- Automated table and figure generation for baseline, attacks, and defenses
- Shared source package for models, datasets, and evaluation

## Repository layout

```text
.
├─ src/
│  ├─ models.py
│  ├─ datasets.py
│  ├─ evaluation.py
│  ├─ config.py
│  └─ __init__.py
├─ scripts/
│  ├─ training/
│  │  ├─ clean/
│  │  │  ├─ train.py
│  │  │  ├─ batch_train.py
│  │  │  └─ run_baseline_training.py
│  │  └─ attack/
│  │     ├─ badnet/
│  │     ├─ wanet/
│  │     └─ refool/
│  ├─ defense/
│  │  ├─ run_defense.py
│  │  ├─ batch_run_defense.py
│  │  └─ run_all_defenses.py
│  ├─ evaluation/
│  ├─ visualization/
│  │  ├─ generate_baseline_tables.py
│  │  ├─ generate_baseline_figures.py
│  │  ├─ generate_tables.py
│  │  ├─ generate_figures.py
│  │  ├─ generate_defense_tables.py
│  │  ├─ generate_defense_figures.py
│  │  └─ generate_attack_trigger_figures.py
│  └─ utils/
├─ data/                    # local datasets (not for git)
├─ results/                 # local models, logs, tables, figures (not for git)
├─ backdoor-toolbox-main/   # local dependency and data-generation toolbox
├─ environment.yml
├─ pyrightconfig.json
├─ QUICKSTART.md
├─ GETTING_STARTED.md
└─ SETUP_SUMMARY.md
```

## Models and datasets

- CNNs: ResNet18/34/50, VGG16/19
- Transformers: ViT (small/base), DeiT (small/base)
- Datasets: CIFAR-10 and GTSRB

## Environment setup

1. Create environment:

   conda env create -f environment.yml

2. Activate:

   conda activate backdoor-toolbox

## Data setup

- CIFAR-10 downloads automatically on first use.
- GTSRB must be placed under data/gtsrb with Train, Test, and GT-final_test.csv.
- Refool requires VOCdevkit under backdoor-toolbox-main/data/VOCdevkit.

## Main workflows

### 1) Clean baseline training

Run all predefined clean experiments:

```bash
conda run -n backdoor-toolbox python scripts/training/clean/run_baseline_training.py --mode 1
```

Useful modes:
- mode 1: all groups
- mode 2: scratch-only groups
- mode 3: CIFAR-10 groups
- mode 4: GTSRB groups

### 2) Attack training

BadNet:

```bash
conda run -n backdoor-toolbox python scripts/training/attack/badnet/run_badnet_attack.py --mode 1
```

WaNet:

```bash
conda run -n backdoor-toolbox python scripts/training/attack/wanet/run_wanet_attack.py --mode 1
```

Refool:

```bash
conda run -n backdoor-toolbox python scripts/training/attack/refool/run_refool_attack.py --mode 1
```

Each attack also supports single-run and batch scripts in its folder.

### 3) Defense runs

Run both defenses across attacks:

```bash
conda run -n backdoor-toolbox python scripts/defense/run_all_defenses.py --mode 1
```

Run one defense over selected checkpoints:

```bash
conda run -n backdoor-toolbox python scripts/defense/run_defense.py --help
```

### 4) Reporting: tables and figures

Baseline:

```bash
python scripts/visualization/generate_baseline_tables.py --results results/clean_results.json --save-dir results/tables/clean
python scripts/visualization/generate_baseline_figures.py --results results/clean_results.json --save-dir results/figures/clean
```

Attack results:

```bash
python scripts/visualization/generate_tables.py --results results/badnet_results.json --save-dir results/tables/attack/badnet
python scripts/visualization/generate_figures.py --results results/badnet_results.json --save-dir results/figures/attack/badnet --poison-type badnet
```

Defense results:

```bash
python scripts/visualization/generate_defense_tables.py --results results/defense/all_defenses_results.json --save-dir results/tables/defense
python scripts/visualization/generate_defense_figures.py --results results/defense/all_defenses_results.json --save-dir results/figures/defense
```

Trigger visual examples:

```bash
python scripts/visualization/generate_attack_trigger_figures.py --poison-type badnet --datasets cifar10 gtsrb --output-dir results/figures/attack_triggers
```

## Output conventions

- Model checkpoints and logs: results/models/
- Aggregated result json files: results/*.json and results/defense/*.json
- Publication tables: results/tables/
- Publication figures: results/figures/

## Notes for reproducibility

- Use the same conda environment for all scripts.
- Prefer running master scripts (run_baseline_training.py, run_*_attack.py, run_all_defenses.py) to keep result formats consistent.
- Visualization scripts consume json outputs produced by the training/defense scripts and save deterministic table/figure artifacts.

## Troubleshooting

- CUDA OOM: lower batch size and number of workers.
- Missing GTSRB/VOC: verify data folder structure exactly.
- Environment import issues: ensure backdoor-toolbox conda env is active.

## License and citation

If you use this repository in academic work, please cite the datasets and core methods below.

### Datasets

- CIFAR-10
   - Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. University of Toronto.
- GTSRB (German Traffic Sign Recognition Benchmark)
   - Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks, 32, 323-332.

### Backdoor attacks

- BadNets
   - Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.
- WaNet
   - Nguyen, A., Tran, A., et al. (2021). WaNet: Imperceptible Warping-based Backdoor Attack.
- Refool
   - Liu, Y., Ma, X., et al. (2020). Refool: Reflection Backdoor Attack on Deep Neural Network.

### Defenses

- STRIP
   - Gao, Y., Xu, C., et al. (2019). STRIP: A Defence Against Trojan Attacks on Deep Neural Networks.
- Neural Cleanse
   - Wang, B., Yao, Y., et al. (2019). Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks.

### Model families

- ResNet
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- VGG
   - Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.
- Vision Transformer (ViT)
   - Dosovitskiy, A., Beyer, L., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
- DeiT
   - Touvron, H., Cord, M., et al. (2021). Training data-efficient image transformers & distillation through attention.
