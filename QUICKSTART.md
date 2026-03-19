# Quick Start Guide

## Initial Setup (5 minutes)

### 1. Create Environment
```bash
cd "C:\Users\manua\Desktop\Codice Tesi"
conda env create -f environment.yml
conda activate backdoor-toolbox
```

### 2. Verify Installation
```bash
# Test model creation
python src/models.py

# Test dataset loading (CIFAR-10 will download automatically)
python src/datasets.py
```

## Quick Training Examples

### Single Model Training (30 minutes - 2 hours depending on model)

```bash
# Train ResNet18 on CIFAR-10 from scratch
python scripts/training/train.py --model resnet18 --dataset cifar10

# Quick test (5 epochs, for testing the pipeline)
python scripts/training/train.py --model resnet18 --dataset cifar10 --epochs 5 --batch-size 32

# Fine-tune VGG16 on CIFAR-10 with pre-trained weights
python scripts/training/train.py --model vgg16 --dataset cifar10 --pretrained
```

### Batch Training

```bash
# Dry run to see what will be executed
python scripts/training/batch_train.py --experiment cifar10_scratch --dry-run

# Train all CNN models on CIFAR-10 from scratch
python scripts/training/batch_train.py --experiment cifar10_scratch

# Train only specific models
python scripts/training/batch_train.py \
    --models vgg16 resnet18 resnet50 \
    --dataset cifar10 \
    --config scratch
```

## View Training Progress

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir results/models

# Then open browser to http://localhost:6006
```

### List Trained Models
```bash
# List all models
python scripts/utils/list_models.py

# Compare models
python scripts/utils/list_models.py --compare --summary

# Filter by dataset
python scripts/utils/list_models.py --dataset cifar10 --detailed
```

## Evaluate Models

```bash
# Evaluate a trained model
python scripts/evaluation/evaluate_model.py \
    --checkpoint results/models/MODEL_DIR/best_model.pth \
    --detailed

# Evaluate with per-class analysis
python scripts/evaluation/evaluate_model.py \
    --checkpoint results/models/MODEL_DIR/best_model.pth \
    --detailed \
    --show-report
```

## Recommended Training Pipeline

### For Comprehensive Analysis

I recommend training both from scratch and fine-tuned versions:

```bash
# 1. Start with CIFAR-10 (faster, good for testing)
python scripts/training/batch_train.py --experiment cifar10_scratch
python scripts/training/batch_train.py --experiment cifar10_finetune
python scripts/training/batch_train.py --experiment cifar10_vit

# 2. Then GTSRB (more classes, more challenging)
python scripts/training/batch_train.py --experiment gtsrb_scratch
python scripts/training/batch_train.py --experiment gtsrb_finetune
python scripts/training/batch_train.py --experiment gtsrb_vit
```

This will give you:
- **CIFAR-10**: 5 CNNs (scratch) + 5 CNNs (fine-tuned) + 4 ViTs = 14 models
- **GTSRB**: 5 CNNs (scratch) + 5 CNNs (fine-tuned) + 4 ViTs = 14 models
- **Total**: 28 models for comprehensive analysis

### Estimated Training Time

On a modern GPU (e.g., RTX 3080/4080):
- CNN from scratch: ~2-3 hours each
- CNN fine-tuned: ~1-1.5 hours each
- ViT fine-tuned: ~2-4 hours each

**Total estimated time**: ~60-80 hours for all 28 models

**Recommendation**: Train in batches overnight or over a weekend.

## GTSRB Setup

GTSRB requires manual download:

1. Download from: https://benchmark.ini.rub.de/gtsrb_dataset.html
   - Training set (GTSRB_Final_Training_Images.zip)
   - Test set (GTSRB_Final_Test_Images.zip + GT-final_test.csv)

2. Extract to the data directory:
```
data/gtsrb/
├── Train/
│   ├── 00000/
│   ├── 00001/
│   └── ... (00042)
├── Test/
│   ├── 00000.ppm
│   └── ...
└── GT-final_test.csv
```

3. Verify:
```bash
python src/datasets.py
```

## Common Use Cases

### Test Training Pipeline Quickly
```bash
# 5 epochs, small batch size, should complete in ~10-15 minutes
python scripts/training/train.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 5 \
    --batch-size 32
```

### Train Best-Performing Models
```bash
# Based on literature, these typically perform best:
python scripts/training/batch_train.py \
    --models resnet18 resnet50 vit_base \
    --dataset cifar10 \
    --config finetune
```

### Train for Quick Iteration (Testing Backdoor Attacks)
```bash
# Fewer epochs for faster iteration when testing attacks
python scripts/training/train.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 50 \
    --pretrained
```

## Next Steps After Training

1. **List your models**:
   ```bash
   python scripts/utils/list_models.py --compare --summary
   ```

2. **Evaluate on test set**:
   ```bash
   python scripts/evaluation/evaluate_model.py \
       --checkpoint results/models/YOUR_MODEL/best_model.pth \
       --detailed
   ```

3. **Apply backdoor attacks** using backdoor-toolbox-main

4. **Analyze results** and compare attack success rates

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/training/train.py --model vit_base --dataset cifar10 --batch-size 32

# Or use CPU (much slower)
python scripts/training/train.py --model resnet18 --dataset cifar10 --gpu -1
```

### Dataset Not Found
```bash
# CIFAR-10: Will download automatically on first run
# GTSRB: Must be downloaded manually (see GTSRB Setup above)
```

### Import Errors
```bash
# Make sure conda environment is activated
conda activate backdoor-toolbox

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
python -c "import timm; print(timm.__version__)"
```

## Tips

1. **Start small**: Test with 5-10 epochs first to ensure everything works
2. **Monitor with TensorBoard**: Track training progress in real-time
3. **Save GPU memory**: Close other GPU-intensive applications
4. **Batch training**: Train overnight using `batch_train.py`
5. **Backup models**: Models are saved in `results/models/` - back up regularly

## Questions?

Check the main README.md for detailed documentation or review the code in:
- `src/models.py` - Model architectures
- `src/datasets.py` - Dataset loaders
- `scripts/training/train.py` - Training logic
- `src/config.py` - Configuration options
