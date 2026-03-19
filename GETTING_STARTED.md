# 🎯 Getting Started Checklist

## ✅ Immediate Next Steps

### 1. Activate the Environment
```powershell
conda activate backdoor-toolbox
```

### 2. Install timm (if not already done)
```powershell
pip install timm
```

### 3. Verify Installation
```powershell
python test_setup.py
```

You should see all checks pass (✓).

### 4. Run a Quick Test Training (5-10 minutes)
```powershell
python scripts/training/train.py --model resnet18 --dataset cifar10 --epochs 5 --batch-size 32
```

This will:
- Download CIFAR-10 (first time only, ~170 MB)
- Train ResNet18 for 5 epochs
- Save results to `results/models/`

## 📊 Recommended Training Order

### Week 1: CIFAR-10 Baseline Models
```powershell
# Start overnight - Train all CNNs from scratch (~10-15 hours)
python scripts/training/batch_train.py --experiment cifar10_scratch

# Next day - Fine-tune pre-trained models (~5-8 hours)
python scripts/training/batch_train.py --experiment cifar10_finetune

# Weekend - Train Vision Transformers (~8-16 hours)
python scripts/training/batch_train.py --experiment cifar10_vit
```

### Week 2: GTSRB Models (after downloading GTSRB data)
```powershell
python scripts/training/batch_train.py --experiment gtsrb_scratch
python scripts/training/batch_train.py --experiment gtsrb_finetune
python scripts/training/batch_train.py --experiment gtsrb_vit
```

## 🔍 Monitor Training Progress

### Option 1: TensorBoard (Recommended)
```powershell
# In a separate terminal
tensorboard --logdir results/models

# Open browser to: http://localhost:6006
```

### Option 2: Check Training Logs
```powershell
# List all trained models
python scripts/utils/list_models.py --compare --summary

# View specific model log
code results/models/YOUR_MODEL_DIR/training_log.json
```

## 📈 After Training

### 1. Evaluate Your Best Models
```powershell
python scripts/evaluation/evaluate_model.py `
    --checkpoint results/models/YOUR_MODEL/best_model.pth `
    --detailed `
    --show-report
```

### 2. Compare All Models
```powershell
python scripts/utils/list_models.py --compare --summary --export results/tables/model_comparison.csv
```

### 3. Start Backdoor Attack Experiments
Now you can use these trained models as victims in the backdoor-toolbox-main!

## 💡 Quick Reference Commands

### Training
```powershell
# Single model
python scripts/training/train.py --model MODEL --dataset DATASET [OPTIONS]

# Batch training
python scripts/training/batch_train.py --experiment EXPERIMENT_NAME

# Dry run (see what will be executed)
python scripts/training/batch_train.py --experiment cifar10_scratch --dry-run
```

### Evaluation
```powershell
# Basic evaluation
python scripts/evaluation/evaluate_model.py --checkpoint PATH/best_model.pth

# Detailed evaluation
python scripts/evaluation/evaluate_model.py --checkpoint PATH/best_model.pth --detailed --show-report
```

### Management
```powershell
# List all models
python scripts/utils/list_models.py

# Filter and compare
python scripts/utils/list_models.py --dataset cifar10 --compare --summary

# Export to CSV
python scripts/utils/list_models.py --export results/tables/models.csv
```

## 🎓 Using Pre-trained vs From Scratch

**My recommendation: Use BOTH for comprehensive analysis**

### Pre-trained Models (Fine-tuned)
- ✅ Use these when: Testing real-world scenarios (most practitioners use pre-trained)
- ✅ Benefits: Better baseline accuracy, faster training, realistic attack scenarios
- ✅ Command: `--pretrained` flag or use `finetune` config
- ⏱️ Time: ~1-2 hours per CNN, ~2-4 hours per ViT

### From Scratch
- ✅ Use these when: Testing full training pipeline attacks, maximum control
- ✅ Benefits: Full attack surface, no pre-learned biases, some defenses require this
- ✅ Command: No `--pretrained` flag or use `scratch` config
- ⏱️ Time: ~2-3 hours per CNN, ~4-6 hours per ViT

**For thesis**: Train both to show comprehensive analysis!

## 📋 Model Architecture Details

### Available Models
| Model | Parameters | Training Time (scratch) | Training Time (finetune) | Best Use Case |
|-------|------------|------------------------|-------------------------|---------------|
| VGG16 | ~15M | ~2-3 hours | ~1-1.5 hours | Strong baseline |
| VGG19 | ~20M | ~3-4 hours | ~1.5-2 hours | Deeper baseline |
| ResNet18 | ~11M | ~2-3 hours | ~1-1.5 hours | Fast, efficient |
| ResNet34 | ~21M | ~3-4 hours | ~1.5-2 hours | Balanced |
| ResNet50 | ~25M | ~4-5 hours | ~2-3 hours | Strong performance |
| ViT-Small | ~22M | ~4-6 hours | ~2-3 hours | ViT baseline |
| ViT-Base | ~86M | ~6-8 hours | ~3-4 hours | Best ViT |
| DeiT-Small | ~22M | ~4-6 hours | ~2-3 hours | Distilled ViT |
| DeiT-Base | ~86M | ~6-8 hours | ~3-4 hours | Best DeiT |

Times are for RTX 4090 with CIFAR-10. GTSRB may take 20-30% longer due to more classes.

## 🚨 Common Issues & Solutions

### Issue: "No module named 'timm'"
```powershell
conda activate backdoor-toolbox
pip install timm
```

### Issue: CUDA out of memory
```powershell
# Add to any training command:
--batch-size 32
# Or even smaller:
--batch-size 16
```

### Issue: Training is slow
```powershell
# Check GPU usage:
nvidia-smi

# Make sure using GPU:
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

### Issue: GTSRB not found
1. Download from: https://benchmark.ini.rub.de/gtsrb_dataset.html
2. Extract to: `data/gtsrb/`
3. Verify structure:
```
data/gtsrb/
├── Train/00000/ ... 00042/
├── Test/
└── GT-final_test.csv
```

## 📁 Important Files & Locations

### Your Code (Can Modify)
- `src/models.py` - Model architectures
- `src/datasets.py` - Dataset loaders
- `src/config.py` - Training configurations
- `scripts/training/train.py` - Main training script

### Generated During Training
- `results/models/*` - Saved models and logs
- `results/models/*/best_model.pth` - Best checkpoint
- `results/models/*/training_log.json` - Training history

### Read-Only (Do Not Modify)
- `backdoor-toolbox-main/` - Original toolbox

### Documentation
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `SETUP_SUMMARY.md` - What was set up
- `GETTING_STARTED.md` - This file

## 🎯 Your Immediate Action Plan

1. **Right Now (5 minutes)**:
   ```powershell
   conda activate backdoor-toolbox
   pip install timm
   python test_setup.py
   ```

2. **Today (15 minutes)**:
   ```powershell
   # Quick test to verify everything works
   python scripts/training/train.py --model resnet18 --dataset cifar10 --epochs 5
   ```

3. **Tonight (Start Overnight)**:
   ```powershell
   # Start your first batch training
   python scripts/training/batch_train.py --experiment cifar10_scratch
   ```

4. **Tomorrow**:
   - Check results with `python scripts/utils/list_models.py`
   - Start next batch: `python scripts/training/batch_train.py --experiment cifar10_finetune`

5. **This Week**:
   - Complete all CIFAR-10 training
   - Download and set up GTSRB
   - Start GTSRB training

6. **Next Week**:
   - Begin backdoor attack experiments
   - Use trained models with backdoor-toolbox-main

## 🎉 You're All Set!

Everything is configured and ready to go. Just:
1. ✅ Activate environment: `conda activate backdoor-toolbox`
2. ✅ Verify with: `python test_setup.py`
3. ✅ Start training: `python scripts/training/train.py --model resnet18 --dataset cifar10 --epochs 5`

Good luck with your comprehensive backdoor attack analysis! 🚀
