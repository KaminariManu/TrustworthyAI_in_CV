# Setup Summary - Backdoor Attack Analysis Framework

## ✅ What Has Been Set Up

### 1. **Model Architectures** (`src/models.py`)
- **VGG**: VGG16, VGG19
  - Custom implementations optimized for 32×32 images
  - Pre-trained versions with adaptive input resizing
- **ResNet**: ResNet18, ResNet34, ResNet50
  - Modified architecture for small images (no initial maxpool)
  - Pre-trained ImageNet weights support
- **Vision Transformers**: ViT (base/small), DeiT (base/small)
  - Using `timm` library for state-of-the-art ViT models
  - Automatic input resizing from 32×32 to 224×224

**Key Features:**
- Unified interface: `get_model(model_name, num_classes, pretrained, dataset)`
- Automatic adaptation between 32×32 (CIFAR/GTSRB) and 224×224 (ImageNet pre-trained)
- Parameter counting utilities

### 2. **Dataset Loaders** (`src/datasets.py`)
- **CIFAR-10**: 10 classes, 32×32 RGB images
  - Automatic download
  - Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
- **GTSRB**: 43 classes, 32×32 RGB images (German Traffic Signs)
  - Manual download required
  - Normalization: mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]

**Key Features:**
- Unified interface: `get_dataloader(dataset_name, batch_size, train, ...)`
- Configurable data augmentation (random crop, horizontal flip)
- Class names and metadata utilities

### 3. **Training Scripts**

#### Single Model Training (`scripts/training/train.py`)
- Comprehensive training loop with validation
- Configurable hyperparameters (lr, epochs, optimizer, scheduler)
- TensorBoard logging
- Automatic checkpoint saving (best model, final model, periodic)
- Training log saved as JSON

#### Batch Training (`scripts/training/batch_train.py`)
- Train multiple models sequentially
- Predefined experiment configurations
- Custom experiment support
- Dry-run mode for testing
- Continue-on-error option

### 4. **Configuration System** (`src/config.py`)
Pre-defined training configurations:
- **scratch**: CNNs trained from random initialization (200 epochs)
- **finetune**: Fine-tuning pre-trained models (100 epochs)
- **vit_scratch**: ViTs from scratch (300 epochs, AdamW)
- **vit_finetune**: Fine-tuning pre-trained ViTs (50 epochs)
- **test**: Quick testing (5 epochs)

Pre-defined experiments:
- `cifar10_scratch`, `cifar10_finetune`, `cifar10_vit`
- `gtsrb_scratch`, `gtsrb_finetune`, `gtsrb_vit`

### 5. **Evaluation Tools**

#### Model Evaluation (`scripts/evaluation/evaluate_model.py`)
- Load trained models and evaluate on test sets
- Per-class accuracy analysis
- Confusion matrix and classification reports
- Export results to JSON

#### Evaluation Utilities (`src/evaluation.py`)
- `evaluate_model()`: Comprehensive evaluation function
- `load_checkpoint()`: Load model weights and training state
- `compute_per_class_accuracy()`: Detailed per-class metrics
- `print_classification_report()`: Formatted reports

### 6. **Management Utilities**

#### List Models (`scripts/utils/list_models.py`)
- List all trained models with metadata
- Filter by model, dataset, or training mode
- Compare models and show rankings
- Generate summary statistics
- Export to CSV

### 7. **Documentation**
- **README.md**: Comprehensive documentation
- **QUICKSTART.md**: Quick start guide with examples
- **test_setup.py**: Automated verification script
- **setup.bat**: Windows setup script

### 8. **Project Structure**
```
├── src/                          # Source code
│   ├── models.py                 # Model architectures
│   ├── datasets.py               # Dataset loaders
│   ├── config.py                 # Configurations
│   ├── evaluation.py             # Evaluation utilities
│   └── __init__.py               # Package initialization
│
├── scripts/                      # Executable scripts
│   ├── training/
│   │   ├── train.py             # Single model training
│   │   └── batch_train.py       # Batch training
│   ├── evaluation/
│   │   └── evaluate_model.py    # Model evaluation
│   └── utils/
│       └── list_models.py       # Model management
│
├── data/                         # Datasets (auto-created)
│   ├── cifar10/                 # CIFAR-10 (auto-download)
│   └── gtsrb/                   # GTSRB (manual download)
│
├── results/                      # Experiment outputs
│   ├── models/                  # Trained models
│   ├── figures/                 # Visualizations
│   └── tables/                  # Results tables
│
├── backdoor-toolbox-main/       # Original toolbox (unchanged)
│
├── environment.yml               # Conda environment
├── test_setup.py                # Setup verification
├── setup.bat                    # Windows setup script
├── README.md                    # Main documentation
├── QUICKSTART.md                # Quick start guide
└── SETUP_SUMMARY.md             # This file
```

## 🚀 How to Get Started

### Step 1: Create Environment
```bash
conda env create -f environment.yml
conda activate backdoor-toolbox
```

### Step 2: Verify Installation
```bash
python test_setup.py
```

### Step 3: (Optional) Download GTSRB
If you want to use GTSRB:
1. Download from https://benchmark.ini.rub.de/gtsrb_dataset.html
2. Extract to `data/gtsrb/`

### Step 4: Start Training
```bash
# Quick test (5 epochs)
python scripts/training/train.py --model resnet18 --dataset cifar10 --epochs 5

# Full training
python scripts/training/batch_train.py --experiment cifar10_scratch
```

## 📊 Recommended Training Strategy

For comprehensive backdoor attack analysis:

### Phase 1: CIFAR-10 (Start Here)
```bash
# CNNs from scratch (5 models × ~2-3 hours = ~10-15 hours)
python scripts/training/batch_train.py --experiment cifar10_scratch

# CNNs fine-tuned (5 models × ~1-1.5 hours = ~5-8 hours)
python scripts/training/batch_train.py --experiment cifar10_finetune

# Vision Transformers (4 models × ~2-4 hours = ~8-16 hours)
python scripts/training/batch_train.py --experiment cifar10_vit
```

### Phase 2: GTSRB (More Challenging)
```bash
# Similar commands for GTSRB
python scripts/training/batch_train.py --experiment gtsrb_scratch
python scripts/training/batch_train.py --experiment gtsrb_finetune
python scripts/training/batch_train.py --experiment gtsrb_vit
```

**Total Models**: 28 (14 per dataset)
**Total Time**: ~60-80 hours on RTX 4090 (can run overnight/weekends)

## 💡 Key Design Decisions

### 1. Pre-trained vs From Scratch
**Both are supported** because:
- Pre-trained: More realistic, faster convergence, tests transfer learning scenarios
- From scratch: Full control, tests complete attack surface, important for some defenses

### 2. Input Size Handling
- 32×32 images (CIFAR-10/GTSRB) are automatically resized to 224×224 for pre-trained models
- Custom architectures optimized for 32×32 when training from scratch
- Wrapped in `AdaptiveInputWrapper` for seamless handling

### 3. Modular Design
- Each component (models, datasets, training, evaluation) is independent
- Easy to extend with new models or datasets
- Compatible with backdoor-toolbox-main (kept unchanged)

### 4. Comprehensive Logging
- TensorBoard for real-time monitoring
- JSON logs for programmatic analysis
- Multiple checkpoint types (best, final, periodic)

## 🎯 Next Steps After Training

1. **List trained models**:
   ```bash
   python scripts/utils/list_models.py --compare --summary
   ```

2. **Evaluate models**:
   ```bash
   python scripts/evaluation/evaluate_model.py \
       --checkpoint results/models/YOUR_MODEL/best_model.pth \
       --detailed
   ```

3. **Apply backdoor attacks** using backdoor-toolbox-main:
   - Use trained models as victims
   - Test various attack methods (BadNets, Blend, WaNet, etc.)
   - Measure attack success rate (ASR)

4. **Apply defenses**:
   - Test detection methods (Activation Clustering, STRIP, etc.)
   - Test mitigation methods (Fine-Pruning, Neural Cleanse, etc.)

5. **Analysis**:
   - Compare attack success across models
   - Analyze pre-trained vs scratch vulnerability
   - Generate visualizations and tables for thesis

## 📝 Important Notes

1. **GPU Memory**: Vision Transformers require more memory. Reduce batch size if needed (`--batch-size 32`)

2. **CIFAR-10 Auto-Download**: Downloads automatically on first use (~170 MB)

3. **GTSRB Manual Download**: Must download manually (~300 MB train + 90 MB test)

4. **Checkpoint Files**: Models saved in `results/models/` can be large (100-500 MB each)

5. **TensorBoard**: View training progress at http://localhost:6006
   ```bash
   tensorboard --logdir results/models
   ```

6. **Training Time**: Estimates are for RTX 4090. Adjust expectations for your GPU.

## 🔧 Customization Examples

### Custom Model Subset
```bash
python scripts/training/batch_train.py \
    --models vgg16 resnet18 resnet50 \
    --dataset cifar10 \
    --config scratch
```

### Custom Hyperparameters
```bash
python scripts/training/train.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 100 \
    --lr 0.05 \
    --batch-size 256 \
    --optimizer adamw
```

### Specific GPU
```bash
python scripts/training/batch_train.py \
    --experiment cifar10_scratch \
    --gpu 0
```

## 📚 Code Examples

### Load and Evaluate Model
```python
import torch
from src import get_model, get_dataloader, evaluate_model, load_checkpoint

# Load model
model = get_model('resnet18', num_classes=10)
load_checkpoint('results/models/MODEL_DIR/best_model.pth', model)

# Evaluate
device = torch.device('cuda')
model = model.to(device)
test_loader = get_dataloader('cifar10', train=False)
results = evaluate_model(model, test_loader, device)

print(f"Accuracy: {results['accuracy']:.2f}%")
```

### Create Custom Dataset
```python
from src.datasets import get_transforms

# Get transforms for your custom dataset
transform = get_transforms('cifar10', train=True, augmentation=True)

# Use with your dataset
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
    # ... implement __getitem__ and __len__
```

## ❓ Troubleshooting

**Issue**: ModuleNotFoundError: No module named 'timm'
**Solution**: 
```bash
conda activate backdoor-toolbox
pip install timm
```

**Issue**: CUDA out of memory
**Solution**: Reduce batch size with `--batch-size 64` or `--batch-size 32`

**Issue**: GTSRB not found
**Solution**: Download manually from official website and extract to `data/gtsrb/`

**Issue**: Training is slow
**Solution**: Check GPU is being used: `--gpu 0`. If on CPU, training will be 10-50x slower.

## 📧 Support

- Check QUICKSTART.md for quick examples
- Check README.md for detailed documentation
- Review code comments in src/ files
- Test with `python test_setup.py`

## ✨ Summary

You now have a complete framework for:
- ✅ Training 6 model architectures (VGG16/19, ResNet18/34/50, ViT/DeiT)
- ✅ On 2 datasets (CIFAR-10, GTSRB)
- ✅ With 2 training modes (from scratch, fine-tuned)
- ✅ = 28 base models for comprehensive backdoor analysis

**All ready to use! Just activate the environment and start training.**
