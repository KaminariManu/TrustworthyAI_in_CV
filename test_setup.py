"""
Test script to verify installation and setup.
Run this after environment setup to ensure everything works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import timm
        print(f"✓ timm {timm.__version__}")
    except ImportError as e:
        print(f"✗ timm import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError as e:
        print(f"✗ tqdm import failed: {e}")
        return False
    
    print()
    return True


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA is not available (CPU only)")
        print("  Training will be much slower on CPU")
    
    print()
    return True


def test_src_modules():
    """Test that custom modules can be imported."""
    print("Testing custom modules...")
    
    try:
        from models import get_model, count_parameters
        print("✓ models.py")
    except ImportError as e:
        print(f"✗ models.py import failed: {e}")
        return False
    
    try:
        from datasets import get_dataloader, get_num_classes
        print("✓ datasets.py")
    except ImportError as e:
        print(f"✗ datasets.py import failed: {e}")
        return False
    
    try:
        from config import get_config, list_experiments
        print("✓ config.py")
    except ImportError as e:
        print(f"✗ config.py import failed: {e}")
        return False
    
    try:
        from evaluation import evaluate_model
        print("✓ evaluation.py")
    except ImportError as e:
        print(f"✗ evaluation.py import failed: {e}")
        return False
    
    print()
    return True


def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    import torch
    from models import get_model, count_parameters
    
    models_to_test = [
        ('resnet18', False),
        ('vgg16', False),
        ('resnet18', True),
    ]
    
    for model_name, pretrained in models_to_test:
        try:
            model = get_model(model_name, num_classes=10, pretrained=pretrained)
            total, trainable = count_parameters(model)
            mode = 'pretrained' if pretrained else 'scratch'
            print(f"✓ {model_name} ({mode}): {total:,} parameters")
        except Exception as e:
            print(f"✗ {model_name} creation failed: {e}")
            return False
    
    print()
    return True


def test_forward_pass():
    """Test forward pass through a model."""
    print("Testing forward pass...")
    
    import torch
    from models import get_model
    
    try:
        model = get_model('resnet18', num_classes=10, pretrained=False)
        model.eval()
        
        # Create dummy input (batch of 2, 3 channels, 32x32)
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            y = model(x)
        
        if y.shape == (2, 10):
            print(f"✓ Forward pass successful: input {x.shape} -> output {y.shape}")
        else:
            print(f"✗ Unexpected output shape: {y.shape}")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    print()
    return True


def test_dataset_loading():
    """Test dataset loading (CIFAR-10 only, as GTSRB requires manual download)."""
    print("Testing dataset loading...")
    
    try:
        from datasets import get_dataloader
        
        print("Loading CIFAR-10 (will download if not present)...")
        train_loader = get_dataloader('cifar10', train=True, batch_size=32, download=True)
        test_loader = get_dataloader('cifar10', train=False, batch_size=32, download=True)
        
        print(f"✓ CIFAR-10 train: {len(train_loader)} batches")
        print(f"✓ CIFAR-10 test: {len(test_loader)} batches")
        
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"✓ Batch shape: {images.shape}, labels: {labels.shape}")
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    print()
    return True


def test_directories():
    """Test that required directories exist or can be created."""
    print("Testing directory structure...")
    
    required_dirs = [
        'data',
        'results',
        'results/models',
        'results/figures',
        'results/tables',
        'scripts/training',
        'scripts/evaluation',
        'scripts/utils',
        'src',
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ {dir_path} (created)")
            except Exception as e:
                print(f"✗ {dir_path} (failed to create): {e}")
                return False
    
    print()
    return True


def main():
    print("=" * 80)
    print("Backdoor Attack Analysis Framework - Setup Test")
    print("=" * 80)
    print()
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_cuda()
    all_passed &= test_directories()
    all_passed &= test_src_modules()
    all_passed &= test_model_creation()
    all_passed &= test_forward_pass()
    all_passed &= test_dataset_loading()
    
    # Summary
    print("=" * 80)
    if all_passed:
        print("✓ All tests passed! Setup is complete.")
        print()
        print("Next steps:")
        print("1. If using GTSRB, download the dataset manually (see QUICKSTART.md)")
        print("2. Start training: python scripts/training/train.py --model resnet18 --dataset cifar10 --epochs 5")
        print("3. Or see QUICKSTART.md for more examples")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print()
        print("Common issues:")
        print("- Missing packages: conda env create -f environment.yml")
        print("- Wrong environment: conda activate backdoor-toolbox")
        print("- Import errors: Make sure you're in the project root directory")
    print("=" * 80)


if __name__ == '__main__':
    main()
