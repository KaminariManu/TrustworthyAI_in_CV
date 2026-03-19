"""
Model architectures for backdoor attack analysis.
Supports both CIFAR-10 (32x32) and GTSRB (32x32) datasets.

Uses standardized implementations from:
- torchvision: Pre-trained ImageNet models
- timm: Well-maintained model library with many variants

Includes:
- VGG models (VGG16, VGG19)
- ResNet models (ResNet18, ResNet34, ResNet50)
- Vision Transformers (ViT, DeiT)

Each model can be:
1. Trained from scratch - uses timm/torchvision with CIFAR adaptations
2. Fine-tuned from ImageNet pre-trained weights (with adaptive input size)

For 32x32 images (CIFAR-10/GTSRB), models are adapted by:
- Replacing aggressive 7x7 stride=2 conv with 3x3 stride=1
- Removing initial maxpool layer
This is standard practice in CIFAR research.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional, Tuple


class AdaptiveInputWrapper(nn.Module):
    """
    Wrapper to adapt 32x32 images to 224x224 for pre-trained models.
    Uses interpolation to resize inputs.
    """
    def __init__(self, model: nn.Module, target_size: int = 224):
        super().__init__()
        self.model = model
        self.target_size = target_size
        
    def forward(self, x):
        # Resize from 32x32 to target_size (e.g., 224x224)
        if x.shape[-1] != self.target_size:
            x = nn.functional.interpolate(
                x, size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            )
        return self.model(x)


def _adapt_resnet_for_cifar(model: nn.Module) -> nn.Module:
    """
    Adapt a standard ResNet for 32x32 CIFAR images.
    Replaces the aggressive initial downsampling with CIFAR-friendly layers.
    
    Standard ResNet starts with:
    - 7x7 conv, stride=2 (reduces 224→112)
    - maxpool 3x3, stride=2 (reduces 112→56)
    
    For CIFAR (32x32), we need:
    - 3x3 conv, stride=1
    - no maxpool
    """
    # Replace first conv: 7x7 stride=2 -> 3x3 stride=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool (set to identity)
    model.maxpool = nn.Identity()
    
    return model


def _adapt_vgg_for_cifar(model: nn.Module) -> nn.Module:
    """
    Adapt VGG for 32x32 images.
    VGG is less aggressive with downsampling, so fewer changes needed.
    Main issue is the FC layers expect specific dimensions.
    """
    # VGG for CIFAR works reasonably well out of the box
    # The main adaptation needed is in the classifier
    # timm's VGG should handle this with num_classes parameter
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = False,
    dataset: str = 'cifar10'
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model (vgg16, vgg19, resnet18, resnet34, resnet50, vit, deit)
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pre-trained weights
        dataset: Dataset name (cifar10 or gtsrb) - affects architecture choice
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    # Use timm library for better, standardized implementations
    # timm has many pre-configured variants and is better maintained
    
    # VGG Models
    if model_name == 'vgg16':
        if pretrained:
            # Use torchvision pre-trained model with adaptive input
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model.classifier[6] = nn.Linear(4096, num_classes)
            model = AdaptiveInputWrapper(model)
        else:
            # Use timm's VGG implementation
            model = timm.create_model('vgg16_bn', pretrained=False, num_classes=num_classes)
            model = _adapt_vgg_for_cifar(model)
    
    elif model_name == 'vgg19':
        if pretrained:
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            model.classifier[6] = nn.Linear(4096, num_classes)
            model = AdaptiveInputWrapper(model)
        else:
            model = timm.create_model('vgg19_bn', pretrained=False, num_classes=num_classes)
            model = _adapt_vgg_for_cifar(model)
    
    # ResNet Models - Use timm for better variants
    elif model_name == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(512, num_classes)
            model = AdaptiveInputWrapper(model)
        else:
            # timm has good ResNet variants for CIFAR
            # Using standard ResNet but will adapt for small images
            model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
            model = _adapt_resnet_for_cifar(model)
    
    elif model_name == 'resnet34':
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(512, num_classes)
            model = AdaptiveInputWrapper(model)
        else:
            model = timm.create_model('resnet34', pretrained=False, num_classes=num_classes)
            model = _adapt_resnet_for_cifar(model)
    
    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(2048, num_classes)
            model = AdaptiveInputWrapper(model)
        else:
            model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
            model = _adapt_resnet_for_cifar(model)
    
    # Vision Transformer Models (using timm library)
    elif model_name == 'vit_base' or model_name == 'vit':
        # ViT requires 224x224 input, so always use adaptive wrapper
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        model = AdaptiveInputWrapper(model)
    
    elif model_name == 'deit' or model_name == 'deit_base':
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        model = AdaptiveInputWrapper(model)
    
    elif model_name == 'vit_small':
        model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        model = AdaptiveInputWrapper(model)
    
    elif model_name == 'deit_small':
        model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        model = AdaptiveInputWrapper(model)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: "
                         "vgg16, vgg19, resnet18, resnet34, resnet50, "
                         "vit_base, vit_small, deit_base, deit_small")
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...\n")
    
    models_to_test = [
        ('vgg16', False),
        ('vgg16', True),
        ('resnet18', False),
        ('resnet18', True),
        ('resnet50', False),
        ('vit_base', True),
        ('deit_base', True),
    ]
    
    for model_name, pretrained in models_to_test:
        model = get_model(model_name, num_classes=10, pretrained=pretrained)
        total, trainable = count_parameters(model)
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        
        print(f"{model_name} (pretrained={pretrained}):")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Output shape: {y.shape}")
        print()
