"""
Configuration definitions for training experiments.
Provides default configurations for different scenarios.
"""

# Base configuration
BASE_CONFIG = {
    'batch_size': 128,
    'num_workers': 4,
    'seed': 42,
    'gpu': 0,
    'data_dir': './data',
    'output_dir': './results/models',
    'tensorboard': True,
    'save_interval': 50,
}

# Training from scratch configuration
SCRATCH_CONFIG = {
    **BASE_CONFIG,
    'pretrained': False,
    'epochs': 200,
    'lr': 0.01,  # Lower LR to prevent gradient explosion
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'scheduler': 'multistep',
    'augmentation': True,
    'patience': 10,  # Early stopping after 10 epochs without improvement
    'max_grad_norm': 5.0,  # Gradient clipping to prevent explosion
}

# Fine-tuning pre-trained models configuration
FINETUNE_CONFIG = {
    **BASE_CONFIG,
    'pretrained': True,
    'epochs': 100,
    'lr': 0.01,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'scheduler': 'multistep',
    'augmentation': True,
    'patience': 7  # Early stopping for fine-tuning
}

# Vision Transformer configuration (requires more epochs and different optimizer)
VIT_SCRATCH_CONFIG = {
    **BASE_CONFIG,
    'pretrained': False,
    'epochs': 300,
    'lr': 0.001,
    'optimizer': 'adamw',
    'weight_decay': 0.05,
    'scheduler': 'cosine',
    'augmentation': True,
    'batch_size': 64,  # Smaller batch size for ViT
}

VIT_FINETUNE_CONFIG = {
    **BASE_CONFIG,
    'pretrained': True,
    'epochs': 50,
    'lr': 0.0001,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'scheduler': 'cosine',
    'augmentation': True,
    'batch_size': 64,
    'patience': 5  # Early stopping for ViT
}

# Quick test configuration (for debugging)
TEST_CONFIG = {
    **BASE_CONFIG,
    'pretrained': False,
    'epochs': 5,
    'lr': 0.1,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'scheduler': 'none',
    'augmentation': True,
    'batch_size': 32,
}


def get_config(config_name='scratch'):
    """
    Get configuration by name.
    
    Args:
        config_name: One of 'scratch', 'finetune', 'vit_scratch', 'vit_finetune', 'test'
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'scratch': SCRATCH_CONFIG,
        'finetune': FINETUNE_CONFIG,
        'vit_scratch': VIT_SCRATCH_CONFIG,
        'vit_finetune': VIT_FINETUNE_CONFIG,
        'test': TEST_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name].copy()


# Experiment definitions
EXPERIMENTS = {
    # CIFAR-10 from scratch
    'cifar10_scratch': {
        'dataset': 'cifar10',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'scratch',
    },
    
    # CIFAR-10 fine-tune
    'cifar10_finetune': {
        'dataset': 'cifar10',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'finetune',
    },
    
    # CIFAR-10 ViT
    'cifar10_vit': {
        'dataset': 'cifar10',
        'models': ['vit_small', 'vit_base', 'deit_small', 'deit_base'],
        'config': 'vit_finetune',  # Always use pre-trained ViT
    },
    
    # GTSRB from scratch
    'gtsrb_scratch': {
        'dataset': 'gtsrb',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'scratch',
    },
    
    # GTSRB fine-tune
    'gtsrb_finetune': {
        'dataset': 'gtsrb',
        'models': ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'],
        'config': 'finetune',
    },
    
    # GTSRB ViT
    'gtsrb_vit': {
        'dataset': 'gtsrb',
        'models': ['vit_small', 'vit_base', 'deit_small', 'deit_base'],
        'config': 'vit_finetune',
    },
}


def get_experiment(experiment_name):
    """Get experiment definition by name."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[experiment_name].copy()


def list_experiments():
    """List all available experiments."""
    return list(EXPERIMENTS.keys())


if __name__ == '__main__':
    import json
    
    print("Available configurations:")
    print("-" * 50)
    for name in ['scratch', 'finetune', 'vit_scratch', 'vit_finetune', 'test']:
        config = get_config(name)
        print(f"\n{name}:")
        print(json.dumps(config, indent=2))
    
    print("\n\nAvailable experiments:")
    print("-" * 50)
    for exp_name in list_experiments():
        exp = get_experiment(exp_name)
        print(f"\n{exp_name}:")
        print(f"  Dataset: {exp['dataset']}")
        print(f"  Models: {', '.join(exp['models'])}")
        print(f"  Config: {exp['config']}")
