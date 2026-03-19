"""
Source package for backdoor attack analysis.
Provides models, datasets, and utilities for comprehensive backdoor attack research.
"""

from .models import get_model, count_parameters
from .datasets import get_dataloader, get_dataset, get_num_classes, get_class_names
from .config import get_config, get_experiment, list_experiments
from .evaluation import evaluate_model, load_checkpoint

__version__ = '1.0.0'

__all__ = [
    'get_model',
    'count_parameters',
    'get_dataloader',
    'get_dataset',
    'get_num_classes',
    'get_class_names',
    'get_config',
    'get_experiment',
    'list_experiments',
    'evaluate_model',
    'load_checkpoint',
]
