"""
Utilities for model evaluation and testing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to use
        criterion: Loss function (optional)
        return_predictions: If True, returns all predictions and targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader) if criterion is not None else None
    
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }
    
    if avg_loss is not None:
        results['loss'] = avg_loss
    
    if return_predictions:
        results['predictions'] = np.array(all_predictions)
        results['targets'] = np.array(all_targets)
    
    return results


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(targets, predictions)


def compute_per_class_accuracy(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute per-class accuracy."""
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    
    for pred, target in zip(predictions, targets):
        per_class_total[target] += 1
        if pred == target:
            per_class_correct[target] += 1
    
    # Avoid division by zero
    per_class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_accuracy[i] = 100. * per_class_correct[i] / per_class_total[i]
    
    return per_class_accuracy


def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[list] = None
):
    """Print detailed classification report."""
    print("\nClassification Report:")
    print("=" * 80)
    print(classification_report(targets, predictions, target_names=class_names))


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        Epoch number and metrics from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', -1)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


def count_correct_samples(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[int, int]:
    """
    Count correctly classified samples.
    
    Returns:
        (correct, total)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return correct, total


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from models import get_model
    from datasets import get_dataloader, get_num_classes, get_class_names
    
    print("Testing evaluation utilities...\n")
    
    # Test with a simple model
    model = get_model('resnet18', num_classes=10, pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get test data
    test_loader = get_dataloader('cifar10', train=False, batch_size=128)
    
    print("Evaluating model on CIFAR-10 test set...")
    results = evaluate_model(
        model,
        test_loader,
        device,
        criterion=nn.CrossEntropyLoss(),
        return_predictions=True
    )
    
    print(f"\nAccuracy: {results['accuracy']:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"Loss: {results['loss']:.4f}")
    
    # Compute per-class accuracy
    per_class_acc = compute_per_class_accuracy(
        results['predictions'],
        results['targets'],
        num_classes=10
    )
    
    class_names = get_class_names('cifar10')
    print("\nPer-class accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"  {name:15s}: {acc:.2f}%")
