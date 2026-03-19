"""
Dataset loaders for CIFAR-10 and GTSRB.
Provides unified interface for loading and preprocessing data.
"""

import os
from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class GTSRBDataset(Dataset):
    """
    German Traffic Sign Recognition Benchmark (GTSRB) dataset.
    
    The dataset should be organized as:
    data/gtsrb/
        Train/
            00000/  (class 0)
                00000_00000.ppm
                ...
            00001/  (class 1)
                ...
        Test/
            00000.ppm
            00001.ppm
            ...
        GT-final_test.csv  (test labels)
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        """
        Args:
            root: Root directory of dataset
            train: If True, creates dataset from training set, otherwise test set
            transform: A function/transform to apply to the PIL image
            download: If True, downloads the dataset (not implemented - manual download required)
        """
        self.root = root
        self.train = train
        self.transform = transform
        
        if train:
            self.data_dir = os.path.join(root, 'Train')
            self.samples = self._load_train_data()
        else:
            self.data_dir = os.path.join(root, 'Test')
            self.samples = self._load_test_data()
    
    def _load_train_data(self):
        """Load training data from class directories."""
        samples = []
        
        if not os.path.exists(self.data_dir):
            raise RuntimeError(
                f"GTSRB training data not found at {self.data_dir}. "
                "Please download from https://benchmark.ini.rub.de/gtsrb_dataset.html"
            )
        
        # Iterate through class directories (00000 to 00042)
        for class_id in range(43):
            class_dir = os.path.join(self.data_dir, f'{class_id:05d}')
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.ppm') or img_name.endswith('.png'):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_id))
        
        return samples
    
    def _load_test_data(self):
        """Load test data with labels from CSV file."""
        import csv
        samples = []
        
        if not os.path.exists(self.data_dir):
            raise RuntimeError(
                f"GTSRB test data not found at {self.data_dir}. "
                "Please download from https://benchmark.ini.rub.de/gtsrb_dataset.html"
            )
        
        # Load test labels from CSV
        csv_path = os.path.join(self.root, 'GT-final_test.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)  # Skip header
                for row in reader:
                    img_name = row[0]
                    class_id = int(row[7])
                    img_path = os.path.join(self.data_dir, img_name)
                    if os.path.exists(img_path):
                        samples.append((img_path, class_id))
        else:
            # Fallback: load all images without labels
            for img_name in sorted(os.listdir(self.data_dir)):
                if img_name.endswith('.ppm') or img_name.endswith('.png'):
                    img_path = os.path.join(self.data_dir, img_name)
                    samples.append((img_path, -1))  # Unknown label
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize to 32x32
        img = img.resize((32, 32), Image.BILINEAR)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


def get_transforms(
    dataset: str = 'cifar10',
    train: bool = True,
    augmentation: bool = True
) -> transforms.Compose:
    """
    Get data transforms for a specific dataset.
    
    Args:
        dataset: Dataset name ('cifar10' or 'gtsrb')
        train: If True, returns training transforms, otherwise test transforms
        augmentation: If True and train=True, applies data augmentation
    
    Returns:
        Composed transforms
    """
    dataset = dataset.lower()
    
    # Dataset-specific normalization
    if dataset == 'cifar10':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261]
        )
    elif dataset == 'gtsrb':
        normalize = transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],
            std=[0.2672, 0.2564, 0.2629]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if train and augmentation:
        # Training with augmentation
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Test or training without augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    return transform


def get_dataset(
    dataset_name: str = 'cifar10',
    data_dir: str = './data',
    train: bool = True,
    download: bool = False,
    augmentation: bool = True
) -> Dataset:
    """
    Get dataset object.
    
    Args:
        dataset_name: Name of dataset ('cifar10' or 'gtsrb')
        data_dir: Root directory for data
        train: If True, loads training set, otherwise test set
        download: If True, downloads dataset (only for CIFAR-10)
        augmentation: If True, applies data augmentation to training set
    
    Returns:
        PyTorch Dataset
    """
    dataset_name = dataset_name.lower()
    transform = get_transforms(dataset_name, train=train, augmentation=augmentation)
    
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=os.path.join(data_dir, 'cifar10'),
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_name == 'gtsrb':
        dataset = GTSRBDataset(
            root=os.path.join(data_dir, 'gtsrb'),
            train=train,
            transform=transform,
            download=download
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_dataloader(
    dataset_name: str = 'cifar10',
    data_dir: str = './data',
    batch_size: int = 128,
    train: bool = True,
    shuffle: bool = None,
    num_workers: int = 4,
    download: bool = False,
    augmentation: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    Get DataLoader for training or testing.
    
    Args:
        dataset_name: Name of dataset ('cifar10' or 'gtsrb')
        data_dir: Root directory for data
        batch_size: Batch size
        train: If True, loads training set, otherwise test set
        shuffle: Whether to shuffle data. If None, uses True for train, False for test
        num_workers: Number of data loading workers
        download: If True, downloads dataset (only for CIFAR-10)
        augmentation: If True, applies data augmentation to training set
        pin_memory: If True, pins memory for faster GPU transfer
    
    Returns:
        PyTorch DataLoader
    """
    if shuffle is None:
        shuffle = train
    
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=train,
        download=download,
        augmentation=augmentation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for a dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'gtsrb':
        return 43
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_class_names(dataset_name: str) -> list:
    """Get class names for a dataset."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    elif dataset_name == 'gtsrb':
        # GTSRB class names (speed limits, prohibitory signs, etc.)
        return [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
            'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
            'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
            'No passing', 'No passing for vehicles over 3.5 metric tons',
            'Right-of-way at the next intersection', 'Priority road', 'Yield',
            'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
            'No entry', 'General caution', 'Dangerous curve to the left',
            'Dangerous curve to the right', 'Double curve', 'Bumpy road',
            'Slippery road', 'Road narrows on the right', 'Road work',
            'Traffic signals', 'Pedestrians', 'Children crossing',
            'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
            'End of all speed and passing limits', 'Turn right ahead',
            'Turn left ahead', 'Ahead only', 'Go straight or right',
            'Go straight or left', 'Keep right', 'Keep left',
            'Roundabout mandatory', 'End of no passing',
            'End of no passing by vehicles over 3.5 metric tons'
        ]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    # Test data loading
    print("Testing dataset loaders...\n")
    
    # Test CIFAR-10
    print("=" * 50)
    print("CIFAR-10")
    print("=" * 50)
    
    try:
        train_loader = get_dataloader('cifar10', train=True, batch_size=128)
        test_loader = get_dataloader('cifar10', train=False, batch_size=128)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Number of classes: {get_num_classes('cifar10')}")
        print()
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}\n")
    
    # Test GTSRB
    print("=" * 50)
    print("GTSRB")
    print("=" * 50)
    
    try:
        train_loader = get_dataloader('gtsrb', train=True, batch_size=128)
        test_loader = get_dataloader('gtsrb', train=False, batch_size=128)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Number of classes: {get_num_classes('gtsrb')}")
        print()
    except Exception as e:
        print(f"Error loading GTSRB: {e}\n")
        print("GTSRB data may need to be downloaded manually.")
        print("Download from: https://benchmark.ini.rub.de/gtsrb_dataset.html")
