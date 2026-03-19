"""
Download and prepare GTSRB dataset.

This script downloads the German Traffic Sign Recognition Benchmark (GTSRB) dataset
and prepares it in the format expected by our training pipeline.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_gtsrb(data_dir='data/gtsrb'):
    """
    Download and extract GTSRB dataset.
    
    Args:
        data_dir: Directory to save the dataset
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GTSRB Dataset Download and Preparation")
    print("=" * 80)
    
    # URLs for GTSRB dataset
    train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    test_labels_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
    
    # Download training data
    train_zip = data_dir / "GTSRB_Final_Training_Images.zip"
    if not train_zip.exists():
        print("\n📥 Downloading training images (263 MB)...")
        download_url(train_url, train_zip)
    else:
        print("\n✓ Training images already downloaded")
    
    # Download test data
    test_zip = data_dir / "GTSRB_Final_Test_Images.zip"
    if not test_zip.exists():
        print("\n📥 Downloading test images (84 MB)...")
        download_url(test_url, test_zip)
    else:
        print("\n✓ Test images already downloaded")
    
    # Download test labels
    test_labels_zip = data_dir / "GTSRB_Final_Test_GT.zip"
    if not test_labels_zip.exists():
        print("\n📥 Downloading test labels...")
        download_url(test_labels_url, test_labels_zip)
    else:
        print("\n✓ Test labels already downloaded")
    
    # Extract training data
    train_dir = data_dir / "Train"
    if not train_dir.exists():
        print("\n📦 Extracting training images...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Reorganize training data
        if (data_dir / "GTSRB" / "Final_Training").exists():
            final_train = data_dir / "GTSRB" / "Final_Training" / "Images"
            if final_train.exists():
                shutil.move(str(final_train), str(train_dir))
                shutil.rmtree(data_dir / "GTSRB")
    else:
        print("\n✓ Training images already extracted")
    
    # Extract test data
    test_dir = data_dir / "Test"
    if not test_dir.exists():
        print("\n📦 Extracting test images...")
        with zipfile.ZipFile(test_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Reorganize test data
        if (data_dir / "GTSRB" / "Final_Test").exists():
            final_test = data_dir / "GTSRB" / "Final_Test" / "Images"
            if final_test.exists():
                shutil.move(str(final_test), str(test_dir))
                shutil.rmtree(data_dir / "GTSRB")
    else:
        print("\n✓ Test images already extracted")
    
    # Extract test labels
    if not (test_dir / "GT-final_test.csv").exists():
        print("\n📦 Extracting test labels...")
        with zipfile.ZipFile(test_labels_zip, 'r') as zip_ref:
            zip_ref.extractall(test_dir)
    else:
        print("\n✓ Test labels already extracted")
    
    # Count files
    if train_dir.exists():
        train_images = sum([len(files) for r, d, files in os.walk(train_dir)])
        print(f"\n✓ Training images: {train_images}")
    
    if test_dir.exists():
        test_images = len(list(test_dir.glob("*.ppm")))
        print(f"✓ Test images: {test_images}")
    
    print("\n" + "=" * 80)
    print("✅ GTSRB dataset ready!")
    print(f"📁 Location: {data_dir.absolute()}")
    print("=" * 80)
    
    return data_dir


if __name__ == "__main__":
    download_gtsrb()
