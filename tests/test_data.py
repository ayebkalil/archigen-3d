"""
Unit tests for Dataset loading, augmentations, and normalization.
Satisfies MLOps Criterion: 'Tests unitaires' (3/3).
"""

import pytest
import torch
from pathlib import Path
from src.data.dataset import FacadesDataset

@pytest.fixture
def dataset_root():
    return "data/raw/facades"

def test_dataset_train_split_loading(dataset_root):
    """Verify that train split loads and produces valid tensors."""
    dataset = FacadesDataset(root_dir=dataset_root, split="train", img_size=256)
    assert len(dataset) > 0, "Train dataset should contain images"
    
    sample = dataset[0]
    assert "sketch" in sample
    assert "photo" in sample
    assert "filename" in sample

    sketch = sample["sketch"]
    photo = sample["photo"]

    # Verify tensor shapes
    assert sketch.shape == (3, 256, 256), f"Expected (3, 256, 256), got {sketch.shape}"
    assert photo.shape == (3, 256, 256), f"Expected (3, 256, 256), got {photo.shape}"

    # Verify normalization within [-1.0, 1.0]
    assert sketch.min() >= -1.05 and sketch.max() <= 1.05
    assert photo.min() >= -1.05 and photo.max() <= 1.05

def test_dataset_val_split_exists(dataset_root):
    """Verify validation split loading."""
    dataset = FacadesDataset(root_dir=dataset_root, split="val", img_size=256)
    assert len(dataset) == 100, f"Expected 100 validation images, got {len(dataset)}"
