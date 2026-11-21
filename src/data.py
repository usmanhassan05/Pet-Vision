"""
PetVision AI - Data Ingestion, Stratified Splitting, and PyTorch Datasets
========================================================================
This module handles:
1. Scanning Oxford-IIIT Pet dataset from disk without modifying raw files.
2. Robust label & species extraction from filenames using regex.
3. Stratified Train / Validation / Test splitting (70% / 15% / 15%) with fixed seed.
4. PyTorch Dataset implementation with RGB conversion and data augmentations.
"""

import os
import re
from typing import Tuple, Dict, List, Optional
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError:
    # Allow running purely for metadata/EDA without PyTorch locally
    torch = None
    Dataset = object
    DataLoader = None
    transforms = None

from sklearn.model_selection import train_test_split


def parse_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parses Oxford-IIIT Pet filename to extract breed name and species.
    Naming convention:
      - '<breed_name>_<number>.jpg'
      - First letter uppercase -> Cat (e.g., 'Abyssinian_1.jpg')
      - First letter lowercase -> Dog (e.g., 'american_bulldog_1.jpg')
    """
    match = re.match(r"^([A-Za-z_]+)_(\d+)\.jpg$", filename, re.IGNORECASE)
    if not match:
        return None
    breed = match.group(1)
    species = "Cat" if breed[0].isupper() else "Dog"
    return breed, species


def scan_dataset(images_dir: str) -> pd.DataFrame:
    """
    Scans the images directory and returns a clean DataFrame.
    Filters out non-image files (e.g., MATLAB .mat files).
    """
    records = []
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        breed, species = parsed
        full_path = os.path.join(images_dir, fname)
        records.append({
            "filename": fname,
            "filepath": os.path.abspath(full_path),
            "breed": breed,
            "species": species,
        })
    df = pd.DataFrame(records)
    df.sort_values(by=["breed", "filename"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_class_mappings(breeds: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, str]]:
    """
    Builds deterministic breed-to-index and index-to-breed mappings.
    """
    unique_breeds = sorted(list(set(breeds)))
    breed_to_idx = {b: i for i, b in enumerate(unique_breeds)}
    idx_to_breed = {i: b for b, i in breed_to_idx.items()}
    breed_to_species = {b: ("Cat" if b[0].isupper() else "Dog") for b in unique_breeds}
    return breed_to_idx, idx_to_breed, breed_to_species


def create_stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> pd.DataFrame:
    """
    Creates a stratified split (Train/Val/Test) preserving exact class proportions.
    
    Justification vs 10-fold CV:
    ----------------------------
    1. Computational feasibility: Training deep models (ViT, MobileNet, CNN) across 10 folds
       would require 30 heavy training runs, which is impractical and computationally redundant.
    2. Statistical power: With 7,390 total images (~200 per class), a 15% test split provides
       over 1,100 unseen images (~30 images per breed), which is statistically robust for
       evaluating generalization, macro/weighted F1, and top-3 accuracy.
    3. Stratification guarantees zero class skew across train, validation, and test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["breed"]
    )
    
    # Second split: Val vs Test
    val_share_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=temp_df["breed"]
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return full_df


class PetDataset(Dataset):
    """
    PyTorch Dataset for Oxford-IIIT Pet.
    Loads images on-the-fly, forces 3-channel RGB, and applies torchvision transforms.
    """
    def __init__(self, df: pd.DataFrame, breed_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.breed_to_idx = breed_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        breed = row["breed"]
        species = row["species"]
        
        # Load and ensure RGB format (handles grayscale and RGBA safely)
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)
            
        label = self.breed_to_idx[breed]
        species_label = 0 if species == "Cat" else 1
        
        return image, label, species_label, breed


def get_transforms(image_size: int = 224):
    """
    Returns training and validation/test transforms using ImageNet normalization.
    """
    if transforms is None:
        raise ImportError("torchvision is required for transforms.")
        
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    return train_transform, eval_transform
