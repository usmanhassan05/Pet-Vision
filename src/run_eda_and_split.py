"""
PetVision AI - EDA and Stratified Dataset Splitting Script
==========================================================
Executes complete Exploratory Data Analysis, generates visualization figures,
computes stratified train/val/test splits, and saves metadata into results/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from data import scan_dataset, build_class_mappings, create_stratified_split


def generate_eda_and_splits(images_dir: str, results_dir: str, seed: int = 42):
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Scanning dataset from: {images_dir}...")
    df = scan_dataset(images_dir)
    print(f"Total valid images loaded: {len(df)}")
    
    # 1. Class Mappings
    breed_to_idx, idx_to_breed, breed_to_species = build_class_mappings(df["breed"].tolist())
    df["class_id"] = df["breed"].map(breed_to_idx)
    
    mappings = {
        "num_classes": len(breed_to_idx),
        "breed_to_idx": breed_to_idx,
        "idx_to_breed": idx_to_breed,
        "breed_to_species": breed_to_species,
    }
    with open(os.path.join(results_dir, "class_mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)
    print("Saved class mappings to results/class_mappings.json")
    
    # 2. Stratified Split (70% train, 15% val, 15% test)
    print("\nCreating stratified split (70% train, 15% val, 15% test, seed=42)...")
    split_df = create_stratified_split(df, seed=seed, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    split_csv_path = os.path.join(results_dir, "dataset_split.csv")
    split_df.to_csv(split_csv_path, index=False)
    print(f"Saved stratified split dataset to: {split_csv_path}")
    
    # Print Split Summary
    split_counts = split_df["split"].value_counts()
    print("\nSplit Distribution:")
    for split_name, count in split_counts.items():
        pct = (count / len(split_df)) * 100
        print(f"  - {split_name.capitalize():<6}: {count} images ({pct:.2f}%)")
        
    # 3. EDA Plot 1: Class Distribution Bar Chart
    plt.figure(figsize=(16, 7))
    class_counts = df["breed"].value_counts().sort_index()
    colors = ["#e74c3c" if breed_to_species[b] == "Cat" else "#3498db" for b in class_counts.index]
    
    bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors, edgecolor="black", linewidth=0.6)
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=90, fontsize=9)
    plt.ylabel("Number of Images", fontsize=12, fontweight="bold")
    plt.title("Oxford-IIIT Pet Dataset: Class Distribution (37 Breeds)\nRed = Cat Breeds, Blue = Dog Breeds", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "eda_class_distribution.png"), dpi=300)
    plt.close()
    print("Saved results/eda_class_distribution.png")
    
    # 4. EDA Plot 2: Species Donut Chart
    plt.figure(figsize=(7, 7))
    species_counts = df["species"].value_counts()
    colors = ["#3498db", "#e74c3c"]
    explode = (0.04, 0.04)
    plt.pie(
        species_counts.values,
        labels=[f"{s} ({c})" for s, c in zip(species_counts.index, species_counts.values)],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops={"fontsize": 12, "fontweight": "bold"}
    )
    plt.title("Species Ratio (Cat vs Dog)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "eda_species_distribution.png"), dpi=300)
    plt.close()
    print("Saved results/eda_species_distribution.png")
    
    # 5. EDA Plot 3: Image Dimensions Analysis
    print("Collecting image dimensions...")
    widths, heights = [], []
    for path in df["filepath"].sample(min(1000, len(df)), random_state=seed):
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.4, color="#8e44ad", edgecolors="none", s=25)
    plt.axvline(224, color="red", linestyle="--", label="Model Input Res: 224x224")
    plt.axhline(224, color="red", linestyle="--")
    plt.xlabel("Width (pixels)", fontsize=12)
    plt.ylabel("Height (pixels)", fontsize=12)
    plt.title("Sample Image Dimensions Distribution (Width vs. Height)", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "eda_resolutions.png"), dpi=300)
    plt.close()
    print("Saved results/eda_resolutions.png")
    
    # 6. EDA Plot 4: Sample Images Grid (Cat vs Dog)
    print("Generating sample images grid...")
    sample_breeds = list(breed_to_idx.keys())[:16] # 16 sample breeds
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()
    
    for ax, breed in zip(axes, sample_breeds):
        breed_rows = df[df["breed"] == breed]
        sample_row = breed_rows.sample(1, random_state=seed).iloc[0]
        with Image.open(sample_row["filepath"]) as img:
            ax.imshow(img.convert("RGB"))
            species_badge = "🐱 Cat" if sample_row["species"] == "Cat" else "🐶 Dog"
            ax.set_title(f"{breed}\n[{species_badge}]", fontsize=10, fontweight="bold")
            ax.axis("off")
            
    plt.suptitle("PetVision AI: Oxford-IIIT Pet Representative Breed Samples", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "eda_sample_grid.png"), dpi=300)
    plt.close()
    print("Saved results/eda_sample_grid.png")
    print("\nEDA & Splitting successfully completed!")


if __name__ == "__main__":
    IMAGES_DIR = os.path.join("dataset", "images", "images")
    RESULTS_DIR = "results"
    generate_eda_and_splits(IMAGES_DIR, RESULTS_DIR, seed=42)
