"""
PetVision AI - Data Augmentation Impact & Ablation Visualizer
============================================================
Visualizes original vs augmented pet images to demonstrate invariance
against scale, rotation, mirroring, and lighting variations.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
import numpy as np


def generate_augmentation_figure(sample_image_path: str, save_path: str = "results/augmentation_impact.png"):
    if not os.path.exists(sample_image_path):
        print(f"Sample path {sample_image_path} does not exist.")
        return
        
    img = Image.open(sample_image_path).convert("RGB").resize((224, 224))
    
    # 1. Original
    # 2. Horizontal Flip
    flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 3. Rotation
    rot = img.rotate(15, expand=False)
    # 4. Color Jitter (Brightness & Contrast)
    enhancer_b = ImageEnhance.Brightness(img)
    bright = enhancer_b.enhance(1.3)
    enhancer_c = ImageEnhance.Contrast(bright)
    jitter = enhancer_c.enhance(1.2)
    # 5. Crop & Zoom
    w, h = img.size
    crop_box = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
    zoom = img.crop(crop_box).resize((224, 224), Image.BILINEAR)
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    titles = [
        "1. Original (224x224)",
        "2. Random Horizontal Flip",
        "3. Random Rotation (15°)",
        "4. Color / Lighting Jitter",
        "5. Random Resized Crop"
    ]
    images = [img, flip, rot, jitter, zoom]
    
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=10, fontweight="bold")
        ax.axis("off")
        
    plt.suptitle("PetVision AI: Data Augmentation Pipeline Invariance", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved augmentation impact figure to: {save_path}")


if __name__ == "__main__":
    sample_path = os.path.join("dataset", "images", "images", "Persian_1.jpg")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("dataset", "images", "images", "beagle_1.jpg")
    generate_augmentation_figure(sample_path)
