"""
PetVision AI - Helper Utilities & Visualization Functions
=========================================================
Includes:
- Deterministic seed setting for reproducibility.
- Hardware device detection.
- Training loss & accuracy curves visualization.
- Seaborn confusion matrix plotting and most-confused breed pair analysis.
- Model parameter & disk size measurement.
"""

import os
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and PyTorch for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Returns 'cuda' if GPU is available, otherwise 'cpu'."""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plots training and validation loss & accuracy curves side-by-side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#d62728", linewidth=2)
    plt.title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy (%)", color="#2ca02c", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Accuracy (%)", color="#ff7f0e", linewidth=2)
    plt.title("Classification Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """Plots high-resolution Seaborn confusion matrix for all 37 breeds."""
    plt.figure(figsize=(18, 15))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True
        )
    else:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.yticks(range(len(class_names)), class_names)
        
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted Breed", fontsize=13, labelpad=10)
    plt.ylabel("True Breed", fontsize=13, labelpad=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_most_confused_pairs(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 10
) -> List[Tuple[str, str, int]]:
    """Identifies the top-k most confused breed pairs from the confusion matrix."""
    n = len(class_names)
    confused = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confused.append((class_names[i], class_names[j], int(cm[i, j])))
    confused.sort(key=lambda x: x[2], reverse=True)
    return confused[:top_k]


def count_parameters(model) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
