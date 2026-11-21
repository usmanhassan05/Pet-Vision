"""
PetVision AI - Model Training Engine and Callbacks
==================================================
Includes:
- EarlyStopping callback to halt training when validation metric stops improving.
- Checkpoint saving for model persistence.
- Learning rate scheduling with ReduceLROnPlateau.
- Two-stage training support for transfer learning (frozen feature extractor -> fine-tuning).
"""

import os
import time
import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class EarlyStopping:
    """Monitors validation loss and stops training early if no improvement after `patience` epochs."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, checkpoint_path: Optional[str] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
            if self.checkpoint_path:
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Trains model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Evaluates model performance on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_pipeline(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau] = None,
    num_epochs: int = 15,
    patience: int = 5,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """Complete training loop with EarlyStopping, ReduceLROnPlateau, and checkpointing."""
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": []
    }
    
    print(f"\n--- Starting Training on {device.upper()} (Max Epochs: {num_epochs}, Patience: {patience}) ---")
    start_total = time.time()
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        epoch_duration = time.time() - t0
        
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(val_loss)
            
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_duration)
        
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s"
        )
        
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}!")
            break
            
    total_time = time.time() - start_total
    print(f"Training finished in {total_time:.1f}s. Best Val Loss: {early_stopping.best_loss:.4f}")
    
    # Restore best weights
    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
        
    return history
