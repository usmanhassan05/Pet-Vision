"""
PetVision AI - Evaluation Engine & Metrics Computation
======================================================
Computes:
- Top-1 and Top-3 Accuracy
- Macro and Weighted Precision, Recall, and F1-Scores
- Per-class accuracy breakdown
- Confusion Matrix and Most Confused Breed Pairs
- Inference latency per image (ms)
- Examples of correctly classified and misclassified instances
"""

import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
import torch.nn.functional as F


def evaluate_model_on_test_set(
    model: torch.nn.Module,
    test_loader,
    device: str,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Runs full evaluation on the unseen test set and computes all required metrics.
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_top3_hits = []
    all_probs = []
    sample_records = []
    
    inference_times = []
    
    with torch.no_grad():
        for images, labels, species_labels, breeds in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Measure inference time per batch
            t0 = time.perf_counter()
            outputs = model(images)
            t_batch = (time.perf_counter() - t0) * 1000.0  # ms
            t_per_img = t_batch / images.size(0)
            inference_times.append(t_per_img)
            
            probs = F.softmax(outputs, dim=1)
            
            # Top-1 prediction
            _, preds = torch.max(probs, dim=1)
            
            # Top-3 predictions
            top3_prob, top3_indices = torch.topk(probs, k=3, dim=1)
            
            for i in range(images.size(0)):
                true_idx = labels[i].item()
                pred_idx = preds[i].item()
                top3 = top3_indices[i].cpu().tolist()
                top3_p = top3_prob[i].cpu().tolist()
                
                is_top3 = true_idx in top3
                all_top3_hits.append(is_top3)
                all_preds.append(pred_idx)
                all_targets.append(true_idx)
                
                sample_records.append({
                    "true_class": class_names[true_idx],
                    "pred_class": class_names[pred_idx],
                    "is_correct": bool(true_idx == pred_idx),
                    "confidence": float(probs[i, pred_idx].item()),
                    "top3_classes": [class_names[idx] for idx in top3],
                    "top3_confs": [float(c) for c in top3_p],
                })
                
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    # 1. Primary Metrics
    acc = accuracy_score(y_true, y_pred)
    top3_acc = np.mean(all_top3_hits)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    weighted_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    avg_inference_ms = float(np.mean(inference_times))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    # 3. Per-class report
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    results = {
        "accuracy": float(acc),
        "top3_accuracy": float(top3_acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "avg_inference_ms": avg_inference_ms,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "samples": sample_records,
    }
    
    return results
