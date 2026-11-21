"""
PetVision AI - Explainability Engine (Grad-CAM & Attention Visualization)
========================================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
salient spatial regions that influence model predictions.
"""

import os
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a specified convolutional target layer.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)    # [H, W]
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Resizes CAM to match original image dimensions and creates a blended heatmap."""
    w, h = img_pil.size
    cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]  # RGB heatmap
    img_np = np.array(img_pil.convert("RGB")) / 255.0
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay
