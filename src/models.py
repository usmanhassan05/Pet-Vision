"""
PetVision AI - Model Architectures
==================================
This module defines the three deep learning architectures for comparison:
1. CustomPetCNN: Custom Convolutional Neural Network built from scratch with
   Batch Normalization, Dropout, and Adaptive Average Pooling.
2. MobileNetV2PetClassifier: Pre-trained MobileNetV2 with two-stage transfer learning
   (Stage 1: Frozen base + new classifier head; Stage 2: Fine-tuning top convolutional blocks).
3. ViTPetClassifier: Vision Transformer (ViT-B/16) leveraging multi-head self-attention
   fine-tuned for 37-class pet breed classification.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models


class CustomPetCNN(nn.Module):
    """
    Custom 4-stage Convolutional Neural Network built from scratch.
    Features:
    - Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d in every stage.
    - Progressive channel scaling: 32 -> 64 -> 128 -> 256.
    - Dropout (0.25 in conv blocks, 0.5 in classifier head) to prevent overfitting.
    - AdaptiveAvgPool2d to allow flexible input spatial dimensions.
    """
    def __init__(self, num_classes: int = 37):
        super().__init__()
        
        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Stage 1: (3 -> 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
            
            # Stage 2: (32 -> 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
            
            # Stage 3: (64 -> 128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
            nn.Dropout2d(p=0.25),
            
            # Stage 4: (128 -> 256)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Dropout2d(p=0.25),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.global_pool(feat)
        out = self.classifier(pooled)
        return out


class MobileNetV2PetClassifier(nn.Module):
    """
    MobileNetV2 Transfer Learning Architecture.
    Stage 1: Frozen backbone + newly initialized classification head.
    Stage 2: Unfreeze top inverted residual blocks for gradual fine-tuning.
    """
    def __init__(self, num_classes: int = 37, pretrained: bool = True):
        super().__init__()
        weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.mobilenet_v2(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features  # 1280
        
        # Custom head with regularized dense layers
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def freeze_base(self):
        """Freezes all feature extractor layers for Stage 1 training."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
    def unfreeze_top_layers(self, num_blocks: int = 4):
        """Unfreezes top inverted residual blocks for Stage 2 fine-tuning."""
        total_layers = len(self.backbone.features)
        for i in range(total_layers - num_blocks, total_layers):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTPetClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) Architecture.
    Splits 224x224 image into 16x16 non-overlapping patches (196 tokens),
    passes through 12 Multi-Head Self-Attention layers, and classifies via [CLS] token.
    """
    def __init__(self, num_classes: int = 37, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = tv_models.vit_b_16(weights=weights)
        
        in_features = self.vit.heads.head.in_features  # 768
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def freeze_base(self):
        """Freezes patch projection and early encoder transformer layers."""
        for param in self.vit.conv_proj.parameters():
            param.requires_grad = False
        # Freeze first 8 transformer blocks, fine-tune last 4 blocks
        for block in self.vit.encoder.layers[:8]:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


def build_model(model_name: str, num_classes: int = 37, pretrained: bool = True) -> nn.Module:
    """Factory function to instantiate models by name."""
    name_clean = model_name.lower().replace("-", "_").replace(" ", "_")
    if "custom" in name_clean or "cnn" in name_clean:
        return CustomPetCNN(num_classes=num_classes)
    elif "mobilenet" in name_clean:
        return MobileNetV2PetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif "vit" in name_clean or "transformer" in name_clean:
        return ViTPetClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model architecture name: {model_name}")
