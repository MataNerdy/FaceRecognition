"""Embedding and classification models for aligned face images."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """ResNet50 backbone with an L2-normalized embedding head."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized image embeddings."""
        features = self.backbone(images)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)


class FaceClassifier(nn.Module):
    """Embedding network with a linear identity classifier."""

    def __init__(self, num_classes: int, embedding_dim: int = 512, pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = EmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained, dropout=dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor, return_embedding: bool = False):
        """Return logits, or `(logits, embeddings)` when requested."""
        embeddings = self.encoder(images)
        logits = self.classifier(embeddings)
        return (logits, embeddings) if return_embedding else logits

