"""Loss functions used by the face recognition experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """Additive angular margin loss over normalized embeddings."""

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return ArcFace cross-entropy for a batch of embeddings and labels."""
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weights).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=weights.size(0)).to(dtype=cosine.dtype, device=cosine.device)
        logits = torch.where(one_hot.bool(), target_logits, cosine)
        return F.cross_entropy(logits * self.scale, labels)


class TripletLoss(nn.Module):
    """Triplet margin loss for anchor-positive-negative embeddings."""

    def __init__(self, margin: float = 0.3, p: float = 2.0) -> None:
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Return mean triplet loss for a batch of triplets."""
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class ArcFaceCELoss(nn.Module):
    """Weighted combination of standard CE logits and ArcFace embedding loss."""

    def __init__(self, arcface_loss: ArcFaceLoss, ce_weight: float = 1.0, arcface_weight: float = 1.0) -> None:
        super().__init__()
        self.arcface_loss = arcface_loss
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.arcface_weight = arcface_weight

    def forward(self, logits: torch.Tensor, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return the weighted hybrid loss."""
        ce_loss = self.ce(logits, labels)
        arc_loss = self.arcface_loss(embeddings, labels)
        return self.ce_weight * ce_loss + self.arcface_weight * arc_loss

