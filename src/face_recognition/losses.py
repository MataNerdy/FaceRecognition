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



def mine_triplets_from_batch(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Create simple in-batch triplets, returning None when the batch is unsuitable."""
    anchors = []
    positives = []
    negatives = []
    labels = labels.detach()
    for index, label in enumerate(labels):
        positive_indices = torch.nonzero((labels == label) & (torch.arange(labels.numel(), device=labels.device) != index), as_tuple=False).flatten()
        negative_indices = torch.nonzero(labels != label, as_tuple=False).flatten()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue
        anchors.append(embeddings[index])
        positives.append(embeddings[positive_indices[0]])
        negatives.append(embeddings[negative_indices[0]])
    if not anchors:
        return None
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


class ArcFaceTripletLoss(nn.Module):
    """Weighted ArcFace plus in-batch Triplet loss.

    If the batch does not contain at least one positive and one negative pair for
    triplet mining, the loss falls back to ArcFace only.
    """

    def __init__(
        self,
        arcface_loss: ArcFaceLoss,
        triplet_loss: TripletLoss | None = None,
        arcface_weight: float = 1.0,
        triplet_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.arcface_loss = arcface_loss
        self.triplet_loss = triplet_loss or TripletLoss()
        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return weighted ArcFace plus Triplet loss for one batch."""
        arc_loss = self.arcface_loss(embeddings, labels)
        triplets = mine_triplets_from_batch(embeddings, labels)
        if triplets is None or self.triplet_weight == 0:
            return self.arcface_weight * arc_loss
        anchor, positive, negative = triplets
        tri_loss = self.triplet_loss(anchor, positive, negative)
        return self.arcface_weight * arc_loss + self.triplet_weight * tri_loss
