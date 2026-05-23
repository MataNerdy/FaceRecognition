import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """Additive angular margin loss."""

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)
        cosine = F.linear(embeddings, weights).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=weights.size(0)).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        return F.cross_entropy(logits * self.scale, labels)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class ArcFaceCELoss(nn.Module):
    def __init__(self, arcface_loss, ce_weight=1.0, arcface_weight=1.0):
        super().__init__()
        self.arcface_loss = arcface_loss
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.arcface_weight = arcface_weight

    def forward(self, logits, embeddings, labels):
        return self.ce_weight * self.ce(logits, labels) + self.arcface_weight * self.arcface_loss(embeddings, labels)
