import torch
import torch.nn as nn
from torchvision import models


class EmbeddingNet(nn.Module):
    """ResNet50 backbone with normalized embedding head."""

    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return nn.functional.normalize(x, p=2, dim=1)


class FaceClassifier(nn.Module):
    """Embedding backbone plus classification head for CE-style training."""

    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super().__init__()
        self.encoder = EmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits
