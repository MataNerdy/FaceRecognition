"""Stacked Hourglass Network for five-point facial landmark heatmaps."""

from __future__ import annotations

import torch
import torch.nn as nn


class Residual(nn.Module):
    """Bottleneck residual block used inside the hourglass modules."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        return self.main(x) + self.skip(x)


class Hourglass(nn.Module):
    """Recursive hourglass block with top-down and bottom-up paths."""

    def __init__(self, depth: int, channels: int, num_blocks: int = 1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Hourglass depth must be at least 1")
        self.up1 = self._make_residual(channels, channels, num_blocks)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = self._make_residual(channels, channels, num_blocks)
        self.low2 = Hourglass(depth - 1, channels, num_blocks) if depth > 1 else self._make_residual(channels, channels, num_blocks)
        self.low3 = self._make_residual(channels, channels, num_blocks)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    @staticmethod
    def _make_residual(in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = [Residual(in_channels, out_channels)]
        layers.extend(Residual(out_channels, out_channels) for _ in range(num_blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one recursive hourglass module."""
        upper = self.up1(x)
        lower = self.pool(x)
        lower = self.low1(lower)
        lower = self.low2(lower)
        lower = self.low3(lower)
        return upper + self.upsample(lower)


class StackedHourglassNet(nn.Module):
    """Stacked Hourglass Network that predicts facial landmark heatmaps."""

    def __init__(self, num_stacks: int = 2, num_blocks: int = 4, channels: int = 128, num_keypoints: int = 5) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, channels),
        )
        self.hourglasses = nn.ModuleList([Hourglass(4, channels, num_blocks) for _ in range(num_stacks)])
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    Residual(channels, channels),
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_stacks)
            ]
        )
        self.outs = nn.ModuleList([nn.Conv2d(channels, num_keypoints, kernel_size=1) for _ in range(num_stacks)])
        self.merge_features = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_stacks - 1)])
        self.merge_preds = nn.ModuleList([nn.Conv2d(num_keypoints, channels, kernel_size=1) for _ in range(num_stacks - 1)])

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of heatmap predictions, one tensor per stack."""
        inter = self.pre(images)
        outputs: list[torch.Tensor] = []
        for index, hourglass in enumerate(self.hourglasses):
            features = hourglass(inter)
            features = self.features[index](features)
            prediction = self.outs[index](features)
            outputs.append(prediction)
            if index < len(self.hourglasses) - 1:
                inter = inter + self.merge_features[index](features) + self.merge_preds[index](prediction)
        return outputs

