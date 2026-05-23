import torch
import torch.nn as nn


class Residual(nn.Module):
    """Residual block used inside Stacked Hourglass."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 2
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.main(x) + self.skip(x)


class Hourglass(nn.Module):
    def __init__(self, depth: int, channels: int, num_blocks: int = 1):
        super().__init__()
        self.depth = depth
        self.up1 = self._make_residual(channels, channels, num_blocks)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = self._make_residual(channels, channels, num_blocks)
        self.low2 = Hourglass(depth - 1, channels, num_blocks) if depth > 1 else self._make_residual(channels, channels, num_blocks)
        self.low3 = self._make_residual(channels, channels, num_blocks)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    @staticmethod
    def _make_residual(in_channels, out_channels, num_blocks):
        layers = [Residual(in_channels, out_channels)]
        layers += [Residual(out_channels, out_channels) for _ in range(num_blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        up1 = self.up1(x)
        low = self.pool(x)
        low = self.low1(low)
        low = self.low2(low)
        low = self.low3(low)
        return up1 + self.upsample(low)


class StackedHourglassNet(nn.Module):
    """Stacked Hourglass Network for five facial landmarks heatmaps."""

    def __init__(self, num_stacks=2, num_blocks=4, channels=128, num_keypoints=5):
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
        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(channels, channels),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_stacks)
        ])
        self.outs = nn.ModuleList([nn.Conv2d(channels, num_keypoints, kernel_size=1) for _ in range(num_stacks)])
        self.merge_features = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_stacks - 1)])
        self.merge_preds = nn.ModuleList([nn.Conv2d(num_keypoints, channels, kernel_size=1) for _ in range(num_stacks - 1)])

    def forward(self, x):
        x = self.pre(x)
        outputs = []
        inter = x
        for i, hg in enumerate(self.hourglasses):
            y = hg(inter)
            y = self.features[i](y)
            pred = self.outs[i](y)
            outputs.append(pred)
            if i < len(self.hourglasses) - 1:
                inter = inter + self.merge_features[i](y) + self.merge_preds[i](pred)
        return outputs
