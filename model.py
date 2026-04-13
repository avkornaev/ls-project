from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    """ResNet18 adapted for CIFAR-10 with optional feature extraction."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        backbone = resnet18(weights=None, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


def build_model(num_classes: int = 10) -> CIFARResNet18:
    return CIFARResNet18(num_classes=num_classes)
