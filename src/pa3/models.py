"""Modelos: CNN pequena do zero e ResNet-18 (ImageNet)."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm

MAX_PARAMS = 12_000_000


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """5 blocos conv duplos + GAP + linear (~1,2M parâmetros)."""

    def __init__(self, n_classes: int, widths=(32, 64, 128, 192, 256), dropout: float = 0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers.append(_block(cin, w))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.fc(self.pool(self.features(x)))


def build_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "smallcnn":
        return SmallCNN(n_classes)
    if name == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    raise ValueError(f"modelo desconhecido: {name}")


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parâmetros da cabeça (recebem LR cheio); o resto é backbone."""
    return list(model.fc.parameters())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
