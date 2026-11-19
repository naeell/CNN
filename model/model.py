import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Sesuai slide: input (3,100,100) -> (20,32,32) -> (50,9,9) -> FC."""
    def __init__(self, num_classes, input_c=3):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(input_c, 20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        self.fc = nn.Linear(50 * 9 * 9, num_classes)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return self.fc(torch.flatten(x, 1))


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ImprovedCNN(nn.Module):
    """Simple CNN yang diubah: 4 blok, BatchNorm, GAP, Dropout."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_pretrained(num_classes, freeze_backbone=False):
    """ShuffleNetV2 pretrained ImageNet (ringan, bukan MobileNet)."""
    m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def get_model(name, num_classes):
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    if name == "improved_cnn":
        return ImprovedCNN(num_classes)
    if name == "shufflenet":
        return build_pretrained(num_classes)
    raise ValueError(f"Model tidak dikenal: {name}")