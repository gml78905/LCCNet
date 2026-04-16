import torch
import torch.nn as nn
import torchvision.models as tv_models

from .blocks import BasicBlock, make_layer


class CameraResnetEncoder(nn.Module):
    """
    Camera encoder using torchvision ResNet-18 feature pyramid.
    Returns a dict with multi-scale features.
    """

    def __init__(self, pretrained=False):
        super().__init__()
        # torchvision version compatibility:
        # - new API: resnet18(weights=...)
        # - old API: resnet18(pretrained=...)
        try:
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                self.encoder = tv_models.resnet18(weights=weights)
            except TypeError:
                self.encoder = tv_models.resnet18(pretrained=pretrained)
        except AttributeError:
            self.encoder = tv_models.resnet18(pretrained=pretrained)

    def forward(self, x):
        # Keep LCCNet-style normalization convention.
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        s2 = self.encoder.relu(x)                 # [B, 64, H/2,  W/2]
        x = self.encoder.maxpool(s2)
        s4 = self.encoder.layer1(x)              # [B, 64, H/4,  W/4]
        s8 = self.encoder.layer2(s4)             # [B, 128, H/8,  W/8]
        s16 = self.encoder.layer3(s8)            # [B, 256, H/16, W/16]
        s32 = self.encoder.layer4(s16)           # [B, 512, H/32, W/32]
        return {"s2": s2, "s4": s4, "s8": s8, "s16": s16, "s32": s32}


class RangeEncoder(nn.Module):
    """
    Generic LCCNet-style encoder for projected range-like inputs.
    Used for LiDAR and radar branches.
    """

    def __init__(self, in_channels, activation="leakyrelu"):
        super().__init__()
        if activation not in ["leakyrelu", "elu"]:
            raise ValueError("activation must be 'leakyrelu' or 'elu'")

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act_name = activation
        self.act_lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.act_elu = nn.ELU(inplace=True)

        layer1, c = make_layer(BasicBlock, 64, 64, blocks=2, stride=1)
        layer2, c = make_layer(BasicBlock, c, 128, blocks=2, stride=2)
        layer3, c = make_layer(BasicBlock, c, 256, blocks=2, stride=2)
        layer4, c = make_layer(BasicBlock, c, 512, blocks=2, stride=2)
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def _act(self, x):
        if self.act_name == "elu":
            return self.act_elu(x)
        return self.act_lrelu(x)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        s2 = self._act(x)                        # [B, 64, H/2,  W/2]
        x = self.maxpool(s2)
        s4 = self.layer1(x)                      # [B, 64, H/4,  W/4]
        s8 = self.layer2(s4)                     # [B, 128, H/8,  W/8]
        s16 = self.layer3(s8)                    # [B, 256, H/16, W/16]
        s32 = self.layer4(s16)                   # [B, 512, H/32, W/32]
        return {"s2": s2, "s4": s4, "s8": s8, "s16": s16, "s32": s32}


class LidarEncoder(RangeEncoder):
    def __init__(self, activation="leakyrelu"):
        super().__init__(in_channels=1, activation=activation)


class RadarEncoder(RangeEncoder):
    def __init__(self, activation="leakyrelu"):
        super().__init__(in_channels=2, activation=activation)
