import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from models.tri_calib.blocks import BasicBlock, make_layer


class CameraEncoderMS(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        try:
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                self.encoder = tv_models.resnet18(weights=weights)
            except TypeError:
                self.encoder = tv_models.resnet18(pretrained=pretrained)
        except AttributeError:
            self.encoder = tv_models.resnet18(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        return x


class _ProjectEncoder(nn.Module):
    def __init__(self, attr_in_channels: int):
        super().__init__()
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.attr_branch = nn.Sequential(
            nn.Conv2d(attr_in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1, c = make_layer(BasicBlock, 64, 64, blocks=2, stride=1)
        layer2, c = make_layer(BasicBlock, c, 128, blocks=2, stride=2)
        layer3, c = make_layer(BasicBlock, c, 256, blocks=2, stride=2)
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    def forward(self, depth: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        depth_feat = self.depth_branch(depth)
        attr_feat = self.attr_branch(attrs)
        x = torch.cat([depth_feat, attr_feat], dim=1)
        x = self.stem(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def _mask_from_depth(depth: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    valid = (depth > 0).float()
    H, W = depth.shape[-2:]
    h, w = target_hw

    kh = H // h
    kw = W // w

    if H % h == 0 and W % w == 0:
        mask = F.max_pool2d(valid, kernel_size=(kh, kw), stride=(kh, kw))
    else:
        density = F.interpolate(valid, size=target_hw, mode="area")
        mask = (density > 0).float()

    return mask


class LidarEncoderV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.intensity_scale = 32.0
        self.time_scale = 1.0
        self.encoder = _ProjectEncoder(attr_in_channels=2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        depth = x[:, :1]
        intensity = torch.tanh(x[:, 1:2] / self.intensity_scale)
        dt_lidar = torch.clamp(x[:, 2:3] / self.time_scale, min=-1.0, max=1.0)
        feat = self.encoder(depth, torch.cat([intensity, dt_lidar], dim=1))
        mask = _mask_from_depth(depth, feat.shape[-2:])
        return {"feat": feat, "mask": mask}


class RadarEncoderV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_scale = 15.0
        self.rcs_scale = 40.0
        self.time_scale = 1.0
        self.encoder = _ProjectEncoder(attr_in_channels=3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        depth = x[:, :1]
        velocity = torch.clamp(x[:, 1:2] / self.velocity_scale, min=-1.0, max=1.0)
        rcs = torch.tanh(x[:, 2:3] / self.rcs_scale)
        dt_radar = torch.clamp(x[:, 3:4] / self.time_scale, min=-1.0, max=1.0)
        feat = self.encoder(depth, torch.cat([velocity, rcs, dt_radar], dim=1))
        mask = _mask_from_depth(depth, feat.shape[-2:])
        return {"feat": feat, "mask": mask}
