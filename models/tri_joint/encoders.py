from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from models.tri_calib.blocks import BasicBlock, make_layer


class CameraEncoderMS(nn.Module):
    """
    Camera multi-scale encoder.

    Input:
      x: [B, 3, H, W]
    Output dict:
      s16: [B, 256, H/16, W/16]
      s32: [B, 512, H/32, W/32]
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # TODO(v2/v3): consider stronger backbone variants (e.g., ConvNeXt) if needed.
        try:
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                self.encoder = tv_models.resnet18(weights=weights)
            except TypeError:
                self.encoder = tv_models.resnet18(pretrained=pretrained)
        except AttributeError:
            self.encoder = tv_models.resnet18(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Keep legacy normalization style for compatibility with current pipeline.
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        s16 = self.encoder.layer3(x)
        s32 = self.encoder.layer4(s16)
        return {"s16": s16, "s32": s32}


class RangeEncoderMS(nn.Module):
    """
    Generic range-like encoder for LiDAR/Radar projections.

    Input:
      x: [B, C, H, W]
    Output dict:
      s16: [B, 256, H/16, W/16]
      s32: [B, 512, H/32, W/32]
    """

    def __init__(self, in_channels: int, activation: str = "leakyrelu"):
        super().__init__()
        # TODO(v3): swap front-end when equirectangular lidar/radar representation is enabled.
        if activation not in ["leakyrelu", "elu"]:
            raise ValueError("activation must be 'leakyrelu' or 'elu'")

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.act_name = activation
        self.act_lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.act_elu = nn.ELU(inplace=True)

        layer1, c = make_layer(BasicBlock, 64, 64, blocks=2, stride=1)
        layer2, c = make_layer(BasicBlock, c, 128, blocks=2, stride=2)
        layer3, c = make_layer(BasicBlock, c, 256, blocks=2, stride=2)  # s16
        layer4, c = make_layer(BasicBlock, c, 512, blocks=2, stride=2)  # s32
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "elu":
            return self.act_elu(x)
        return self.act_lrelu(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.bn1(self.conv1(x))
        x = self._act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        s16 = self.layer3(x)
        s32 = self.layer4(s16)
        return {"s16": s16, "s32": s32}


class LidarEncoderMS(RangeEncoderMS):
    def __init__(self, activation: str = "leakyrelu"):
        super().__init__(in_channels=1, activation=activation)


class RadarEncoderMS(RangeEncoderMS):
    def __init__(self, activation: str = "leakyrelu"):
        nn.Module.__init__(self)
        if activation not in ["leakyrelu", "elu"]:
            raise ValueError("activation must be 'leakyrelu' or 'elu'")

        self.act_name = activation
        self.act_lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.act_elu = nn.ELU(inplace=True)

        # Radar is sparse and validity-dominated, so keep value / support paths separate
        # before fusing them into a shared stem.
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
        )
        self.aux_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
        )
        self.valid_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
        )
        self.valid_gate = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(32 * 3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer1, c = make_layer(BasicBlock, 64, 64, blocks=2, stride=1)
        layer2, c = make_layer(BasicBlock, c, 128, blocks=2, stride=2)
        layer3, c = make_layer(BasicBlock, c, 256, blocks=2, stride=2)  # s16
        layer4, c = make_layer(BasicBlock, c, 512, blocks=2, stride=2)  # s32
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "elu":
            return self.act_elu(x)
        return self.act_lrelu(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        depth = x[:, :1]
        aux = x[:, 1:2]
        valid = (depth > 0).float()
        local_density = F.avg_pool2d(valid, kernel_size=5, stride=1, padding=2)

        depth_feat = self.depth_stem(depth)
        aux_feat = self.aux_stem(aux)
        valid_feat = self.valid_stem(local_density)
        gate = self.valid_gate(local_density)

        sparse_feat = torch.cat([
            depth_feat * gate,
            aux_feat * gate,
            valid_feat,
        ], dim=1)
        x = self._act(self.fuse1(sparse_feat))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        s16 = self.layer3(x)
        s32 = self.layer4(s16)
        return {"s16": s16, "s32": s32}
