from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tri_joint.encoders import CameraEncoderMS


class _SparseResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class SparseAwareRangeEncoderMS(nn.Module):
    """
    Sparse-aware encoder for LiDAR/Radar projections.

    Input:
      x: [B,C,H,W]

    Output:
      s16: [B,256,H/16,W/16]
      s32: [B,512,H/32,W/32]
      support: [B,1,H/16,W/16]
      valid: [B,1,H/16,W/16]
      density: [B,1,H/16,W/16]
    """

    def __init__(self, in_channels: int, support_prior_scale: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.support_prior_scale = support_prior_scale
        # value + valid + density + support prior
        stem_in = in_channels + 3
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _SparseResidualBlock(64, 64, stride=1)    # /4
        self.layer2 = _SparseResidualBlock(64, 128, stride=2)   # /8
        self.layer3 = _SparseResidualBlock(128, 256, stride=2)  # /16
        self.layer4 = _SparseResidualBlock(256, 512, stride=2)  # /32

    def _build_sparse_priors(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        primary = x[:, :1]
        valid = (primary > 0).float()
        if x.shape[1] > 1:
            valid = torch.maximum(valid, (x[:, 1:2].abs() > 0).float())
        density = F.avg_pool2d(valid, kernel_size=5, stride=1, padding=2)
        support = self.support_prior_scale * (0.5 * valid + 0.5 * density)
        return {"valid": valid, "density": density, "support": support}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        priors = self._build_sparse_priors(x)
        stem_in = torch.cat([x, priors["valid"], priors["density"], priors["support"]], dim=1)
        feat = self.stem(stem_in)
        feat = self.pool(feat)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        s16 = self.layer3(feat)
        s32 = self.layer4(s16)
        target_hw = (s16.shape[-2], s16.shape[-1])
        valid_s16 = F.interpolate(priors["valid"], size=target_hw, mode="nearest")
        density_s16 = F.interpolate(priors["density"], size=target_hw, mode="bilinear", align_corners=False)
        support_s16 = F.interpolate(priors["support"], size=target_hw, mode="bilinear", align_corners=False)
        return {
            "s16": s16,
            "s32": s32,
            "valid": valid_s16,
            "density": density_s16,
            "support": support_s16,
        }


class SparseAwareLidarEncoderMS(SparseAwareRangeEncoderMS):
    def __init__(self):
        super().__init__(in_channels=1, support_prior_scale=1.0)


class RadarUncertaintyEncoderMS(SparseAwareRangeEncoderMS):
    def __init__(self):
        super().__init__(in_channels=2, support_prior_scale=1.25)
