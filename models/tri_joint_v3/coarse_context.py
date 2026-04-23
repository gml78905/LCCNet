from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCoarseContextV3(nn.Module):
    """
    Build a coarse tri-modal shared context from s32 feature maps.

    Inputs:
      cam_s32, lid_s32, rad_s32: [B, 512, H/32, W/32]
      target_hw_s16: (H/16, W/16)

    Outputs:
      ctx_s16:  [B, 256, H/16, W/16]
      z_shared: [B, 256]
    """

    def __init__(self, s32_channels_each: int = 512, d_shared: int = 256):
        super().__init__()
        in_ch = s32_channels_each * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, d_shared, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_shared, d_shared, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        cam_s32: torch.Tensor,
        lid_s32: torch.Tensor,
        rad_s32: torch.Tensor,
        target_hw_s16: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([cam_s32, lid_s32, rad_s32], dim=1)
        ctx_s32 = self.fuse(x)                            # [B,256,H/32,W/32]
        z_shared = self.pool(ctx_s32).flatten(1)          # [B,256]
        ctx_s16 = F.interpolate(
            ctx_s32, size=target_hw_s16, mode="bilinear", align_corners=False
        )                                                 # [B,256,H/16,W/16]
        return {
            "ctx_s16": ctx_s16,
            "z_shared": z_shared,
        }
