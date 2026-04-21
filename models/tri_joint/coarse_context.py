from typing import Dict, Tuple

import torch
import torch.nn as nn

class SharedCoarseContextV2(nn.Module):
    """
    V2 shared coarse context builder.

    Inputs:
      cam_s32, lid_s32, rad_s32: [B, 512, H/32, W/32]
      target_hw_s16: (H/16, W/16)

    Outputs:
      z_shared: [B, 256]
      ctx_s16:  [B, 256, H/16, W/16]  # conditioning map for dense reliability heads
      p_cam, p_lid, p_rad: [B, 512]   # pooled s32 priors
    """

    def __init__(self, s32_channels_each: int = 512, d_shared: int = 256):
        super().__init__()
        in_ch = s32_channels_each * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, d_shared, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_shared, d_shared, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _pool_token(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).flatten(1)

    def forward(
        self,
        cam_s32: torch.Tensor,
        lid_s32: torch.Tensor,
        rad_s32: torch.Tensor,
        target_hw_s16: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([cam_s32, lid_s32, rad_s32], dim=1)
        ctx_s32 = self.fuse(x)
        z_shared = self._pool_token(ctx_s32)  # [B, 256]
        ctx_s16 = torch.nn.functional.interpolate(
            ctx_s32, size=target_hw_s16, mode="bilinear", align_corners=False
        )
        return {
            "z_shared": z_shared,
            "ctx_s16": ctx_s16,
            "p_cam": self._pool_token(cam_s32),
            "p_lid": self._pool_token(lid_s32),
            "p_rad": self._pool_token(rad_s32),
        }
