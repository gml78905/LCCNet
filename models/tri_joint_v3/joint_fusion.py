from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEvidenceFusion(nn.Module):
    """
    Preserve dense alignment information longer before collapsing to a single vector.

    Inputs:
      F_joint_map: [B,384,H/16,W/16]
      E_joint_map: [B,192,H/16,W/16]
      R_cam, R_lid, R_rad: [B,1,H/16,W/16]

    Outputs:
      F_joint_fused:           [B,384,H/16,W/16]
      z_joint_fused:           [B,384]
      z_joint_support_weighted:[B,384]
      z_joint_residual_weighted:[B,384]
      z_lid_valid:             [B,384]
      z_rad_valid:             [B,384]
      fusion_map_summary:      [B,64]
    """

    def __init__(self, d_joint: int = 384, d_res: int = 192, d_fused: int = 384):
        super().__init__()
        self.res_proj = nn.Sequential(
            nn.Conv2d(d_res, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.reli_proj = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(d_joint + 128 + 32, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(384, d_fused, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_fused),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(d_fused * 4, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _weighted_token(feat: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        feat_w = feat * weight
        num = feat_w.sum(dim=(2, 3))                                        # [B,C]
        den = weight.sum(dim=(2, 3)).clamp(min=eps)                         # [B,1]
        return num / den

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        pad = kernel_size // 2
        return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)

    def forward(
        self,
        F_joint_map: torch.Tensor,
        E_joint_map: torch.Tensor,
        R_cam: torch.Tensor,
        R_lid: torch.Tensor,
        R_rad: torch.Tensor,
        valid_lid: torch.Tensor = None,
        valid_rad: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        reli_stack = torch.cat([R_cam, R_lid, R_rad], dim=1)                # [B,3,H/16,W/16]
        residual_proj = self.res_proj(E_joint_map)                          # [B,128,H/16,W/16]
        reliability_proj = self.reli_proj(reli_stack)                       # [B,32,H/16,W/16]
        x = torch.cat([F_joint_map, residual_proj, reliability_proj], dim=1)
        F_joint_fused = self.fuse(x)                                        # [B,384,H/16,W/16]

        residual_strength = E_joint_map.abs().mean(dim=1, keepdim=True)     # [B,1,H/16,W/16]
        support_strength = reli_stack.mean(dim=1, keepdim=True)             # [B,1,H/16,W/16]
        z_joint_fused = self.pool(F_joint_fused).flatten(1)                 # [B,384]
        z_joint_support_weighted = self._weighted_token(F_joint_fused, support_strength)
        z_joint_residual_weighted = self._weighted_token(F_joint_fused, residual_strength)
        if valid_lid is None:
            valid_lid = (R_lid > 0).to(dtype=F_joint_fused.dtype)
        if valid_rad is None:
            valid_rad = (R_rad > 0).to(dtype=F_joint_fused.dtype)
        valid_lid_w = self._dilate_mask(valid_lid, kernel_size=5)
        valid_rad_w = self._dilate_mask(valid_rad, kernel_size=5)
        z_lid_valid = self._weighted_token(F_joint_fused, valid_lid_w)
        z_rad_valid = self._weighted_token(F_joint_fused, valid_rad_w)
        fusion_map_summary = self.summary_mlp(
            torch.cat([z_joint_support_weighted, z_joint_residual_weighted, z_lid_valid, z_rad_valid], dim=1)
        )                                                                   # [B,64]
        return {
            "F_joint_fused": F_joint_fused,
            "z_joint_fused": z_joint_fused,
            "z_joint_support_weighted": z_joint_support_weighted,
            "z_joint_residual_weighted": z_joint_residual_weighted,
            "z_lid_valid": z_lid_valid,
            "z_rad_valid": z_rad_valid,
            "fusion_map_summary": fusion_map_summary,
            "support_strength": support_strength,
            "residual_strength": residual_strength,
            "valid_lid_w": valid_lid_w,
            "valid_rad_w": valid_rad_w,
        }
