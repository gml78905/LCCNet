from typing import Dict

import torch
import torch.nn as nn


class ImportanceAwareJointFusion(nn.Module):
    """
    Fuse modality features using learned importance maps into a dense joint field.
    """

    def __init__(self, feat_ch: int = 128, d_joint: int = 384, d_summary: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_ch * 3 + d_joint + 4, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(384, d_joint, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_joint),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.summary = nn.Sequential(
            nn.Linear(d_joint * 4, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, d_summary),
            nn.LayerNorm(d_summary),
        )

    @staticmethod
    def _weighted_token(feat: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        num = (feat * weight).sum(dim=(2, 3))
        den = weight.sum(dim=(2, 3)).clamp(min=eps)
        return num / den

    def forward(
        self,
        J0: torch.Tensor,
        cam_feat: torch.Tensor,
        lid_feat: torch.Tensor,
        rad_feat: torch.Tensor,
        W_cam: torch.Tensor,
        W_lid: torch.Tensor,
        W_rad: torch.Tensor,
        W_joint: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cam_w = cam_feat * W_cam
        lid_w = lid_feat * W_lid
        rad_w = rad_feat * W_rad
        J1 = self.fuse(torch.cat([cam_w, lid_w, rad_w, J0, W_cam, W_lid, W_rad, W_joint], dim=1))
        z_joint = self.pool(J1).flatten(1)
        z_cam = self._weighted_token(J1, W_cam)
        z_lid = self._weighted_token(J1, W_lid)
        z_rad = self._weighted_token(J1, W_rad)
        z_summary = self.summary(torch.cat([z_joint, z_cam, z_lid, z_rad], dim=1))
        return {
            "J1": J1,
            "z_joint": z_joint,
            "z_cam": z_cam,
            "z_lid": z_lid,
            "z_rad": z_rad,
            "z_summary": z_summary,
            "cam_w": cam_w,
            "lid_w": lid_w,
            "rad_w": rad_w,
        }
