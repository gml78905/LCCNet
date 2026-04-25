from typing import Dict

import torch
import torch.nn as nn


class LearnedImportanceEstimator(nn.Module):
    """
    Learned calibration-usefulness field.

    Inputs:
      J0: [B,384,H/16,W/16]
      support_cam, support_lid, support_rad: [B,1,H/16,W/16]
      diff_energy: [B,3,H/16,W/16]

    Outputs:
      W_cam, W_lid, W_rad, W_joint: [B,1,H/16,W/16]
      z_importance: [B,64]
    """

    def __init__(self, d_joint: int = 384, d_summary: int = 64):
        super().__init__()
        in_ch = d_joint + 6
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head_cam = nn.Conv2d(128, 1, kernel_size=1)
        self.head_lid = nn.Conv2d(128, 1, kernel_size=1)
        self.head_rad = nn.Conv2d(128, 1, kernel_size=1)
        self.head_joint = nn.Conv2d(128, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.summary = nn.Sequential(
            nn.Linear(4 + 3, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, d_summary),
            nn.LayerNorm(d_summary),
        )

    def forward(
        self,
        J0: torch.Tensor,
        support_cam: torch.Tensor,
        support_lid: torch.Tensor,
        support_rad: torch.Tensor,
        diff_cl: torch.Tensor,
        diff_cr: torch.Tensor,
        diff_lr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        diff_energy = torch.cat([
            diff_cl.mean(dim=1, keepdim=True),
            diff_cr.mean(dim=1, keepdim=True),
            diff_lr.mean(dim=1, keepdim=True),
        ], dim=1)
        support = torch.cat([support_cam, support_lid, support_rad], dim=1)
        trunk = self.trunk(torch.cat([J0, support, diff_energy], dim=1))
        W_cam = torch.sigmoid(self.head_cam(trunk))
        W_lid = torch.sigmoid(self.head_lid(trunk)) * support_lid
        W_rad = torch.sigmoid(self.head_rad(trunk)) * support_rad
        W_joint = torch.sigmoid(self.head_joint(trunk))
        summary_in = torch.cat([
            self.pool(W_cam).flatten(1),
            self.pool(W_lid).flatten(1),
            self.pool(W_rad).flatten(1),
            self.pool(W_joint).flatten(1),
            self.pool(diff_energy).flatten(1),
        ], dim=1)
        z_importance = self.summary(summary_in)
        return {
            "W_cam": W_cam,
            "W_lid": W_lid,
            "W_rad": W_rad,
            "W_joint": W_joint,
            "z_importance": z_importance,
            "diff_energy": diff_energy,
            "importance_trunk": trunk,
        }
