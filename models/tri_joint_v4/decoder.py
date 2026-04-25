from typing import Dict

import torch
import torch.nn as nn


class JointCalibrationDecoder(nn.Module):
    """
    Decode dense joint field + dense memory into refined calibration field.
    """

    def __init__(self, d_joint: int = 384, d_mem: int = 256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(d_joint + d_mem + 4, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(384, d_joint, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_joint),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(d_joint, d_joint, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_joint),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        J1: torch.Tensor,
        m_joint: torch.Tensor,
        W_cam: torch.Tensor,
        W_lid: torch.Tensor,
        W_rad: torch.Tensor,
        W_joint: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        J_ref = self.fuse(torch.cat([J1, m_joint, W_cam, W_lid, W_rad, W_joint], dim=1))
        J_ref = J_ref + self.refine(J_ref)
        z_global = self.pool(J_ref).flatten(1)
        return {"J_ref": J_ref, "z_global": z_global}
