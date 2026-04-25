from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseJointFieldBuilder(nn.Module):
    """
    Build a dense tri-sensor joint calibration field.

    Inputs:
      cam_s16, lid_s16, rad_s16: [B,256,H/16,W/16]
      ctx_s16:                   [B,256,H/16,W/16]
      support_cam, support_lid, support_rad: [B,1,H/16,W/16]

    Outputs:
      J0: [B,384,H/16,W/16]
      z_joint0: [B,384]
    """

    def __init__(self, feat_ch: int = 256, d_proj: int = 128, d_joint: int = 384):
        super().__init__()
        self.cam_proj = nn.Sequential(
            nn.Conv2d(feat_ch, d_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_proj),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.lid_proj = nn.Sequential(
            nn.Conv2d(feat_ch, d_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_proj),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.rad_proj = nn.Sequential(
            nn.Conv2d(feat_ch, d_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_proj),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.ctx_gate = nn.Sequential(
            nn.Conv2d(feat_ch, d_proj * 3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.support_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.agreement_branch = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conflict_branch = nn.Sequential(
            nn.Conv2d(d_proj * 3, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.directional_branch = nn.Sequential(
            nn.Conv2d(d_proj * 3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(d_proj * 3 + 32 + 64 + 96 + 64, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(384, d_joint, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_joint),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _cosine_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_n = F.normalize(a, dim=1)
        b_n = F.normalize(b, dim=1)
        return (a_n * b_n).sum(dim=1, keepdim=True)

    def forward(
        self,
        cam_s16: torch.Tensor,
        lid_s16: torch.Tensor,
        rad_s16: torch.Tensor,
        ctx_s16: torch.Tensor,
        support_cam: torch.Tensor,
        support_lid: torch.Tensor,
        support_rad: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cam = self.cam_proj(cam_s16)
        lid = self.lid_proj(lid_s16)
        rad = self.rad_proj(rad_s16)
        gates = self.ctx_gate(ctx_s16)
        gate_cam, gate_lid, gate_rad = torch.chunk(gates, chunks=3, dim=1)
        cam = cam * gate_cam
        lid = lid * gate_lid
        rad = rad * gate_rad

        support_maps = torch.cat([support_cam, support_lid, support_rad], dim=1)
        support_branch = self.support_branch(support_maps)

        agree_maps = torch.cat([
            self._cosine_map(cam, lid),
            self._cosine_map(cam, rad),
            self._cosine_map(lid, rad),
            support_cam * support_lid,
            support_cam * support_rad,
            support_lid * support_rad,
        ], dim=1)
        agreement_branch = self.agreement_branch(agree_maps)

        diff_cl = torch.abs(cam - lid)
        diff_cr = torch.abs(cam - rad)
        diff_lr = torch.abs(lid - rad)
        conflict_branch = self.conflict_branch(torch.cat([diff_cl, diff_cr, diff_lr], dim=1))

        directional_branch = self.directional_branch(torch.cat([
            cam - 0.5 * (lid + rad),
            lid - 0.5 * (cam + rad),
            rad - 0.5 * (cam + lid),
        ], dim=1))

        J0 = self.fuse(torch.cat([
            cam,
            lid,
            rad,
            support_branch,
            agreement_branch,
            conflict_branch,
            directional_branch,
        ], dim=1))
        z_joint0 = self.pool(J0).flatten(1)
        return {
            "J0": J0,
            "z_joint0": z_joint0,
            "cam_proj": cam,
            "lid_proj": lid,
            "rad_proj": rad,
            "support_branch": support_branch,
            "agreement_branch": agreement_branch,
            "conflict_branch": conflict_branch,
            "directional_branch": directional_branch,
            "diff_cl": diff_cl,
            "diff_cr": diff_cr,
            "diff_lr": diff_lr,
            "agree_maps": agree_maps,
        }
