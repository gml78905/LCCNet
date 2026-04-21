from typing import Dict, Tuple

import torch
import torch.nn as nn


class ReliabilityAwarePairAggregator(nn.Module):
    """
    Build pairwise coarse relation states from reliability-weighted s16 evidence.

    Core:
      F_cam_w = R_cam * cam_s16
      F_lid_w = R_lid * lid_s16
      F_rad_w = R_rad * rad_s16
    """

    def __init__(self, feat_ch: int = 256, prior_dim: int = 512, d_shared: int = 256, d_pair: int = 384):
        super().__init__()
        self.feat_ch = feat_ch
        self.eps = 1e-6
        in_dim = feat_ch * 2 + prior_dim * 2 + d_shared

        self.rel_cl = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, d_pair),
        )
        self.rel_cr = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, d_pair),
        )
        self.rel_lr = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, d_pair),
        )

    def _weighted_token(self, feat: torch.Tensor, reli: torch.Tensor) -> torch.Tensor:
        # feat: [B,C,H,W], reli: [B,1,H,W]
        feat_w = feat * reli
        num = feat_w.sum(dim=(2, 3))
        den = reli.sum(dim=(2, 3)).clamp(min=self.eps)
        return num / den

    def forward(
        self,
        cam_s16: torch.Tensor,
        lid_s16: torch.Tensor,
        rad_s16: torch.Tensor,
        r_cam: torch.Tensor,
        r_lid: torch.Tensor,
        r_rad: torch.Tensor,
        p_cam: torch.Tensor,
        p_lid: torch.Tensor,
        p_rad: torch.Tensor,
        z_shared: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Reliability-weighted evidence tokens from s16 maps
        tok_cam = self._weighted_token(cam_s16, r_cam)  # [B,256]
        tok_lid = self._weighted_token(lid_s16, r_lid)  # [B,256]
        tok_rad = self._weighted_token(rad_s16, r_rad)  # [B,256]

        rel_cl_in = torch.cat([tok_cam, tok_lid, p_cam, p_lid, z_shared], dim=1)
        rel_cr_in = torch.cat([tok_cam, tok_rad, p_cam, p_rad, z_shared], dim=1)
        rel_lr_in = torch.cat([tok_lid, tok_rad, p_lid, p_rad, z_shared], dim=1)

        r_cl0 = self.rel_cl(rel_cl_in)  # [B,384]
        r_cr0 = self.rel_cr(rel_cr_in)  # [B,384]
        r_lr0 = self.rel_lr(rel_lr_in)  # [B,384]

        # Optional pair scalar reliability from dense maps (reused by refinement)
        w_cl = ((r_cam + r_lid) * 0.5).mean(dim=(2, 3))
        w_cr = ((r_cam + r_rad) * 0.5).mean(dim=(2, 3))
        w_lr = ((r_lid + r_rad) * 0.5).mean(dim=(2, 3))

        return {
            "F_cam_w": cam_s16 * r_cam,
            "F_lid_w": lid_s16 * r_lid,
            "F_rad_w": rad_s16 * r_rad,
            "r_cl0": r_cl0,
            "r_cr0": r_cr0,
            "r_lr0": r_lr0,
            "w_cl": w_cl,
            "w_cr": w_cr,
            "w_lr": w_lr,
        }

