from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointAlignmentFeatureBuilder(nn.Module):
    """
    Build a dense joint alignment feature instead of a generic fused feature.

    Inputs:
      cam_s16, lid_s16, rad_s16, ctx_s16: [B,256,H/16,W/16]

    Outputs:
      F_joint_map:      [B,384,H/16,W/16]
      z_joint:          [B,384]
      z_joint_support:  [B,384]
      z_joint_conflict: [B,384]
      z_joint_align:    [B,384]

    Notes:
      - internal state keeps support / common structure / conflict / directional
        alignment precursor separate for longer before fusion
      - this is still lightweight enough for v3.1-lite and not a full cost volume
    """

    def __init__(
        self,
        feat_ch: int = 256,
        ctx_ch: int = 256,
        d_common: int = 128,
        d_joint: int = 384,
    ):
        super().__init__()
        self.d_joint = d_joint
        self.proj_cam = nn.Sequential(
            nn.Conv2d(feat_ch, d_common, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_common),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj_lid = nn.Sequential(
            nn.Conv2d(feat_ch, d_common, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_common),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj_rad = nn.Sequential(
            nn.Conv2d(feat_ch, d_common, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_common),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj_ctx = nn.Sequential(
            nn.Conv2d(ctx_ch, d_common, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_common),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.support_branch = nn.Sequential(
            nn.Conv2d(d_common * 4, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.common_branch = nn.Sequential(
            nn.Conv2d(d_common * 4, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conflict_branch = nn.Sequential(
            nn.Conv2d(d_common * 4, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.align_precursor_branch = nn.Sequential(
            nn.Conv2d(d_common * 5, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(96 * 4, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(384, d_joint, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_joint),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.score_head = nn.Conv2d(96, 1, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _weighted_pool(feat: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        # feat: [B,C,H,W], score: [B,1,H,W] -> [B,C]
        weight = torch.softmax(score.flatten(2), dim=-1).unsqueeze(1)  # [B,1,1,HW]
        feat_flat = feat.flatten(2).unsqueeze(2)                       # [B,C,1,HW]
        return (feat_flat * weight).sum(dim=-1).squeeze(2)

    def forward(
        self,
        cam_s16: torch.Tensor,
        lid_s16: torch.Tensor,
        rad_s16: torch.Tensor,
        ctx_s16: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cam_j = self.proj_cam(cam_s16)                                  # [B,128,H/16,W/16]
        lid_j = self.proj_lid(lid_s16)                                  # [B,128,H/16,W/16]
        rad_j = self.proj_rad(rad_s16)                                  # [B,128,H/16,W/16]
        ctx_j = self.proj_ctx(ctx_s16)                                  # [B,128,H/16,W/16]

        mean_mod = (cam_j + lid_j + rad_j) / 3.0                        # [B,128,H/16,W/16]
        diff_cl = torch.abs(cam_j - lid_j)                              # [B,128,H/16,W/16]
        diff_cr = torch.abs(cam_j - rad_j)                              # [B,128,H/16,W/16]
        diff_lr = torch.abs(lid_j - rad_j)                              # [B,128,H/16,W/16]
        conflict_seed = (diff_cl + diff_cr + diff_lr) / 3.0             # [B,128,H/16,W/16]

        support = self.support_branch(
            torch.cat([cam_j, lid_j, rad_j, ctx_j], dim=1)
        )                                                                # [B,96,H/16,W/16]
        common_structure = self.common_branch(
            torch.cat([mean_mod, torch.minimum(cam_j, lid_j), torch.minimum(cam_j, rad_j), ctx_j], dim=1)
        )                                                                # [B,96,H/16,W/16]
        conflict = self.conflict_branch(
            torch.cat([diff_cl, diff_cr, diff_lr, conflict_seed], dim=1)
        )                                                                # [B,96,H/16,W/16]
        align_precursor = self.align_precursor_branch(
            torch.cat([cam_j, lid_j, rad_j, mean_mod, ctx_j], dim=1)
        )                                                                # [B,96,H/16,W/16]

        F_joint_map = self.fusion(
            torch.cat([support, common_structure, conflict, align_precursor], dim=1)
        )                                                                # [B,384,H/16,W/16]
        z_joint = self.pool(F_joint_map).flatten(1)                      # [B,384]

        support_score = self.score_head(support)                         # [B,1,H/16,W/16]
        conflict_score = self.score_head(conflict)                       # [B,1,H/16,W/16]
        align_score = self.score_head(align_precursor)                   # [B,1,H/16,W/16]
        z_joint_support = self._weighted_pool(F_joint_map, support_score)# [B,384]
        z_joint_conflict = self._weighted_pool(F_joint_map, conflict_score)  # [B,384]
        z_joint_align = self._weighted_pool(F_joint_map, align_score)    # [B,384]

        return {
            "F_joint_map": F_joint_map,
            "z_joint": z_joint,
            "z_joint_support": z_joint_support,
            "z_joint_conflict": z_joint_conflict,
            "z_joint_align": z_joint_align,
            "support_branch": support,
            "common_structure": common_structure,
            "conflict_branch": conflict,
            "align_precursor": align_precursor,
            "diff_cl": diff_cl,
            "diff_cr": diff_cr,
            "diff_lr": diff_lr,
        }


# Backward-compatible alias for existing imports.
JointCalibrationFeatureBuilder = JointAlignmentFeatureBuilder
