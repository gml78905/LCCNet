from typing import Dict

import torch
import torch.nn as nn


class JointResidualRefinement(nn.Module):
    """
    Residual refinement with dense-aware summaries.

    z_joint_ref = z_joint_fused + gate_z * delta_z
    """

    def __init__(self, d_joint: int = 384, d_mem: int = 256, d_summary: int = 64):
        super().__init__()
        in_dim = d_joint + d_mem + d_summary * 3
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, d_joint),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.delta = nn.Linear(d_joint, d_joint)
        self.gate = nn.Sequential(
            nn.Linear(d_joint, d_joint),
            nn.Sigmoid(),
        )
        self.map_summary_proj = nn.Sequential(
            nn.Linear(d_joint * 4, d_summary),
            nn.LayerNorm(d_summary),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.summary_bias = nn.Sequential(
            nn.Linear(d_summary * 3, d_joint),
            nn.LayerNorm(d_joint),
        )

    def forward(
        self,
        z_joint_fused: torch.Tensor,
        h_joint: torch.Tensor,
        r_summary: torch.Tensor,
        e_align_summary: torch.Tensor,
        z_joint_support_weighted: torch.Tensor,
        z_joint_residual_weighted: torch.Tensor,
        z_lid_valid: torch.Tensor,
        z_rad_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dense_summary = self.map_summary_proj(
            torch.cat([z_joint_support_weighted, z_joint_residual_weighted, z_lid_valid, z_rad_valid], dim=1)
        )                                                                    # [B,64]
        summary_bias = self.summary_bias(
            torch.cat([r_summary, e_align_summary, dense_summary], dim=1)
        )                                                                    # [B,384]
        trunk_feat = self.trunk(
            torch.cat([z_joint_fused, h_joint, r_summary, e_align_summary, dense_summary], dim=1)
        )                                                                    # [B,384]
        delta_z = self.delta(trunk_feat)                                     # [B,384]
        gate_z = self.gate(trunk_feat)                                       # [B,384]
        z_joint_ref = z_joint_fused + gate_z * delta_z + 0.1 * summary_bias  # [B,384]
        return {
            "z_joint_ref": z_joint_ref,
            "delta_z": delta_z,
            "gate_z": gate_z,
            "dense_summary": dense_summary,
            "summary_bias": summary_bias,
        }
