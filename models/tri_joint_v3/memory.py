from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class JointTemporalMemory(nn.Module):
    """
    Joint-centered temporal memory with map-aware summary input.

    Input:
      z_joint_fused:            [B,384]
      z_joint_support_weighted: [B,384]
      z_joint_residual_weighted:[B,384]
      z_lid_valid:              [B,384]
      z_rad_valid:              [B,384]
      fusion_map_summary:       [B,64]
      state["h_joint"]:         [B,256]
    """

    def __init__(self, d_in: int = 384, d_hidden: int = 256, d_map_summary: int = 64):
        super().__init__()
        self.d_hidden = d_hidden
        self.pre = nn.Sequential(
            nn.Linear(d_in * 5 + d_map_summary, d_in),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.cell = nn.GRUCell(input_size=d_in, hidden_size=d_hidden)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {"h_joint": torch.zeros((batch_size, self.d_hidden), device=device, dtype=dtype)}

    def forward(
        self,
        z_joint_fused: torch.Tensor,
        z_joint_support_weighted: torch.Tensor,
        z_joint_residual_weighted: torch.Tensor,
        z_lid_valid: torch.Tensor,
        z_rad_valid: torch.Tensor,
        fusion_map_summary: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state is None:
            state = self.init_state(z_joint_fused.shape[0], z_joint_fused.device, z_joint_fused.dtype)
        joint_in = self.pre(torch.cat([
            z_joint_fused,
            z_joint_support_weighted,
            z_joint_residual_weighted,
            z_lid_valid,
            z_rad_valid,
            fusion_map_summary,
        ], dim=1))                                                           # [B,384]
        h_joint = self.cell(joint_in, state["h_joint"])                      # [B,256]
        return h_joint, {"h_joint": h_joint}
