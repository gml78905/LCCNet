from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DensePairPoseHead(nn.Module):
    def __init__(
        self,
        d_joint: int = 384,
        d_global: int = 384,
        hidden_dim: int = 256,
        max_delta_t: float = 0.25,
        max_fine_t: float = 0.05,
        max_delta_r_deg: float = 5.0,
    ):
        super().__init__()
        self.max_delta_t = max_delta_t
        self.max_fine_t = max_fine_t
        self.max_delta_r = max_delta_r_deg * 3.141592653589793 / 180.0
        self.attn = nn.Sequential(
            nn.Conv2d(d_joint, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, bias=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_joint + d_global, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc_t_coarse = nn.Linear(hidden_dim, 3)
        self.fc_t_fine = nn.Linear(hidden_dim, 3)
        self.fc_r = nn.Linear(hidden_dim, 3)

    @staticmethod
    def _axis_angle_to_quat(rot_vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        angle = torch.linalg.norm(rot_vec, dim=1, keepdim=True).clamp(min=eps)
        axis = rot_vec / angle
        half = 0.5 * angle
        quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=1)
        return F.normalize(quat, dim=1)

    def _spatial_token(self, J_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.attn(J_ref)
        b, _, h, w = logits.shape
        prob = torch.softmax(logits.view(b, 1, h * w), dim=-1).view(b, 1, h, w)
        token = (J_ref * prob).sum(dim=(2, 3))
        return token, prob

    def forward(self, J_ref: torch.Tensor, z_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token, attn = self._spatial_token(J_ref)
        feat = self.mlp(torch.cat([token, z_global], dim=1))
        delta_t = self.max_delta_t * torch.tanh(self.fc_t_coarse(feat)) + self.max_fine_t * torch.tanh(self.fc_t_fine(feat))
        delta_q = self._axis_angle_to_quat(self.max_delta_r * torch.tanh(self.fc_r(feat)))
        return delta_t, delta_q, attn


class PairwiseDenseDeltaReadout(nn.Module):
    def __init__(
        self,
        d_joint: int = 384,
        d_global: int = 384,
        hidden_dim: int = 256,
        max_delta_t: float = 0.25,
        max_fine_t: float = 0.05,
        max_delta_r_deg: float = 5.0,
    ):
        super().__init__()
        self.head_cl = _DensePairPoseHead(d_joint, d_global, hidden_dim, max_delta_t, max_fine_t, max_delta_r_deg)
        self.head_cr = _DensePairPoseHead(d_joint, d_global, hidden_dim, max_delta_t, max_fine_t, max_delta_r_deg)
        self.head_lr = _DensePairPoseHead(d_joint, d_global, hidden_dim, max_delta_t, max_fine_t, max_delta_r_deg)

    def forward(self, J_ref: torch.Tensor, z_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_cl, q_cl, a_cl = self.head_cl(J_ref, z_global)
        t_cr, q_cr, a_cr = self.head_cr(J_ref, z_global)
        t_lr, q_lr, a_lr = self.head_lr(J_ref, z_global)
        return {
            "T_CL_t": t_cl,
            "T_CL_q": q_cl,
            "T_CR_t": t_cr,
            "T_CR_q": q_cr,
            "T_LR_t": t_lr,
            "T_LR_q": q_lr,
            "A_CL": a_cl,
            "A_CR": a_cr,
            "A_LR": a_lr,
        }
