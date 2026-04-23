from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PairDeltaPoseHead(nn.Module):
    def __init__(
        self,
        d_in: int = 256,
        max_delta_t: float = 0.25,
        max_fine_t: float = 0.05,
        max_delta_r_deg: float = 5.0,
    ):
        super().__init__()
        self.max_delta_t = max_delta_t
        self.max_fine_t = max_fine_t
        self.max_delta_r = max_delta_r_deg * 3.141592653589793 / 180.0
        self.fc_t_coarse = nn.Linear(d_in, 3)
        self.fc_t_fine = nn.Linear(d_in, 3)
        self.fc_r = nn.Linear(d_in, 3)

    @staticmethod
    def _axis_angle_to_quat(rot_vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # rot_vec: [B,3] axis-angle in radians. Output quaternion is [w,x,y,z].
        angle = torch.linalg.norm(rot_vec, dim=1, keepdim=True).clamp(min=eps)
        axis = rot_vec / angle
        half = 0.5 * angle
        quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=1)
        return F.normalize(quat, dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Coarse/fine bounded translation prevents harmful overcorrection while
        # still allowing a smaller residual term for late-stage precision.
        delta_t = (
            self.max_delta_t * torch.tanh(self.fc_t_coarse(x))
            + self.max_fine_t * torch.tanh(self.fc_t_fine(x))
        )                                                          # [B,3]
        rot_vec = self.max_delta_r * torch.tanh(self.fc_r(x))      # [B,3]
        delta_q = self._axis_angle_to_quat(rot_vec)                # [B,4]
        return delta_t, delta_q


class JointToPairDeltaPoseHeads(nn.Module):
    """
    Read pairwise delta pose only at the final stage from a shared joint latent.

    Input:
      z_joint_ref: [B,384]

    Output:
      pred dict with CL / CR / LR delta pose
    """

    def __init__(
        self,
        d_joint: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        max_delta_t: float = 0.25,
        max_fine_t: float = 0.05,
        max_delta_r_deg: float = 5.0,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_joint, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.adapt_cl = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1, inplace=True))
        self.adapt_cr = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1, inplace=True))
        self.adapt_lr = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1, inplace=True))
        self.head_cl = _PairDeltaPoseHead(
            d_in=hidden_dim,
            max_delta_t=max_delta_t,
            max_fine_t=max_fine_t,
            max_delta_r_deg=max_delta_r_deg,
        )
        self.head_cr = _PairDeltaPoseHead(
            d_in=hidden_dim,
            max_delta_t=max_delta_t,
            max_fine_t=max_fine_t,
            max_delta_r_deg=max_delta_r_deg,
        )
        self.head_lr = _PairDeltaPoseHead(
            d_in=hidden_dim,
            max_delta_t=max_delta_t,
            max_fine_t=max_fine_t,
            max_delta_r_deg=max_delta_r_deg,
        )

    def forward(self, z_joint_ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(z_joint_ref)                # [B,256]
        f_cl = self.adapt_cl(shared)                     # [B,256]
        f_cr = self.adapt_cr(shared)                     # [B,256]
        f_lr = self.adapt_lr(shared)                     # [B,256]
        t_cl, q_cl = self.head_cl(f_cl)
        t_cr, q_cr = self.head_cr(f_cr)
        t_lr, q_lr = self.head_lr(f_lr)
        return {
            "T_CL_t": t_cl,
            "T_CL_q": q_cl,
            "T_CR_t": t_cr,
            "T_CR_q": q_cr,
            "T_LR_t": t_lr,
            "T_LR_q": q_lr,
        }
