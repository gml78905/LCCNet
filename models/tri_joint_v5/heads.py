from typing import Dict

import torch
import torch.nn as nn


class _PairHead(nn.Module):
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.t_head = nn.Linear(128, 3)
        self.q_head = nn.Linear(128, 4)

    def forward(self, z: torch.Tensor):
        h = self.trunk(z)
        t = self.t_head(h)
        q = self.q_head(h)
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return t, q


class PairwiseDeltaHeads(nn.Module):
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.head_cl = _PairHead(in_dim)
        self.head_cr = _PairHead(in_dim)
        self.head_lr = _PairHead(in_dim)

    def forward(
        self,
        z_cl: torch.Tensor,
        z_cr: torch.Tensor = None,
        z_lr: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if z_cr is None:
            z_cr = z_cl
        if z_lr is None:
            z_lr = z_cl

        t_cl, q_cl = self.head_cl(z_cl)
        t_cr, q_cr = self.head_cr(z_cr)
        t_lr, q_lr = self.head_lr(z_lr)
        return {
            "T_CL_t": t_cl,
            "T_CL_q": q_cl,
            "T_CR_t": t_cr,
            "T_CR_q": q_cr,
            "T_LR_t": t_lr,
            "T_LR_q": q_lr,
        }
