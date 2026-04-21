from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairDeltaPoseHead(nn.Module):
    """
    Predict pairwise delta pose from refined relation state.

    Input:
      r_pair_ref: [B, 384]
    Output:
      delta_t: [B, 3]
      delta_q: [B, 4] (normalized)
    """

    def __init__(self, d_pair: int = 384, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        # TODO(v2): optionally add uncertainty/confidence head if loss design adopts it.
        self.net = nn.Sequential(
            nn.Linear(d_pair, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.fc_t = nn.Linear(hidden_dim, 3)
        self.fc_q = nn.Linear(hidden_dim, 4)

    def forward(self, r_pair_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(r_pair_ref)
        delta_t = self.fc_t(x)
        delta_q = F.normalize(self.fc_q(x), dim=1)
        return delta_t, delta_q


class TriPairDeltaPoseHeads(nn.Module):
    """
    Three pairwise delta heads: CL / CR / LR
    """

    def __init__(self, d_pair: int = 384, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.head_cl = PairDeltaPoseHead(d_pair=d_pair, hidden_dim=hidden_dim, dropout=dropout)
        self.head_cr = PairDeltaPoseHead(d_pair=d_pair, hidden_dim=hidden_dim, dropout=dropout)
        self.head_lr = PairDeltaPoseHead(d_pair=d_pair, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        r_cl_ref: torch.Tensor,
        r_cr_ref: torch.Tensor,
        r_lr_ref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        t_cl, q_cl = self.head_cl(r_cl_ref)
        t_cr, q_cr = self.head_cr(r_cr_ref)
        t_lr, q_lr = self.head_lr(r_lr_ref)
        return {
            "T_CL_t": t_cl,
            "T_CL_q": q_cl,
            "T_CR_t": t_cr,
            "T_CR_q": q_cr,
            "T_LR_t": t_lr,
            "T_LR_q": q_lr,
        }
