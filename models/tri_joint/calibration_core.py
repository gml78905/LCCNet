from typing import Dict, Tuple

import torch
import torch.nn as nn


class ResidualRefinementCore(nn.Module):
    """
    Shared refinement philosophy:
      shared trunk + pair-specific delta/gate heads.

    Residual refinement rule (required):
      r_ref = r_coarse + gate * delta_r
    """

    def __init__(self, d_pair: int = 384, d_pair_mem: int = 256, d_shared: int = 256):
        super().__init__()
        # TODO(v2): consider shared cross-pair interaction (graph/attention) before pair-specific heads.
        # x_pair = [r_pair_coarse, h_pair, h_shared, w_pair, w_mod_a, w_mod_b]
        in_dim = d_pair + d_pair_mem + d_shared + 1 + 1 + 1  # = 899
        self.shared_trunk = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, d_pair),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.delta_cl = nn.Linear(d_pair, d_pair)
        self.delta_cr = nn.Linear(d_pair, d_pair)
        self.delta_lr = nn.Linear(d_pair, d_pair)

        self.gate_cl = nn.Sequential(nn.Linear(d_pair, d_pair), nn.Sigmoid())
        self.gate_cr = nn.Sequential(nn.Linear(d_pair, d_pair), nn.Sigmoid())
        self.gate_lr = nn.Sequential(nn.Linear(d_pair, d_pair), nn.Sigmoid())

    def _pack(
        self,
        r_pair_coarse: torch.Tensor,
        h_pair: torch.Tensor,
        h_shared: torch.Tensor,
        w_pair: torch.Tensor,
        w_mod_a: torch.Tensor,
        w_mod_b: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([r_pair_coarse, h_pair, h_shared, w_pair, w_mod_a, w_mod_b], dim=1)

    def _refine_one(
        self,
        trunk_feat: torch.Tensor,
        r_pair_coarse: torch.Tensor,
        delta_head: nn.Module,
        gate_head: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_r = delta_head(trunk_feat)                        # [B, 384]
        gate = gate_head(trunk_feat)                            # [B, 384]
        r_ref = r_pair_coarse + gate * delta_r                  # residual refinement
        return r_ref, delta_r, gate

    def forward(
        self,
        r_cl_coarse: torch.Tensor,
        r_cr_coarse: torch.Tensor,
        r_lr_coarse: torch.Tensor,
        h_cl: torch.Tensor,
        h_cr: torch.Tensor,
        h_lr: torch.Tensor,
        h_shared: torch.Tensor,
        w_c: torch.Tensor,
        w_l: torch.Tensor,
        w_r: torch.Tensor,
        w_cl: torch.Tensor,
        w_cr: torch.Tensor,
        w_lr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x_cl = self._pack(r_cl_coarse, h_cl, h_shared, w_cl, w_c, w_l)
        x_cr = self._pack(r_cr_coarse, h_cr, h_shared, w_cr, w_c, w_r)
        x_lr = self._pack(r_lr_coarse, h_lr, h_shared, w_lr, w_l, w_r)

        f_cl = self.shared_trunk(x_cl)
        f_cr = self.shared_trunk(x_cr)
        f_lr = self.shared_trunk(x_lr)

        r_cl_ref, delta_r_cl, gate_cl = self._refine_one(f_cl, r_cl_coarse, self.delta_cl, self.gate_cl)
        r_cr_ref, delta_r_cr, gate_cr = self._refine_one(f_cr, r_cr_coarse, self.delta_cr, self.gate_cr)
        r_lr_ref, delta_r_lr, gate_lr = self._refine_one(f_lr, r_lr_coarse, self.delta_lr, self.gate_lr)

        return {
            "r_cl_ref": r_cl_ref,
            "r_cr_ref": r_cr_ref,
            "r_lr_ref": r_lr_ref,
            "delta_r_cl": delta_r_cl,
            "delta_r_cr": delta_r_cr,
            "delta_r_lr": delta_r_lr,
            "gate_cl": gate_cl,
            "gate_cr": gate_cr,
            "gate_lr": gate_lr,
        }
