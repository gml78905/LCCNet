from typing import Dict, Tuple

import torch
import torch.nn as nn


class ModalityReliabilityHead(nn.Module):
    """
    Learnable scalar modality reliability in (0,1).

    Input:
      z_mod: [B, 256]
      h_shared: [B, 256]
    Output:
      w_mod: [B, 1]
    """

    def __init__(self, d_mod: int = 256, d_shared: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_mod + d_shared, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_mod: torch.Tensor, h_shared: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_mod, h_shared], dim=1)
        return self.net(x)


class PairReliabilityHead(nn.Module):
    """
    Learnable scalar pair reliability in (0,1).

    Input:
      r_pair_coarse: [B, 384]
      h_pair: [B, 256]
      h_shared: [B, 256]
      w_mod_a: [B, 1]
      w_mod_b: [B, 1]
    Output:
      w_pair: [B, 1]
    """

    def __init__(self, d_pair: int = 384, d_pair_mem: int = 256, d_shared: int = 256):
        super().__init__()
        in_dim = d_pair + d_pair_mem + d_shared + 1 + 1  # + modality reliabilities
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        r_pair_coarse: torch.Tensor,
        h_pair: torch.Tensor,
        h_shared: torch.Tensor,
        w_mod_a: torch.Tensor,
        w_mod_b: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([r_pair_coarse, h_pair, h_shared, w_mod_a, w_mod_b], dim=1)
        return self.net(x)


class ReliabilityEstimator(nn.Module):
    """
    Estimate modality and pair reliabilities.
    """

    def __init__(self, d_mod: int = 256, d_pair: int = 384, d_shared: int = 256, d_pair_mem: int = 256):
        super().__init__()
        # TODO(v2): add reliability regularization hooks once loss-side design is finalized.
        self.mod_c = ModalityReliabilityHead(d_mod=d_mod, d_shared=d_shared)
        self.mod_l = ModalityReliabilityHead(d_mod=d_mod, d_shared=d_shared)
        self.mod_r = ModalityReliabilityHead(d_mod=d_mod, d_shared=d_shared)

        self.pair_cl = PairReliabilityHead(d_pair=d_pair, d_pair_mem=d_pair_mem, d_shared=d_shared)
        self.pair_cr = PairReliabilityHead(d_pair=d_pair, d_pair_mem=d_pair_mem, d_shared=d_shared)
        self.pair_lr = PairReliabilityHead(d_pair=d_pair, d_pair_mem=d_pair_mem, d_shared=d_shared)

    def forward(
        self,
        z_cam: torch.Tensor,
        z_lid: torch.Tensor,
        z_rad: torch.Tensor,
        r_cl_coarse: torch.Tensor,
        r_cr_coarse: torch.Tensor,
        r_lr_coarse: torch.Tensor,
        h_shared: torch.Tensor,
        h_cl: torch.Tensor,
        h_cr: torch.Tensor,
        h_lr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        w_c = self.mod_c(z_cam, h_shared)
        w_l = self.mod_l(z_lid, h_shared)
        w_r = self.mod_r(z_rad, h_shared)

        # CL uses w_c, w_l / CR uses w_c, w_r / LR uses w_l, w_r
        w_cl = self.pair_cl(r_cl_coarse, h_cl, h_shared, w_c, w_l)
        w_cr = self.pair_cr(r_cr_coarse, h_cr, h_shared, w_c, w_r)
        w_lr = self.pair_lr(r_lr_coarse, h_lr, h_shared, w_l, w_r)

        return {
            "w_c": w_c,
            "w_l": w_l,
            "w_r": w_r,
            "w_cl": w_cl,
            "w_cr": w_cr,
            "w_lr": w_lr,
        }
