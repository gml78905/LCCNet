from typing import Dict

import torch
import torch.nn as nn


class DenseModalityReliabilityHead(nn.Module):
    """
    Build dense reliability map for a modality on s16 grid.

    Inputs:
      feat_s16: [B, 256, H/16, W/16]
      ctx_s16:  [B, 256, H/16, W/16]
    Output:
      R: [B, 1, H/16, W/16] in (0,1)
    """

    def __init__(self, feat_ch: int = 256, ctx_ch: int = 256, hidden_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch + ctx_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat_s16: torch.Tensor, ctx_s16: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_s16, ctx_s16], dim=1)
        return self.net(x)


class DenseModalityReliabilityEstimator(nn.Module):
    """
    V2 dense reliability maps:
      R_cam, R_lid, R_rad: [B,1,H/16,W/16]
    """

    def __init__(self, feat_ch: int = 256, ctx_ch: int = 256):
        super().__init__()
        self.cam_head = DenseModalityReliabilityHead(feat_ch=feat_ch, ctx_ch=ctx_ch)
        self.lid_head = DenseModalityReliabilityHead(feat_ch=feat_ch, ctx_ch=ctx_ch)
        self.rad_head = DenseModalityReliabilityHead(feat_ch=feat_ch, ctx_ch=ctx_ch)

    @staticmethod
    def _map_to_scalar(reli_map: torch.Tensor) -> torch.Tensor:
        # [B,1,H,W] -> [B,1]
        return reli_map.mean(dim=(2, 3))

    def forward(
        self,
        cam_s16: torch.Tensor,
        lid_s16: torch.Tensor,
        rad_s16: torch.Tensor,
        ctx_s16: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r_cam = self.cam_head(cam_s16, ctx_s16)
        r_lid = self.lid_head(lid_s16, ctx_s16)
        r_rad = self.rad_head(rad_s16, ctx_s16)
        return {
            "R_cam": r_cam,
            "R_lid": r_lid,
            "R_rad": r_rad,
            "w_c": self._map_to_scalar(r_cam),
            "w_l": self._map_to_scalar(r_lid),
            "w_r": self._map_to_scalar(r_rad),
        }

