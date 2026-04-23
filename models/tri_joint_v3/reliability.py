from typing import Dict

import torch
import torch.nn as nn


class _ResidualAwareReliabilityHead(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualAwareReliabilityEstimator(nn.Module):
    """
    Reliability after residual understanding, now more alignment-aware.

    Inputs:
      F_joint_map:      [B,384,H/16,W/16]
      E_joint_map:      [B,192,H/16,W/16]
      support_cam:      [B,1,H/16,W/16]
      valid_lid/rad:    [B,1,H/16,W/16]
      e_joint:          [B,192]
      e_align_summary:  [B,64]

    Outputs:
      R_cam, R_lid, R_rad: [B,1,H/16,W/16]
      r_summary:           [B,64]
      invalid_penalty:     [B,3]
    """

    def __init__(
        self,
        d_joint: int = 384,
        d_res: int = 192,
        d_summary: int = 64,
        range_reliability_floor: float = 0.05,
        radar_reliability_floor: float = 0.15,
    ):
        super().__init__()
        self.range_reliability_floor = range_reliability_floor
        self.radar_reliability_floor = radar_reliability_floor
        in_ch = d_joint + d_res + 2
        self.cam_head = _ResidualAwareReliabilityHead(in_ch=in_ch)
        self.lid_head = _ResidualAwareReliabilityHead(in_ch=in_ch)
        self.rad_head = _ResidualAwareReliabilityHead(in_ch=in_ch)
        self.summary_mlp = nn.Sequential(
            nn.Linear(192 + 64 + 18, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, d_summary),
            nn.LayerNorm(d_summary),
        )

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)   # [B,2]

    @staticmethod
    def _invalid_penalty(r_map: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Hook for future loss regularization:
        # high confidence on invalid cells should be penalized outside the model.
        return (r_map * (1.0 - valid)).mean(dim=(2, 3))                    # [B,1]

    @staticmethod
    def _valid_stats(x: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Valid-cell normalized stats keep sparse LiDAR/Radar confidence visible
        # instead of letting empty cells dominate the mean/std.
        valid = valid.float()
        den = valid.sum(dim=(2, 3)).clamp(min=eps)                         # [B,1]
        mean = (x * valid).sum(dim=(2, 3)) / den                            # [B,1]
        var = (((x - mean[:, :, None, None]) * valid) ** 2).sum(dim=(2, 3)) / den
        return torch.cat([mean, torch.sqrt(var + eps)], dim=1)              # [B,2]

    def forward(
        self,
        F_joint_map: torch.Tensor,
        E_joint_map: torch.Tensor,
        support_cam: torch.Tensor,
        valid_lid: torch.Tensor,
        valid_rad: torch.Tensor,
        e_joint: torch.Tensor,
        e_align_summary: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cam_in = torch.cat([F_joint_map, E_joint_map, support_cam, 1.0 - support_cam], dim=1)
        lid_in = torch.cat([F_joint_map, E_joint_map, valid_lid, 1.0 - valid_lid], dim=1)
        rad_in = torch.cat([F_joint_map, E_joint_map, valid_rad, 1.0 - valid_rad], dim=1)

        R_cam_raw = self.cam_head(cam_in)                                   # [B,1,H/16,W/16]
        R_lid_raw = self.lid_head(lid_in)                                   # [B,1,H/16,W/16]
        R_rad_raw = self.rad_head(rad_in)                                   # [B,1,H/16,W/16]

        # Keep camera dense but explicitly support-aware to reduce collapse risk.
        camera_floor = 0.15 + 0.85 * support_cam
        R_cam = torch.sigmoid(R_cam_raw) * camera_floor                     # [B,1,H/16,W/16]
        # LiDAR/Radar remain validity-aware by construction, but valid cells get a
        # floor so sparse modalities do not collapse to near-zero everywhere.
        range_floor_lid = self.range_reliability_floor * valid_lid
        radar_density = torch.nn.functional.avg_pool2d(valid_rad.float(), kernel_size=5, stride=1, padding=2)
        range_floor_rad = valid_rad * (
            self.radar_reliability_floor * (0.5 + 0.5 * radar_density)
        )
        R_lid = range_floor_lid + (1.0 - self.range_reliability_floor) * torch.sigmoid(R_lid_raw) * valid_lid
        R_rad = range_floor_rad + (1.0 - self.radar_reliability_floor) * torch.sigmoid(R_rad_raw) * valid_rad

        invalid_penalty = torch.cat([
            self._invalid_penalty(R_cam, support_cam),
            self._invalid_penalty(R_lid, valid_lid),
            self._invalid_penalty(R_rad, valid_rad),
        ], dim=1)                                                           # [B,3]

        summary_in = torch.cat([
            e_joint,
            e_align_summary,
            self._stats(R_cam),
            self._stats(R_lid),
            self._stats(R_rad),
            self._valid_stats(R_cam, support_cam),
            self._valid_stats(R_lid, valid_lid),
            self._valid_stats(R_rad, valid_rad),
            invalid_penalty,
            support_cam.mean(dim=(2, 3)),
            valid_lid.mean(dim=(2, 3)),
            valid_rad.mean(dim=(2, 3)),
        ], dim=1)                                                           # [B,274]
        r_summary = self.summary_mlp(summary_in)                            # [B,64]
        return {
            "R_cam": R_cam,
            "R_lid": R_lid,
            "R_rad": R_rad,
            "r_summary": r_summary,
            "invalid_penalty": invalid_penalty,
        }
