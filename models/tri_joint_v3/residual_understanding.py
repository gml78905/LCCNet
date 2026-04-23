from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointResidualUnderstanding(nn.Module):
    """
    Build a denser alignment residual field after joint feature construction.

    Inputs:
      cam_s16, lid_s16, rad_s16: [B,256,H/16,W/16]
      F_joint_map:               [B,384,H/16,W/16]
      support_cam, valid_lid, valid_rad: [B,1,H/16,W/16] or None

    Outputs:
      E_joint_map:         [B,192,H/16,W/16]
      e_joint:             [B,192]
      e_align_summary:     [B,64]

    The residual field now combines:
      - feature disagreement
      - support mismatch
      - local soft alignment cue
      - lightweight range residual cue
    """

    def __init__(self, feat_ch: int = 256, d_joint: int = 384, d_res: int = 192, align_radius: int = 1):
        super().__init__()
        self.align_radius = align_radius
        self.disagree_proj = nn.Sequential(
            nn.Conv2d(feat_ch * 3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.support_proj = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.align_proj = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.geom_proj = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(d_joint + 128 + 32 + 64 + 32, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, d_res, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_res),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(d_res + 15, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        # Sparse LiDAR/Radar supports are dilated for local matching only.
        # The original validity maps are still used by reliability masking.
        pad = kernel_size // 2
        return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)

    def _local_soft_alignment(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # src/ref: [B,C,H,W]
        b, c, h, w = src.shape
        r = self.align_radius
        k = 2 * r + 1
        src_n = F.normalize(src, dim=1)
        ref_n = F.normalize(ref, dim=1)
        ref_patch = F.unfold(ref_n, kernel_size=k, padding=r)            # [B,C*k*k,HW]
        ref_patch = ref_patch.view(b, c, k * k, h * w)                   # [B,C,K,HW]
        src_flat = src_n.flatten(2).unsqueeze(2)                         # [B,C,1,HW]
        sim = (src_flat * ref_patch).sum(dim=1)                          # [B,K,HW]
        prob = torch.softmax(sim, dim=1)                                 # [B,K,HW]

        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offsets.append((float(dx), float(dy)))
        offset_xy = src.new_tensor(offsets)                              # [K,2]
        disp = (prob.unsqueeze(-1) * offset_xy.view(1, k * k, 1, 2)).sum(dim=1)   # [B,HW,2]
        disp = disp.permute(0, 2, 1).contiguous().view(b, 2, h, w)       # [B,2,H,W]
        best_sim = sim.max(dim=1)[0].view(b, 1, h, w)                    # [B,1,H,W]
        exp_sim = (prob * sim).sum(dim=1, keepdim=False).view(b, 1, h, w)# [B,1,H,W]
        disp_norm = torch.norm(disp, dim=1, keepdim=True)                # [B,1,H,W]
        return torch.cat([best_sim, exp_sim, disp_norm, disp], dim=1)    # [B,5,H,W]

    def _local_soft_alignment_masked(
        self,
        src: torch.Tensor,
        ref: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> torch.Tensor:
        # src/ref: [B,C,H,W], pair_valid: [B,1,H,W]
        b, c, h, w = src.shape
        r = self.align_radius
        k = 2 * r + 1
        src_n = F.normalize(src, dim=1)
        ref_n = F.normalize(ref, dim=1)
        ref_patch = F.unfold(ref_n, kernel_size=k, padding=r)            # [B,C*K,HW]
        ref_patch = ref_patch.view(b, c, k * k, h * w)                   # [B,C,K,HW]
        src_flat = src_n.flatten(2).unsqueeze(2)                         # [B,C,1,HW]
        sim = (src_flat * ref_patch).sum(dim=1)                          # [B,K,HW]

        valid_patch = F.unfold(pair_valid.float(), kernel_size=k, padding=r)  # [B,K,HW]
        center_valid = pair_valid.flatten(2)                                  # [B,1,HW]
        joint_valid = (valid_patch > 0.05) & (center_valid > 0.05)            # [B,K,HW]

        sim = sim.masked_fill(~joint_valid, -1e4)
        prob = torch.softmax(sim, dim=1)
        prob = prob * joint_valid.float()
        prob = prob / prob.sum(dim=1, keepdim=True).clamp(min=1e-6)

        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offsets.append((float(dx), float(dy)))
        offset_xy = src.new_tensor(offsets)                                   # [K,2]
        disp = (prob.unsqueeze(-1) * offset_xy.view(1, k * k, 1, 2)).sum(dim=1)
        disp = disp.permute(0, 2, 1).contiguous().view(b, 2, h, w)            # [B,2,H,W]

        best_sim = sim.max(dim=1)[0].view(b, 1, h, w)
        center_mask = center_valid.view(b, 1, h, w).clamp(0.0, 1.0)
        best_sim = torch.where(center_mask > 0.05, best_sim, torch.zeros_like(best_sim))
        exp_sim = (prob * torch.where(joint_valid, sim, torch.zeros_like(sim))).sum(dim=1, keepdim=False).view(b, 1, h, w)
        disp_norm = torch.norm(disp, dim=1, keepdim=True)
        out = torch.cat([best_sim, exp_sim, disp_norm, disp], dim=1)
        return out * center_mask

    def _range_residual(self, valid_lid: torch.Tensor, valid_rad: torch.Tensor) -> torch.Tensor:
        # Placeholder geometric residual: support asymmetry and overlap.
        overlap = valid_lid * valid_rad                                  # [B,1,H,W]
        asym = torch.abs(valid_lid - valid_rad)                          # [B,1,H,W]
        return torch.cat([overlap, asym], dim=1)                         # [B,2,H,W]

    def forward(
        self,
        cam_s16: torch.Tensor,
        lid_s16: torch.Tensor,
        rad_s16: torch.Tensor,
        F_joint_map: torch.Tensor,
        support_cam: Optional[torch.Tensor] = None,
        valid_lid: Optional[torch.Tensor] = None,
        valid_rad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if support_cam is None:
            support_cam = torch.ones(
                (cam_s16.shape[0], 1, cam_s16.shape[2], cam_s16.shape[3]),
                device=cam_s16.device,
                dtype=cam_s16.dtype,
            )
        if valid_lid is None:
            valid_lid = torch.ones_like(support_cam)
        if valid_rad is None:
            valid_rad = torch.ones_like(support_cam)

        feat_disagree = self.disagree_proj(
            torch.cat([
                torch.abs(cam_s16 - lid_s16),
                torch.abs(cam_s16 - rad_s16),
                torch.abs(lid_s16 - rad_s16),
            ], dim=1)
        )                                                                  # [B,128,H/16,W/16]

        support_mismatch = self.support_proj(
            torch.cat([
                torch.abs(support_cam - valid_lid),
                torch.abs(support_cam - valid_rad),
                torch.abs(valid_lid - valid_rad),
            ], dim=1)
        )                                                                  # [B,32,H/16,W/16]

        # Local soft alignment compares a cell to a small neighborhood rather than
        # only exact cell matches. This is the direct alignment-aware addition in v3.1-lite.
        support_cam_align = 0.2 + 0.8 * support_cam                         # [B,1,H/16,W/16]
        valid_lid_align = self._dilate_mask(valid_lid, kernel_size=5)        # [B,1,H/16,W/16]
        valid_rad_align = self._dilate_mask(valid_rad, kernel_size=5)        # [B,1,H/16,W/16]
        valid_cl = support_cam_align * valid_lid_align                      # [B,1,H/16,W/16]
        valid_cr = support_cam_align * valid_rad_align                      # [B,1,H/16,W/16]
        valid_lr = valid_lid_align * valid_rad_align                        # [B,1,H/16,W/16]
        align_cl = self._local_soft_alignment_masked(cam_s16, lid_s16, valid_cl)       # [B,5,H/16,W/16]
        align_cr = self._local_soft_alignment_masked(cam_s16, rad_s16, valid_cr)       # [B,5,H/16,W/16]
        align_lr_disp = self._local_soft_alignment_masked(lid_s16, rad_s16, valid_lr)  # [B,5,H/16,W/16]
        local_align_input = torch.cat([
            align_cl,                                                       # CL: best/exp/disp_norm/dx/dy
            align_cr,                                                       # CR: best/exp/disp_norm/dx/dy
            align_lr_disp,                                                  # LR: best/exp/disp_norm/dx/dy
        ], dim=1)                                                           # [B,15,H/16,W/16]
        local_align = self.align_proj(local_align_input)                    # [B,64,H/16,W/16]

        geom_residual = self.geom_proj(
            self._range_residual(valid_lid, valid_rad)
        )                                                                   # [B,32,H/16,W/16]

        E_joint_map = self.fuse(
            torch.cat([F_joint_map, feat_disagree, support_mismatch, local_align, geom_residual], dim=1)
        )                                                                   # [B,192,H/16,W/16]
        e_joint = self.pool(E_joint_map).flatten(1)                         # [B,192]

        align_stats = torch.cat([
            align_cl.mean(dim=(2, 3)),
            align_cr.mean(dim=(2, 3)),
            align_lr_disp.mean(dim=(2, 3)),
        ], dim=1)                                                           # [B,15]
        e_align_summary = self.summary_mlp(torch.cat([e_joint, align_stats], dim=1))  # [B,64]

        return {
            "E_joint_map": E_joint_map,
            "e_joint": e_joint,
            "e_align_summary": e_align_summary,
            "feat_disagree": feat_disagree,
            "support_mismatch": support_mismatch,
            "local_align": local_align,
            "align_cl": align_cl,
            "align_cr": align_cr,
            "align_lr": align_lr_disp,
            "support_cam_align": support_cam_align,
            "valid_lid_align": valid_lid_align,
            "valid_rad_align": valid_rad_align,
            "geom_residual": geom_residual,
        }
