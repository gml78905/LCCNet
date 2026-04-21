from typing import Dict, Tuple

import torch
import torch.nn as nn


class CoarseContextBuilder(nn.Module):
    """
    Build tri-modal coarse shared context from s32 features.

    Inputs:
      cam_s32, lid_s32, rad_s32: [B, 512, h, w]
    Outputs:
      z_shared_coarse: [B, 256]
      z_cam, z_lid, z_rad: [B, 256]
      p_cam, p_lid, p_rad: [B, 512]   # pooled s32 prior tokens
    """

    def __init__(self, s32_channels_each: int = 512, d_shared: int = 256, d_mod: int = 256):
        super().__init__()
        in_ch = s32_channels_each * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, d_shared, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_shared, d_shared, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mod_proj = nn.Linear(s32_channels_each, d_mod)

    def _pool_token(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).flatten(1)

    def forward(
        self,
        cam_s32: torch.Tensor,
        lid_s32: torch.Tensor,
        rad_s32: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([cam_s32, lid_s32, rad_s32], dim=1)
        ctx = self.fuse(x)
        z_shared_coarse = self._pool_token(ctx)              # [B, 256]

        p_cam = self._pool_token(cam_s32)                    # [B, 512]
        p_lid = self._pool_token(lid_s32)                    # [B, 512]
        p_rad = self._pool_token(rad_s32)                    # [B, 512]

        z_cam = self.mod_proj(p_cam)                         # [B, 256]
        z_lid = self.mod_proj(p_lid)                         # [B, 256]
        z_rad = self.mod_proj(p_rad)                         # [B, 256]

        return {
            "z_shared_coarse": z_shared_coarse,
            "z_cam": z_cam,
            "z_lid": z_lid,
            "z_rad": z_rad,
            "p_cam": p_cam,
            "p_lid": p_lid,
            "p_rad": p_rad,
        }


class PairRelationBuilder(nn.Module):
    """
    Build coarse pair relation states using:
      - s16 main relation features
      - s32 pooled prior/supplement
      - shared coarse context

    Outputs:
      r_cl_coarse, r_cr_coarse, r_lr_coarse: [B, 384]
    """

    def __init__(
        self,
        s16_channels_each: int = 256,
        s32_prior_dim: int = 512,
        d_shared: int = 256,
        d_pair: int = 384,
        s16_token_dim: int = 256,
    ):
        super().__init__()
        # TODO(v2): optionally replace pooled s16 tokens with lightweight spatial relation extraction.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.s16_proj = nn.Linear(s16_channels_each, s16_token_dim)
        self.pair_head_cl = self._make_pair_head(s16_token_dim, s32_prior_dim, d_shared, d_pair)
        self.pair_head_cr = self._make_pair_head(s16_token_dim, s32_prior_dim, d_shared, d_pair)
        self.pair_head_lr = self._make_pair_head(s16_token_dim, s32_prior_dim, d_shared, d_pair)

    @staticmethod
    def _make_pair_head(
        s16_token_dim: int,
        s32_prior_dim: int,
        d_shared: int,
        d_pair: int,
    ) -> nn.Module:
        # Main relation from s16 + prior from s32 pooled tokens + shared coarse token.
        in_dim = (s16_token_dim * 2) + (s32_prior_dim * 2) + d_shared
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, d_pair),
        )

    def _pool_s16(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).flatten(1)

    def forward(
        self,
        cam_s16: torch.Tensor,
        lid_s16: torch.Tensor,
        rad_s16: torch.Tensor,
        p_cam: torch.Tensor,
        p_lid: torch.Tensor,
        p_rad: torch.Tensor,
        z_shared_coarse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c16 = self.s16_proj(self._pool_s16(cam_s16))   # [B, 256]
        l16 = self.s16_proj(self._pool_s16(lid_s16))   # [B, 256]
        r16 = self.s16_proj(self._pool_s16(rad_s16))   # [B, 256]

        feat_cl = torch.cat([c16, l16, p_cam, p_lid, z_shared_coarse], dim=1)
        feat_cr = torch.cat([c16, r16, p_cam, p_rad, z_shared_coarse], dim=1)
        feat_lr = torch.cat([l16, r16, p_lid, p_rad, z_shared_coarse], dim=1)

        r_cl = self.pair_head_cl(feat_cl)              # [B, 384]
        r_cr = self.pair_head_cr(feat_cr)              # [B, 384]
        r_lr = self.pair_head_lr(feat_lr)              # [B, 384]
        return r_cl, r_cr, r_lr


class SharedCoarseContextV2(nn.Module):
    """
    V2 shared coarse context builder.

    Inputs:
      cam_s32, lid_s32, rad_s32: [B, 512, H/32, W/32]
      target_hw_s16: (H/16, W/16)

    Outputs:
      z_shared: [B, 256]
      ctx_s16:  [B, 256, H/16, W/16]  # conditioning map for dense reliability heads
      p_cam, p_lid, p_rad: [B, 512]   # pooled s32 priors
    """

    def __init__(self, s32_channels_each: int = 512, d_shared: int = 256):
        super().__init__()
        in_ch = s32_channels_each * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, d_shared, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_shared, d_shared, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d_shared),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _pool_token(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x).flatten(1)

    def forward(
        self,
        cam_s32: torch.Tensor,
        lid_s32: torch.Tensor,
        rad_s32: torch.Tensor,
        target_hw_s16: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([cam_s32, lid_s32, rad_s32], dim=1)
        ctx_s32 = self.fuse(x)
        z_shared = self._pool_token(ctx_s32)  # [B, 256]
        ctx_s16 = torch.nn.functional.interpolate(
            ctx_s32, size=target_hw_s16, mode="bilinear", align_corners=False
        )
        return {
            "z_shared": z_shared,
            "ctx_s16": ctx_s16,
            "p_cam": self._pool_token(cam_s32),
            "p_lid": self._pool_token(lid_s32),
            "p_rad": self._pool_token(rad_s32),
        }
