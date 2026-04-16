import torch
import torch.nn as nn

from .context import CoarseContextFusion
from .encoders import CameraResnetEncoder, LidarEncoder, RadarEncoder
from .heads import TriPairwiseHeads


class TriModalCalibNet(nn.Module):
    """
    Minimal first working version:
      - camera encoder
      - lidar encoder
      - radar encoder
      - coarse context fusion
      - pairwise pose heads (CL, CR, LR)

    No cost volume, no reliability, no recurrent memory.
    """

    def __init__(self, camera_pretrained=False, activation="leakyrelu", head_hidden_dim=256, dropout=0.0,
                 debug_shapes=False):
        super().__init__()
        self.camera_encoder = CameraResnetEncoder(pretrained=camera_pretrained)
        self.lidar_encoder = LidarEncoder(activation=activation)
        self.radar_encoder = RadarEncoder(activation=activation)

        self.context = CoarseContextFusion(in_channels_each=512, out_channels=256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Pair feature dimension:
        # [za, zb, za-zb, za*zb, z_ctx] -> 512 + 512 + 512 + 512 + 256 = 2304
        self.pair_feature_dim = 2304
        self.heads = TriPairwiseHeads(
            pair_feature_dim=self.pair_feature_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        self.debug_shapes = bool(debug_shapes)
        self._debug_printed = False

        # Placeholder: reliability module will be inserted here later.
        # self.reliability = CalibrationConditionedReliability(...)
        #
        # Placeholder: recurrent memory core will be inserted here later.
        # self.recurrent_core = TriModalCalibrationCore(...)

    @staticmethod
    def _global_token(feat):
        return nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)

    @staticmethod
    def _pair_feature(za, zb, zctx):
        return torch.cat([za, zb, za - zb, za * zb, zctx], dim=1)

    def forward(self, rgb, lidar_proj, radar_proj):
        """
        Args:
            rgb:        [B, 3, 288, 512]
            lidar_proj: [B, 1, 288, 512]
            radar_proj: [B, 2, 288, 512]

        Returns:
            dict with:
              T_CL_t, T_CL_q, T_CR_t, T_CR_q, T_LR_t, T_LR_q
        """
        cam_feats = self.camera_encoder(rgb)
        lidar_feats = self.lidar_encoder(lidar_proj)
        radar_feats = self.radar_encoder(radar_proj)

        cam_s32 = cam_feats["s32"]
        lidar_s32 = lidar_feats["s32"]
        radar_s32 = radar_feats["s32"]

        _, z_ctx = self.context(cam_s32, lidar_s32, radar_s32)   # [B, 256]

        z_c = self._global_token(cam_s32)                         # [B, 512]
        z_l = self._global_token(lidar_s32)                       # [B, 512]
        z_r = self._global_token(radar_s32)                       # [B, 512]

        feat_cl = self._pair_feature(z_c, z_l, z_ctx)            # [B, 2304]
        feat_cr = self._pair_feature(z_c, z_r, z_ctx)            # [B, 2304]
        feat_lr = self._pair_feature(z_l, z_r, z_ctx)            # [B, 2304]

        out = self.heads(feat_cl, feat_cr, feat_lr)
        if self.debug_shapes and not self._debug_printed:
            print(f"[TriModalCalibNet] cam_s32={tuple(cam_s32.shape)} lidar_s32={tuple(lidar_s32.shape)} radar_s32={tuple(radar_s32.shape)}")
            print(f"[TriModalCalibNet] feat_cl={tuple(feat_cl.shape)} feat_cr={tuple(feat_cr.shape)} feat_lr={tuple(feat_lr.shape)}")
            print(f"[TriModalCalibNet] out T_CL_t={tuple(out['T_CL_t'].shape)} T_CL_q={tuple(out['T_CL_q'].shape)}")
            print(f"[TriModalCalibNet] out T_CR_t={tuple(out['T_CR_t'].shape)} T_CR_q={tuple(out['T_CR_q'].shape)}")
            print(f"[TriModalCalibNet] out T_LR_t={tuple(out['T_LR_t'].shape)} T_LR_q={tuple(out['T_LR_q'].shape)}")
            self._debug_printed = True
        return out
