import torch
import torch.nn as nn


class CoarseContextFusion(nn.Module):
    """
    Minimal coarse fusion module (no cost volume).
    It fuses top-scale features (s32) from camera/lidar/radar.
    """

    def __init__(self, in_channels_each=512, out_channels=256):
        super().__init__()
        in_channels = in_channels_each * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, cam_s32, lidar_s32, radar_s32):
        x = torch.cat([cam_s32, lidar_s32, radar_s32], dim=1)
        ctx_map = self.fuse(x)                                   # [B, C_ctx, H/32, W/32]
        ctx_token = self.pool(ctx_map).flatten(1)               # [B, C_ctx]
        return ctx_map, ctx_token

