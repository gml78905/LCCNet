from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.tri_calib.blocks import BasicBlock

from .encoders import CameraEncoderMS, LidarEncoderV5, RadarEncoderV5
from .evidence import EarlyTriModalEvidence
from .heads import PairwiseDeltaHeads
from .memory import ConvGRUCell


class PairwiseReadoutAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriModalJointCalibNetV5(nn.Module):
    def __init__(
        self,
        camera_pretrained: bool = False,
        activation: str = "relu",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        del activation, head_hidden_dim, head_dropout
        self.camera_encoder = CameraEncoderMS(pretrained=camera_pretrained)
        self.lidar_encoder = LidarEncoderV5()
        self.radar_encoder = RadarEncoderV5()
        self.evidence = EarlyTriModalEvidence(feat_ch=256, proj_ch=128, window_size=5)
        self.memory = ConvGRUCell(input_dim=256, hidden_dim=256, kernel_size=3)
        self.readout_context_window = 5
        self.adapter_cl = PairwiseReadoutAdapter(in_channels=256 + 2, out_channels=256)
        self.adapter_cr = PairwiseReadoutAdapter(in_channels=256 + 2, out_channels=256)
        self.adapter_lr = PairwiseReadoutAdapter(in_channels=256 + 4, out_channels=256)
        self.heads = PairwiseDeltaHeads(in_dim=512)

    def _context_mask(self, mask: torch.Tensor) -> torch.Tensor:
        radius = self.readout_context_window // 2
        return torch.nn.functional.max_pool2d(
            mask,
            kernel_size=self.readout_context_window,
            stride=1,
            padding=radius,
        )

    @staticmethod
    def _global_pool(h: torch.Tensor) -> torch.Tensor:
        avg = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        return torch.cat([avg, mx], dim=1)

    def _readout(self, h: torch.Tensor, lidar_mask: torch.Tensor, radar_mask: torch.Tensor):
        lid_context = self._context_mask(lidar_mask)
        rad_context = self._context_mask(radar_mask)

        h_cl = self.adapter_cl(torch.cat([h, lidar_mask, lid_context], dim=1))
        h_cr = self.adapter_cr(torch.cat([h, radar_mask, rad_context], dim=1))
        h_lr = self.adapter_lr(torch.cat([h, lidar_mask, lid_context, radar_mask, rad_context], dim=1))
        z_cl = self._global_pool(h_cl)
        z_cr = self._global_pool(h_cr)
        z_lr = self._global_pool(h_lr)
        pred = self.heads(z_cl, z_cr, z_lr)
        return pred, {
            "h_cl": h_cl,
            "h_cr": h_cr,
            "h_lr": h_lr,
            "z_cl": z_cl,
            "z_cr": z_cr,
            "z_lr": z_lr,
            "lidar_context": lid_context,
            "radar_context": rad_context,
        }

    def _forward_step(
        self,
        rgb: torch.Tensor,
        lidar_proj: torch.Tensor,
        radar_proj: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        cam_feat = self.camera_encoder(rgb)
        lid = self.lidar_encoder(lidar_proj)
        rad = self.radar_encoder(radar_proj)
        evidence = self.evidence(cam_feat, lid["feat"], rad["feat"], lid["mask"], rad["mask"])
        prev_h = None if state is None else state.get("h")
        h = self.memory(evidence["J_t"], prev_h)
        new_state = {
            "h": h,
            "lidar_mask": lid["mask"],
            "radar_mask": rad["mask"],
        }
        if not return_aux:
            return new_state
        aux = {
            "J_t": evidence["J_t"],
            "query": evidence["query"],
            "attended": evidence["attended"],
            "lidar_mask": lid["mask"],
            "radar_mask": rad["mask"],
            "h": h,
        }
        return new_state, aux

    def forward(
        self,
        rgb: torch.Tensor,
        lidar_proj: torch.Tensor,
        radar_proj: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
    ):
        if rgb.ndim == 4:
            step_out = self._forward_step(rgb, lidar_proj, radar_proj, state=state, return_aux=return_aux)
            if return_aux:
                cur_state, aux = step_out
            else:
                cur_state = step_out
                aux = None
            pred, readout_aux = self._readout(cur_state["h"], cur_state["lidar_mask"], cur_state["radar_mask"])
            if not return_aux:
                return pred, cur_state
            aux = {} if aux is None else dict(aux)
            aux["h_T"] = cur_state["h"]
            aux.update(readout_aux)
            return pred, cur_state, aux
        if rgb.ndim != 5:
            raise ValueError(f"Expected rgb to be 4D or 5D, got shape={tuple(rgb.shape)}")

        cur_state = state
        last_aux: Optional[Dict[str, torch.Tensor]] = None
        for t in range(rgb.shape[1]):
            if return_aux:
                cur_state, aux_t = self._forward_step(
                    rgb[:, t], lidar_proj[:, t], radar_proj[:, t], state=cur_state, return_aux=True
                )
                last_aux = aux_t
            else:
                cur_state = self._forward_step(rgb[:, t], lidar_proj[:, t], radar_proj[:, t], state=cur_state, return_aux=False)

        h_t = cur_state["h"]
        pred, readout_aux = self._readout(h_t, cur_state["lidar_mask"], cur_state["radar_mask"])
        if not return_aux:
            return pred, cur_state
        aux = {} if last_aux is None else dict(last_aux)
        aux["h_T"] = h_t
        aux.update(readout_aux)
        return pred, cur_state, aux
