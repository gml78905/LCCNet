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


class CalibrationConditionedReliability(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            BasicBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 6, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        rel = torch.sigmoid(self.net(x))
        return {
            "rel_cam": rel[:, 0:1],
            "rel_lid": rel[:, 1:2],
            "rel_rad": rel[:, 2:3],
            "rel_cl": rel[:, 3:4],
            "rel_cr": rel[:, 4:5],
            "rel_lr": rel[:, 5:6],
        }


class ReliabilityWeightedRefiner(nn.Module):
    def __init__(self, feat_channels: int = 256):
        super().__init__()
        self.shared_refine = nn.Sequential(
            nn.Conv2d(feat_channels + 6, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            BasicBlock(feat_channels, feat_channels),
        )
        self.pair_cl = nn.Sequential(
            nn.Conv2d(feat_channels + 1 + 1 + 2 + 2, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            BasicBlock(feat_channels, feat_channels),
        )
        self.pair_cr = nn.Sequential(
            nn.Conv2d(feat_channels + 1 + 1 + 2 + 2, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            BasicBlock(feat_channels, feat_channels),
        )
        self.pair_lr = nn.Sequential(
            nn.Conv2d(feat_channels + 1 + 1 + 1 + 4 + 4, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            BasicBlock(feat_channels, feat_channels),
        )

    def forward(
        self,
        coarse: torch.Tensor,
        lidar_mask: torch.Tensor,
        radar_mask: torch.Tensor,
        match_cl: torch.Tensor,
        match_cr: torch.Tensor,
        match_lr: torch.Tensor,
        reliability: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        rel_cam = reliability["rel_cam"]
        rel_lid = reliability["rel_lid"]
        rel_rad = reliability["rel_rad"]
        rel_cl = reliability["rel_cl"]
        rel_cr = reliability["rel_cr"]
        rel_lr = reliability["rel_lr"]

        shared = self.shared_refine(
            torch.cat([
                coarse,
                rel_cam,
                rel_lid,
                rel_rad,
                rel_cl,
                rel_cr,
                rel_lr,
            ], dim=1)
        )
        pair_cl = self.pair_cl(torch.cat([
            coarse * rel_cl,
            rel_cl,
            lidar_mask,
            match_cl,
            match_cl * rel_lid,
        ], dim=1))
        pair_cr = self.pair_cr(torch.cat([
            coarse * rel_cr,
            rel_cr,
            radar_mask,
            match_cr,
            match_cr * rel_rad,
        ], dim=1))
        pair_lr = self.pair_lr(torch.cat([
            coarse * rel_lr,
            rel_lr,
            lidar_mask * rel_lid,
            radar_mask * rel_rad,
            match_lr,
            match_lr * rel_lr,
        ], dim=1))
        return {
            "shared": shared,
            "pair_cl": pair_cl,
            "pair_cr": pair_cr,
            "pair_lr": pair_lr,
        }


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
        reliability_in_channels = 256 + 128 + 1 + 1 + 2 + 2 + 4
        self.reliability = CalibrationConditionedReliability(reliability_in_channels, hidden_channels=128)
        self.reliability_memory = ConvGRUCell(input_dim=6, hidden_dim=32, kernel_size=3)
        self.reliability_out = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, kernel_size=1, bias=True),
        )
        self.refiner = ReliabilityWeightedRefiner(feat_channels=256)
        self.memory = ConvGRUCell(input_dim=256, hidden_dim=256, kernel_size=3)
        self.readout_context_window = 5
        self.adapter_cl = PairwiseReadoutAdapter(in_channels=256 + 256 + 2 + 2 + 1 + 1, out_channels=256)
        self.adapter_cr = PairwiseReadoutAdapter(in_channels=256 + 256 + 2 + 2 + 1 + 1, out_channels=256)
        self.adapter_lr = PairwiseReadoutAdapter(in_channels=256 + 256 + 4 + 4 + 1 + 1 + 1, out_channels=256)
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

    def _readout(
        self,
        h: torch.Tensor,
        lidar_mask: torch.Tensor,
        radar_mask: torch.Tensor,
        pair_cl: torch.Tensor,
        pair_cr: torch.Tensor,
        pair_lr: torch.Tensor,
        match_cl: torch.Tensor,
        match_cr: torch.Tensor,
        match_lr: torch.Tensor,
        reliability: Dict[str, torch.Tensor],
    ):
        lid_context = self._context_mask(lidar_mask)
        rad_context = self._context_mask(radar_mask)

        h_cl = self.adapter_cl(torch.cat([
            h,
            pair_cl,
            lidar_mask,
            lid_context,
            match_cl,
            reliability["rel_cl"],
            reliability["rel_lid"],
        ], dim=1))
        h_cr = self.adapter_cr(torch.cat([
            h,
            pair_cr,
            radar_mask,
            rad_context,
            match_cr,
            reliability["rel_cr"],
            reliability["rel_rad"],
        ], dim=1))
        h_lr = self.adapter_lr(torch.cat([
            h,
            pair_lr,
            lidar_mask,
            lid_context,
            radar_mask,
            rad_context,
            match_lr,
            reliability["rel_lr"],
            reliability["rel_lid"],
            reliability["rel_rad"],
        ], dim=1))
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
            "match_cl": match_cl,
            "match_cr": match_cr,
            "match_lr": match_lr,
            "pair_cl": pair_cl,
            "pair_cr": pair_cr,
            "pair_lr": pair_lr,
            **reliability,
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
        raw_reliability = self.reliability(torch.cat([
            evidence["J_t"],
            evidence["query"],
            lid["mask"],
            rad["mask"],
            evidence["match_cl"],
            evidence["match_cr"],
            evidence["match_lr"],
        ], dim=1))
        rel_input = torch.cat([
            raw_reliability["rel_cam"],
            raw_reliability["rel_lid"],
            raw_reliability["rel_rad"],
            raw_reliability["rel_cl"],
            raw_reliability["rel_cr"],
            raw_reliability["rel_lr"],
        ], dim=1)
        prev_rel_h = None if state is None else state.get("rel_h")
        rel_h = self.reliability_memory(rel_input, prev_rel_h)
        rel_maps = torch.sigmoid(self.reliability_out(rel_h))
        reliability = {
            "rel_cam": rel_maps[:, 0:1],
            "rel_lid": rel_maps[:, 1:2],
            "rel_rad": rel_maps[:, 2:3],
            "rel_cl": rel_maps[:, 3:4],
            "rel_cr": rel_maps[:, 4:5],
            "rel_lr": rel_maps[:, 5:6],
        }
        refined = self.refiner(
            evidence["J_t"],
            lid["mask"],
            rad["mask"],
            evidence["match_cl"],
            evidence["match_cr"],
            evidence["match_lr"],
            reliability,
        )
        prev_h = None if state is None else state.get("h")
        h = self.memory(refined["shared"], prev_h)
        new_state = {
            "h": h,
            "lidar_mask": lid["mask"],
            "radar_mask": rad["mask"],
            "match_cl": evidence["match_cl"],
            "match_cr": evidence["match_cr"],
            "match_lr": evidence["match_lr"],
            "pair_cl": refined["pair_cl"],
            "pair_cr": refined["pair_cr"],
            "pair_lr": refined["pair_lr"],
            "rel_h": rel_h,
            **reliability,
        }
        if not return_aux:
            return new_state
        aux = {
            "J_t": evidence["J_t"],
            "J_refined": refined["shared"],
            "query": evidence["query"],
            "attended": evidence["attended"],
            "lidar_mask": lid["mask"],
            "radar_mask": rad["mask"],
            "h": h,
            "match_cl": evidence["match_cl"],
            "match_cr": evidence["match_cr"],
            "match_lr": evidence["match_lr"],
            "pair_cl": refined["pair_cl"],
            "pair_cr": refined["pair_cr"],
            "pair_lr": refined["pair_lr"],
            "raw_rel_cam": raw_reliability["rel_cam"],
            "raw_rel_lid": raw_reliability["rel_lid"],
            "raw_rel_rad": raw_reliability["rel_rad"],
            "raw_rel_cl": raw_reliability["rel_cl"],
            "raw_rel_cr": raw_reliability["rel_cr"],
            "raw_rel_lr": raw_reliability["rel_lr"],
            **reliability,
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
            pred, readout_aux = self._readout(
                cur_state["h"],
                cur_state["lidar_mask"],
                cur_state["radar_mask"],
                cur_state["pair_cl"],
                cur_state["pair_cr"],
                cur_state["pair_lr"],
                cur_state["match_cl"],
                cur_state["match_cr"],
                cur_state["match_lr"],
                cur_state,
            )
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
        pred, readout_aux = self._readout(
            h_t,
            cur_state["lidar_mask"],
            cur_state["radar_mask"],
            cur_state["pair_cl"],
            cur_state["pair_cr"],
            cur_state["pair_lr"],
            cur_state["match_cl"],
            cur_state["match_cr"],
            cur_state["match_lr"],
            cur_state,
        )
        if not return_aux:
            return pred, cur_state
        aux = {} if last_aux is None else dict(last_aux)
        aux["h_T"] = h_t
        aux.update(readout_aux)
        return pred, cur_state, aux
