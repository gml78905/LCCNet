from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .calibration_core import ResidualRefinementCore
from .coarse_context import SharedCoarseContextV2
from .encoders import CameraEncoderMS, LidarEncoderMS, RadarEncoderMS
from .evidence_aggregation import ReliabilityAwarePairAggregator
from .heads import TriPairDeltaPoseHeads
from .memory import TriTemporalMemory
from .reliability_maps import DenseModalityReliabilityEstimator


class TriModalJointCalibNetV2(nn.Module):
    """
    v2 architecture main path:
      encoders -> shared coarse context -> dense reliability maps
      -> reliability-aware pair evidence aggregation -> temporal memory
      -> pair refinement -> delta pose

    Notes:
      - Returns delta pose only.
      - Corrected pose must be computed outside this model (loss/util side).
      - Device-safe: no hard-coded .cuda().
      - Keeps current input contract (rgb/lidar_proj/radar_proj).
    """

    def __init__(
        self,
        camera_pretrained: bool = False,
        activation: str = "leakyrelu",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.camera_encoder = CameraEncoderMS(pretrained=camera_pretrained)
        self.lidar_encoder = LidarEncoderMS(activation=activation)
        self.radar_encoder = RadarEncoderMS(activation=activation)

        self.shared_context = SharedCoarseContextV2(s32_channels_each=512, d_shared=256)
        self.reli_maps = DenseModalityReliabilityEstimator(feat_ch=256, ctx_ch=256)
        self.agg = ReliabilityAwarePairAggregator(feat_ch=256, prior_dim=512, d_shared=256, d_pair=384)

        self.memory = TriTemporalMemory(d_shared=256, d_pair_in=384, d_pair_hidden=256)
        self.refine = ResidualRefinementCore(d_pair=384, d_pair_mem=256, d_shared=256)
        self.pose_heads = TriPairDeltaPoseHeads(d_pair=384, hidden_dim=head_hidden_dim, dropout=head_dropout)

    def _forward_step(
        self,
        rgb: torch.Tensor,
        lidar_proj: torch.Tensor,
        radar_proj: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        cam = self.camera_encoder(rgb)
        lid = self.lidar_encoder(lidar_proj)
        rad = self.radar_encoder(radar_proj)

        target_hw = (cam["s16"].shape[2], cam["s16"].shape[3])
        coarse = self.shared_context(cam["s32"], lid["s32"], rad["s32"], target_hw_s16=target_hw)

        reli = self.reli_maps(
            cam_s16=cam["s16"],
            lid_s16=lid["s16"],
            rad_s16=rad["s16"],
            ctx_s16=coarse["ctx_s16"],
        )

        agg = self.agg(
            cam_s16=cam["s16"],
            lid_s16=lid["s16"],
            rad_s16=rad["s16"],
            r_cam=reli["R_cam"],
            r_lid=reli["R_lid"],
            r_rad=reli["R_rad"],
            p_cam=coarse["p_cam"],
            p_lid=coarse["p_lid"],
            p_rad=coarse["p_rad"],
            z_shared=coarse["z_shared"],
        )

        h_shared, h_cl, h_cr, h_lr, new_state = self.memory(
            coarse["z_shared"], agg["r_cl0"], agg["r_cr0"], agg["r_lr0"], state=state
        )

        refined = self.refine(
            r_cl_coarse=agg["r_cl0"],
            r_cr_coarse=agg["r_cr0"],
            r_lr_coarse=agg["r_lr0"],
            h_cl=h_cl,
            h_cr=h_cr,
            h_lr=h_lr,
            h_shared=h_shared,
            w_c=reli["w_c"],
            w_l=reli["w_l"],
            w_r=reli["w_r"],
            w_cl=agg["w_cl"],
            w_cr=agg["w_cr"],
            w_lr=agg["w_lr"],
        )

        pred = self.pose_heads(refined["r_cl_ref"], refined["r_cr_ref"], refined["r_lr_ref"])
        if not return_aux:
            return pred, new_state

        aux = {
            "z_shared": coarse["z_shared"],
            "ctx_s16": coarse["ctx_s16"],
            "p_cam": coarse["p_cam"],
            "p_lid": coarse["p_lid"],
            "p_rad": coarse["p_rad"],
            "R_cam": reli["R_cam"],
            "R_lid": reli["R_lid"],
            "R_rad": reli["R_rad"],
            "w_c": reli["w_c"],
            "w_l": reli["w_l"],
            "w_r": reli["w_r"],
            "F_cam_w": agg["F_cam_w"],
            "F_lid_w": agg["F_lid_w"],
            "F_rad_w": agg["F_rad_w"],
            "r_cl0": agg["r_cl0"],
            "r_cr0": agg["r_cr0"],
            "r_lr0": agg["r_lr0"],
            "w_cl": agg["w_cl"],
            "w_cr": agg["w_cr"],
            "w_lr": agg["w_lr"],
            "h_shared": h_shared,
            "h_cl": h_cl,
            "h_cr": h_cr,
            "h_lr": h_lr,
            "delta_r_cl": refined["delta_r_cl"],
            "delta_r_cr": refined["delta_r_cr"],
            "delta_r_lr": refined["delta_r_lr"],
            "gate_cl": refined["gate_cl"],
            "gate_cr": refined["gate_cr"],
            "gate_lr": refined["gate_lr"],
            "r_cl_ref": refined["r_cl_ref"],
            "r_cr_ref": refined["r_cr_ref"],
            "r_lr_ref": refined["r_lr_ref"],
        }
        return pred, new_state, aux

    @staticmethod
    def _stack_time(step_list: Dict[str, list]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack(v, dim=1) for k, v in step_list.items()}

    def forward(
        self,
        rgb: torch.Tensor,
        lidar_proj: torch.Tensor,
        radar_proj: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        if rgb.ndim == 4:
            return self._forward_step(
                rgb=rgb,
                lidar_proj=lidar_proj,
                radar_proj=radar_proj,
                state=state,
                return_aux=return_aux,
            )

        if rgb.ndim != 5:
            raise ValueError(f"Expected rgb 4D or 5D, got shape={tuple(rgb.shape)}")

        t_len = rgb.shape[1]
        pred_steps = {}
        aux_steps = {}
        cur_state = state
        for t in range(t_len):
            if return_aux:
                pred_t, cur_state, aux_t = self._forward_step(
                    rgb[:, t], lidar_proj[:, t], radar_proj[:, t], state=cur_state, return_aux=True
                )
                for k, v in aux_t.items():
                    aux_steps.setdefault(k, []).append(v)
            else:
                pred_t, cur_state = self._forward_step(
                    rgb[:, t], lidar_proj[:, t], radar_proj[:, t], state=cur_state, return_aux=False
                )
            for k, v in pred_t.items():
                pred_steps.setdefault(k, []).append(v)

        pred = self._stack_time(pred_steps)
        if not return_aux:
            return pred, cur_state
        aux = self._stack_time(aux_steps)
        return pred, cur_state, aux
