from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coarse_context import SharedCoarseContextV3
from .encoders import CameraEncoderMS, LidarEncoderMS, RadarEncoderMS
from .heads import JointToPairDeltaPoseHeads
from .joint_feature import JointAlignmentFeatureBuilder
from .joint_fusion import JointEvidenceFusion
from .memory import JointTemporalMemory
from .refinement import JointResidualRefinement
from .reliability import ResidualAwareReliabilityEstimator
from .residual_understanding import JointResidualUnderstanding


class TriModalJointCalibNetV3Lite(nn.Module):
    """
    v3.1-lite main path:
      encoders
      -> shared coarse context
      -> joint alignment feature construction
      -> dense alignment residual field
      -> residual-aware reliability
      -> alignment-preserving joint fusion
      -> map-aware joint temporal memory
      -> dense-aware joint refinement
      -> pairwise delta readout

    Notes:
      - internal main path is joint, not pairwise-centered
      - output remains pairwise delta pose
      - corrected pose stays outside the model (loss/util side)
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

        self.shared_context = SharedCoarseContextV3(s32_channels_each=512, d_shared=256)
        self.joint_feature = JointAlignmentFeatureBuilder(feat_ch=256, ctx_ch=256, d_common=128, d_joint=384)
        self.residual = JointResidualUnderstanding(feat_ch=256, d_joint=384, d_res=192)
        self.reliability = ResidualAwareReliabilityEstimator(d_joint=384, d_res=192, d_summary=64)
        self.joint_fusion = JointEvidenceFusion(d_joint=384, d_res=192, d_fused=384)
        self.memory = JointTemporalMemory(d_in=384, d_hidden=256, d_map_summary=64)
        self.refine = JointResidualRefinement(d_joint=384, d_mem=256, d_summary=64)
        self.pose_heads = JointToPairDeltaPoseHeads(
            d_joint=384,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            max_delta_t=0.25,
            max_fine_t=0.05,
            max_delta_r_deg=5.0,
        )

    @staticmethod
    def _support_cam(rgb: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # Dense camera support prior from local structure rather than raw intensity.
        gray = rgb.mean(dim=1, keepdim=True)                               # [B,1,H,W]
        kernel_x = gray.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = gray.new_tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, kernel_x, padding=1)
        grad_y = F.conv2d(gray, kernel_y, padding=1)
        support = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)    # [B,1,H,W]
        support = F.interpolate(support, size=target_hw, mode="bilinear", align_corners=False)
        denom = support.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return support / denom                                             # [B,1,H/16,W/16]

    @staticmethod
    def _valid_from_proj(proj: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # Use input projection support as a validity prior.
        valid = (proj > 0).float()                                         # [B,1,H,W] or [B,2,H,W]
        if valid.shape[1] > 1:
            valid = valid.amax(dim=1, keepdim=True)
        return F.interpolate(valid, size=target_hw, mode="nearest")        # [B,1,H/16,W/16]

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
        joint = self.joint_feature(
            cam_s16=cam["s16"],
            lid_s16=lid["s16"],
            rad_s16=rad["s16"],
            ctx_s16=coarse["ctx_s16"],
        )

        support_cam = self._support_cam(rgb, target_hw=target_hw)          # [B,1,H/16,W/16]
        valid_lid = self._valid_from_proj(lidar_proj, target_hw=target_hw) # [B,1,H/16,W/16]
        valid_rad = self._valid_from_proj(radar_proj[:, :1], target_hw=target_hw)

        residual = self.residual(
            cam_s16=cam["s16"],
            lid_s16=lid["s16"],
            rad_s16=rad["s16"],
            F_joint_map=joint["F_joint_map"],
            support_cam=support_cam,
            valid_lid=valid_lid,
            valid_rad=valid_rad,
        )
        reli = self.reliability(
            F_joint_map=joint["F_joint_map"],
            E_joint_map=residual["E_joint_map"],
            support_cam=support_cam,
            valid_lid=valid_lid,
            valid_rad=valid_rad,
            e_joint=residual["e_joint"],
            e_align_summary=residual["e_align_summary"],
        )
        fused = self.joint_fusion(
            F_joint_map=joint["F_joint_map"],
            E_joint_map=residual["E_joint_map"],
            R_cam=reli["R_cam"],
            R_lid=reli["R_lid"],
            R_rad=reli["R_rad"],
            valid_lid=valid_lid,
            valid_rad=valid_rad,
        )
        h_joint, new_state = self.memory(
            z_joint_fused=fused["z_joint_fused"],
            z_joint_support_weighted=fused["z_joint_support_weighted"],
            z_joint_residual_weighted=fused["z_joint_residual_weighted"],
            z_lid_valid=fused["z_lid_valid"],
            z_rad_valid=fused["z_rad_valid"],
            fusion_map_summary=fused["fusion_map_summary"],
            state=state,
        )
        refined = self.refine(
            z_joint_fused=fused["z_joint_fused"],
            h_joint=h_joint,
            r_summary=reli["r_summary"],
            e_align_summary=residual["e_align_summary"],
            z_joint_support_weighted=fused["z_joint_support_weighted"],
            z_joint_residual_weighted=fused["z_joint_residual_weighted"],
            z_lid_valid=fused["z_lid_valid"],
            z_rad_valid=fused["z_rad_valid"],
        )
        pred = self.pose_heads(refined["z_joint_ref"])
        if not return_aux:
            return pred, new_state

        aux = {
            "ctx_s16": coarse["ctx_s16"],
            "z_shared": coarse["z_shared"],
            "F_joint_map": joint["F_joint_map"],
            "z_joint": joint["z_joint"],
            "z_joint_support": joint["z_joint_support"],
            "z_joint_conflict": joint["z_joint_conflict"],
            "z_joint_align": joint["z_joint_align"],
            "support_branch": joint["support_branch"],
            "common_structure": joint["common_structure"],
            "conflict_branch": joint["conflict_branch"],
            "align_precursor": joint["align_precursor"],
            "diff_cl": joint["diff_cl"],
            "diff_cr": joint["diff_cr"],
            "diff_lr": joint["diff_lr"],
            "E_joint_map": residual["E_joint_map"],
            "e_joint": residual["e_joint"],
            "e_align_summary": residual["e_align_summary"],
            "feat_disagree": residual["feat_disagree"],
            "support_mismatch": residual["support_mismatch"],
            "local_align": residual["local_align"],
            "align_cl": residual["align_cl"],
            "align_cr": residual["align_cr"],
            "align_lr": residual["align_lr"],
            "support_cam_align": residual["support_cam_align"],
            "valid_lid_align": residual["valid_lid_align"],
            "valid_rad_align": residual["valid_rad_align"],
            "geom_residual": residual["geom_residual"],
            "support_cam": support_cam,
            "valid_lid": valid_lid,
            "valid_rad": valid_rad,
            "R_cam": reli["R_cam"],
            "R_lid": reli["R_lid"],
            "R_rad": reli["R_rad"],
            "r_summary": reli["r_summary"],
            "invalid_penalty": reli["invalid_penalty"],
            "F_joint_fused": fused["F_joint_fused"],
            "z_joint_fused": fused["z_joint_fused"],
            "z_joint_support_weighted": fused["z_joint_support_weighted"],
            "z_joint_residual_weighted": fused["z_joint_residual_weighted"],
            "z_lid_valid": fused["z_lid_valid"],
            "z_rad_valid": fused["z_rad_valid"],
            "fusion_map_summary": fused["fusion_map_summary"],
            "support_strength": fused["support_strength"],
            "residual_strength": fused["residual_strength"],
            "valid_lid_w": fused["valid_lid_w"],
            "valid_rad_w": fused["valid_rad_w"],
            "h_joint": h_joint,
            "delta_z": refined["delta_z"],
            "gate_z": refined["gate_z"],
            "dense_summary": refined["dense_summary"],
            "summary_bias": refined["summary_bias"],
            "z_joint_ref": refined["z_joint_ref"],
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
