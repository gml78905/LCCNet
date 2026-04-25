from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coarse_context import SharedCoarseContextV4
from .decoder import JointCalibrationDecoder
from .encoders import CameraEncoderMS, SparseAwareLidarEncoderMS, RadarUncertaintyEncoderMS
from .fusion import ImportanceAwareJointFusion
from .heads import PairwiseDenseDeltaReadout
from .importance import LearnedImportanceEstimator
from .joint_field import DenseJointFieldBuilder
from .memory import DenseJointTemporalMemory


class TriModalJointCalibNetV4(nn.Module):
    """
    Dense tri-sensor joint calibration model.

    Main path:
      encoders
      -> shared coarse context
      -> dense joint field
      -> learned importance field
      -> importance-aware joint fusion
      -> dense temporal memory
      -> joint calibration decoder
      -> pairwise dense delta readout
    """

    def __init__(
        self,
        camera_pretrained: bool = False,
        activation: str = "leakyrelu",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        del activation, head_dropout
        self.camera_encoder = CameraEncoderMS(pretrained=camera_pretrained)
        self.lidar_encoder = SparseAwareLidarEncoderMS()
        self.radar_encoder = RadarUncertaintyEncoderMS()
        self.shared_context = SharedCoarseContextV4(s32_channels_each=512, d_shared=256)
        self.joint_field = DenseJointFieldBuilder(feat_ch=256, d_proj=128, d_joint=384)
        self.importance = LearnedImportanceEstimator(d_joint=384, d_summary=64)
        self.fusion = ImportanceAwareJointFusion(feat_ch=128, d_joint=384, d_summary=64)
        self.memory = DenseJointTemporalMemory(d_joint=384, d_hidden=256)
        self.decoder = JointCalibrationDecoder(d_joint=384, d_mem=256)
        self.heads = PairwiseDenseDeltaReadout(
            d_joint=384,
            d_global=384,
            hidden_dim=head_hidden_dim,
            max_delta_t=0.25,
            max_fine_t=0.05,
            max_delta_r_deg=5.0,
        )

    @staticmethod
    def _support_cam(rgb: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        gray = rgb.mean(dim=1, keepdim=True)
        kernel_x = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        kernel_y = gray.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, kernel_x, padding=1)
        grad_y = F.conv2d(gray, kernel_y, padding=1)
        support = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        support = F.interpolate(support, size=target_hw, mode="bilinear", align_corners=False)
        denom = support.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return support / denom

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
        support_cam = self._support_cam(rgb, target_hw)
        support_lid = lid["support"]
        support_rad = rad["support"]

        joint = self.joint_field(
            cam_s16=cam["s16"],
            lid_s16=lid["s16"],
            rad_s16=rad["s16"],
            ctx_s16=coarse["ctx_s16"],
            support_cam=support_cam,
            support_lid=support_lid,
            support_rad=support_rad,
        )
        importance = self.importance(
            J0=joint["J0"],
            support_cam=support_cam,
            support_lid=support_lid,
            support_rad=support_rad,
            diff_cl=joint["diff_cl"],
            diff_cr=joint["diff_cr"],
            diff_lr=joint["diff_lr"],
        )
        fused = self.fusion(
            J0=joint["J0"],
            cam_feat=joint["cam_proj"],
            lid_feat=joint["lid_proj"],
            rad_feat=joint["rad_proj"],
            W_cam=importance["W_cam"],
            W_lid=importance["W_lid"],
            W_rad=importance["W_rad"],
            W_joint=importance["W_joint"],
        )
        m_joint, new_state, z_mem = self.memory(fused["J1"], state=state)
        decoded = self.decoder(
            J1=fused["J1"],
            m_joint=m_joint,
            W_cam=importance["W_cam"],
            W_lid=importance["W_lid"],
            W_rad=importance["W_rad"],
            W_joint=importance["W_joint"],
        )
        pred = self.heads(decoded["J_ref"], decoded["z_global"])
        pred_out = {k: v for k, v in pred.items() if not k.startswith("A_")}
        if not return_aux:
            return pred_out, new_state
        aux = {
            "ctx_s16": coarse["ctx_s16"],
            "z_shared": coarse["z_shared"],
            "support_cam": support_cam,
            "support_lid": support_lid,
            "support_rad": support_rad,
            "valid_lid": lid["valid"],
            "valid_rad": rad["valid"],
            "density_lid": lid["density"],
            "density_rad": rad["density"],
            "J0": joint["J0"],
            "z_joint0": joint["z_joint0"],
            "cam_proj": joint["cam_proj"],
            "lid_proj": joint["lid_proj"],
            "rad_proj": joint["rad_proj"],
            "support_branch": joint["support_branch"],
            "agreement_branch": joint["agreement_branch"],
            "conflict_branch": joint["conflict_branch"],
            "directional_branch": joint["directional_branch"],
            "diff_cl": joint["diff_cl"],
            "diff_cr": joint["diff_cr"],
            "diff_lr": joint["diff_lr"],
            "agree_maps": joint["agree_maps"],
            "W_cam": importance["W_cam"],
            "W_lid": importance["W_lid"],
            "W_rad": importance["W_rad"],
            "W_joint": importance["W_joint"],
            "z_importance": importance["z_importance"],
            "diff_energy": importance["diff_energy"],
            "importance_trunk": importance["importance_trunk"],
            "J1": fused["J1"],
            "z_joint": fused["z_joint"],
            "z_cam": fused["z_cam"],
            "z_lid": fused["z_lid"],
            "z_rad": fused["z_rad"],
            "z_summary": fused["z_summary"],
            "cam_w": fused["cam_w"],
            "lid_w": fused["lid_w"],
            "rad_w": fused["rad_w"],
            "m_joint": m_joint,
            "z_mem": z_mem,
            "J_ref": decoded["J_ref"],
            "z_global": decoded["z_global"],
            "A_CL": pred["A_CL"],
            "A_CR": pred["A_CR"],
            "A_LR": pred["A_LR"],
        }
        return pred_out, new_state, aux

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
            return self._forward_step(rgb, lidar_proj, radar_proj, state=state, return_aux=return_aux)
        if rgb.ndim != 5:
            raise ValueError(f"Expected rgb 4D or 5D, got shape={tuple(rgb.shape)}")
        t_len = rgb.shape[1]
        pred_steps: Dict[str, list] = {}
        aux_steps: Dict[str, list] = {}
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
