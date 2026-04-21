import torch
from torch import nn

from quaternion_distances import quaternion_distance
from utils import quaternion_from_matrix


class TriModalPairwiseLoss(nn.Module):
    """
    Minimal tri-modal pairwise loss:
      L = L_CL + L_CR + L_LR
      L_pair = w_t * SmoothL1(t_pred, t_gt) + w_q * quat_distance(q_pred, q_gt)
    """

    def __init__(self, w_t=1.0, w_q=1.0, lambda_loop=0.0):
        super().__init__()
        self.w_t = float(w_t)
        self.w_q = float(w_q)
        self.lambda_loop = float(lambda_loop)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    @staticmethod
    def _flatten_pose_tensor(tensor):
        if tensor.ndim <= 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])

    def _pair_loss(self, t_pred, q_pred, t_gt, q_gt):
        t_pred = self._flatten_pose_tensor(t_pred)
        q_pred = self._flatten_pose_tensor(q_pred)
        t_gt = self._flatten_pose_tensor(t_gt)
        q_gt = self._flatten_pose_tensor(q_gt)
        l_t = self.transl_loss(t_pred, t_gt).sum(1).mean()
        l_q = quaternion_distance(q_pred, q_gt, q_pred.device).mean()
        return self.w_t * l_t + self.w_q * l_q, l_t, l_q

    @staticmethod
    def _flatten_matrix_tensor(tensor):
        if tensor.ndim <= 3:
            return tensor
        return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])

    @staticmethod
    def _pose_to_matrix(t_vec, q_vec):
        t_vec = TriModalPairwiseLoss._flatten_pose_tensor(t_vec)
        q_vec = TriModalPairwiseLoss._flatten_pose_tensor(q_vec)
        T = torch.eye(4, device=t_vec.device, dtype=t_vec.dtype).unsqueeze(0).repeat(t_vec.shape[0], 1, 1)
        T[:, :3, :3] = TriModalPairwiseLoss._quat_to_rotmat(q_vec)
        T[:, :3, 3] = t_vec
        return T

    @staticmethod
    def _matrix_to_pose(T):
        T = TriModalPairwiseLoss._flatten_matrix_tensor(T)
        t = T[:, :3, 3]
        q = quaternion_from_matrix(T)
        return t, q

    def _delta_target(self, input_T, gt_t, gt_q):
        input_T = self._flatten_matrix_tensor(input_T)
        gt_T = self._pose_to_matrix(gt_t, gt_q)
        delta_T = torch.bmm(gt_T, torch.linalg.inv(input_T))
        return self._matrix_to_pose(delta_T)

    def _apply_delta(self, pred_t, pred_q, input_T):
        delta_T = self._pose_to_matrix(pred_t, pred_q)
        input_T = self._flatten_matrix_tensor(input_T)
        corrected_T = torch.bmm(delta_T, input_T)
        return self._matrix_to_pose(corrected_T)

    @staticmethod
    def _quat_multiply(q, r):
        """
        Hamilton product for batched quaternions [w, x, y, z].
        q, r: [B, 4]
        """
        w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=1)

    @staticmethod
    def _quat_inv(q):
        # For unit quaternions from the model head, inverse is conjugate.
        qi = q.clone()
        qi[:, 1:] = -qi[:, 1:]
        return qi

    @staticmethod
    def _quat_to_rotmat(q):
        """
        Convert normalized quaternions [B,4] to rotation matrices [B,3,3].
        """
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        two = 2.0
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        r00 = ww + xx - yy - zz
        r01 = two * (xy - wz)
        r02 = two * (xz + wy)
        r10 = two * (xy + wz)
        r11 = ww - xx + yy - zz
        r12 = two * (yz - wx)
        r20 = two * (xz - wy)
        r21 = two * (yz + wx)
        r22 = ww - xx - yy + zz

        return torch.stack([
            torch.stack([r00, r01, r02], dim=1),
            torch.stack([r10, r11, r12], dim=1),
            torch.stack([r20, r21, r22], dim=1),
        ], dim=1)

    def _compose(self, t_ab, q_ab, t_bc, q_bc):
        """
        Compose T_ab * T_bc = T_ac.
        t_*: [B,3], q_*: [B,4]
        """
        r_ab = self._quat_to_rotmat(q_ab)
        t_ac = t_ab + torch.bmm(r_ab, t_bc.unsqueeze(-1)).squeeze(-1)
        q_ac = self._quat_multiply(q_ab, q_bc)
        q_ac = q_ac / q_ac.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return t_ac, q_ac

    def _invert(self, t_ab, q_ab):
        """
        Invert T_ab -> T_ba.
        """
        q_ba = self._quat_inv(q_ab)
        r_ba = self._quat_to_rotmat(q_ba)
        t_ba = -torch.bmm(r_ba, t_ab.unsqueeze(-1)).squeeze(-1)
        return t_ba, q_ba

    def _loop_loss(self, pred, batch):
        """
        Enforce tri-modal loop consistency among corrected T_CL, T_CR, T_LR.
        """
        T_cl_input = batch['T_CL_input']
        T_cr_input = batch['T_CR_input']
        T_lr_input = batch.get('T_LR_input')
        if T_lr_input is None:
            T_lr_input = torch.bmm(
                torch.linalg.inv(self._flatten_matrix_tensor(T_cl_input)),
                self._flatten_matrix_tensor(T_cr_input),
            )

        t_cl, q_cl = self._apply_delta(pred['T_CL_t'], pred['T_CL_q'], T_cl_input)
        t_cr, q_cr = self._apply_delta(pred['T_CR_t'], pred['T_CR_q'], T_cr_input)
        t_lr, q_lr = self._apply_delta(pred['T_LR_t'], pred['T_LR_q'], T_lr_input)

        # Forward closure: C->L->R should match C->R
        t_cr_hat, q_cr_hat = self._compose(t_cl, q_cl, t_lr, q_lr)
        l_fwd_t = self.transl_loss(t_cr_hat, t_cr).sum(1).mean()
        l_fwd_q = quaternion_distance(q_cr_hat, q_cr, q_cr.device).mean()

        # Backward closure: C->R->L should match C->L
        t_rl, q_rl = self._invert(t_lr, q_lr)
        t_cl_hat, q_cl_hat = self._compose(t_cr, q_cr, t_rl, q_rl)
        l_bwd_t = self.transl_loss(t_cl_hat, t_cl).sum(1).mean()
        l_bwd_q = quaternion_distance(q_cl_hat, q_cl, q_cl.device).mean()

        l_loop_t = 0.5 * (l_fwd_t + l_bwd_t)
        l_loop_q = 0.5 * (l_fwd_q + l_bwd_q)
        l_loop = self.w_t * l_loop_t + self.w_q * l_loop_q
        return l_loop, l_loop_t, l_loop_q

    def forward(self, pred, batch):
        delta_cl_t_gt, delta_cl_q_gt = self._delta_target(batch['T_CL_input'], batch['T_CL_t_gt'], batch['T_CL_q_gt'])
        delta_cr_t_gt, delta_cr_q_gt = self._delta_target(batch['T_CR_input'], batch['T_CR_t_gt'], batch['T_CR_q_gt'])

        T_lr_input = batch.get('T_LR_input')
        if T_lr_input is None:
            T_lr_input = torch.bmm(
                torch.linalg.inv(self._flatten_matrix_tensor(batch['T_CL_input'])),
                self._flatten_matrix_tensor(batch['T_CR_input']),
            )
        delta_lr_t_gt, delta_lr_q_gt = self._delta_target(T_lr_input, batch['T_LR_t_gt'], batch['T_LR_q_gt'])

        l_cl, l_cl_t, l_cl_q = self._pair_loss(pred['T_CL_t'], pred['T_CL_q'], delta_cl_t_gt, delta_cl_q_gt)
        l_cr, l_cr_t, l_cr_q = self._pair_loss(pred['T_CR_t'], pred['T_CR_q'], delta_cr_t_gt, delta_cr_q_gt)
        l_lr, l_lr_t, l_lr_q = self._pair_loss(pred['T_LR_t'], pred['T_LR_q'], delta_lr_t_gt, delta_lr_q_gt)
        l_pairwise = l_cl + l_cr + l_lr
        l_loop, l_loop_t, l_loop_q = self._loop_loss(pred, batch)
        total = l_pairwise + self.lambda_loop * l_loop
        return {
            'total_loss': total,
            'loss_pairwise': l_pairwise,
            'loss_loop': l_loop,
            'loss_loop_t': l_loop_t,
            'loss_loop_q': l_loop_q,
            'loss_cl': l_cl,
            'loss_cr': l_cr,
            'loss_lr': l_lr,
            'loss_cl_t': l_cl_t,
            'loss_cl_q': l_cl_q,
            'loss_cr_t': l_cr_t,
            'loss_cr_q': l_cr_q,
            'loss_lr_t': l_lr_t,
            'loss_lr_q': l_lr_q,
        }
