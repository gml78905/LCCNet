import torch
from torch import nn

from quaternion_distances import quaternion_distance


class TriModalPairwiseLoss(nn.Module):
    """
    Minimal tri-modal pairwise loss:
      L = L_CL + L_CR + L_LR
      L_pair = w_t * SmoothL1(t_pred, t_gt) + w_q * quat_distance(q_pred, q_gt)
    """

    def __init__(self, w_t=1.0, w_q=1.0):
        super().__init__()
        self.w_t = float(w_t)
        self.w_q = float(w_q)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def _pair_loss(self, t_pred, q_pred, t_gt, q_gt):
        l_t = self.transl_loss(t_pred, t_gt).sum(1).mean()
        l_q = quaternion_distance(q_pred, q_gt, q_pred.device).mean()
        return self.w_t * l_t + self.w_q * l_q, l_t, l_q

    def forward(self, pred, batch):
        l_cl, l_cl_t, l_cl_q = self._pair_loss(pred['T_CL_t'], pred['T_CL_q'], batch['T_CL_t_gt'], batch['T_CL_q_gt'])
        l_cr, l_cr_t, l_cr_q = self._pair_loss(pred['T_CR_t'], pred['T_CR_q'], batch['T_CR_t_gt'], batch['T_CR_q_gt'])
        l_lr, l_lr_t, l_lr_q = self._pair_loss(pred['T_LR_t'], pred['T_LR_q'], batch['T_LR_t_gt'], batch['T_LR_q_gt'])
        total = l_cl + l_cr + l_lr
        return {
            'total_loss': total,
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

