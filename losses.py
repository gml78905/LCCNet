# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/losses.pyy

import torch
from torch import nn as nn

from quaternion_distances import quaternion_distance
from utils import quat2mat, rotate_back, rotate_forward, tvector2mat, quaternion_from_matrix


class GeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sx = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.Tensor([-3.0]), requires_grad=True)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = torch.exp(-self.sx) * loss_transl + self.sx
        total_loss += torch.exp(-self.sq) * loss_rot + self.sq
        return total_loss


class ProposedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(ProposedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.losses = {}

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean() * 100
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        self.losses['total_loss'] = total_loss
        self.losses['transl_loss'] = loss_transl
        self.losses['rot_loss'] = loss_rot
        return self.losses


class L1Loss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(L1Loss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss


class DistancePoints3D(nn.Module):
    def __init__(self):
        super(DistancePoints3D, self).__init__()

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The mean distance between 3D points
        """
        #start = time.time()
        total_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()
            if point_cloud_gt.numel() == 0 or point_cloud_gt.shape[-1] == 0:
                continue

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            total_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)

        return total_loss/target_transl.shape[0]


# The combination of L1 loss of translation part,
# quaternion angle loss of rotation part,
# distance loss of the pointclouds
class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        #start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()
            if point_cloud_gt.numel() == 0 or point_cloud_gt.shape[-1] == 0:
                continue

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)
        total_loss = (1 - self.weight_point_cloud) * pose_loss +\
                     self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0])

        if not torch.isfinite(loss_transl):
            print("Debug: loss_transl is not finite")
            print(f"  target_transl finite: {torch.isfinite(target_transl).all().item()}")
            print(f"  transl_err finite: {torch.isfinite(transl_err).all().item()}")
            print(f"  transl_err min/max: {transl_err.min().item()} / {transl_err.max().item()}")
        if not torch.isfinite(loss_rot):
            print("Debug: loss_rot is not finite")
            print(f"  target_rot finite: {torch.isfinite(target_rot).all().item()}")
            print(f"  rot_err finite: {torch.isfinite(rot_err).all().item()}")
            print(f"  target_rot norm mean: {torch.norm(target_rot, dim=1).mean().item()}")
            print(f"  rot_err norm mean: {torch.norm(rot_err, dim=1).mean().item()}")
        if not torch.isfinite(point_clouds_loss):
            print("Debug: point_clouds_loss is not finite")
            print(f"  point_cloud batch size: {len(point_clouds)}")
            print(f"  first point_cloud finite: {torch.isfinite(point_clouds[0]).all().item() if len(point_clouds) > 0 else 'n/a'}")
        if not torch.isfinite(total_loss):
            print("Debug: total_loss is not finite")
            print(f"  loss_transl: {loss_transl.item() if torch.is_tensor(loss_transl) else loss_transl}")
            print(f"  loss_rot: {loss_rot.item() if torch.is_tensor(loss_rot) else loss_rot}")
            print(f"  point_clouds_loss: {(point_clouds_loss/target_transl.shape[0]).item()}")
            print(f"  pose_loss: {pose_loss.item() if torch.is_tensor(pose_loss) else pose_loss}")

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]

        return self.loss
