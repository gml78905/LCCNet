# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/utils.py

import math

try:
    import mathutils
except ImportError:
    mathutils = None
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from torch.utils.data.dataloader import default_collate


def rotate_points(PC, R, T=None, inverse=True):
    if mathutils is None:
        raise ImportError("mathutils is required for rotate_points but is not installed.")
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T*R
    else:
        RT=R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    if mathutils is None:
        raise ImportError("mathutils is required for invert_pose but is not installed.")
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def merge_inputs(queries):
    # Generic path for datasets that already return fully collatable tensors.
    if 'point_cloud' not in queries[0]:
        non_collatable_keys = {'lidar_pc', 'radar_pc', 'lidar_pc_seq', 'radar_pc_seq'}
        collated = {}
        for key in queries[0]:
            if key in non_collatable_keys:
                collated[key] = [d[key] for d in queries]
            else:
                collated[key] = default_collate([d[key] for d in queries])
        return collated

    point_clouds = []
    imgs = []
    reflectances = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb' and key != 'reflectance'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    return returns


def project_pointcloud_to_image_torch(pc_sensor, T_cam_sensor, calib, image_hw, max_depth):
    if pc_sensor.numel() == 0:
        h, w = image_hw
        depth = torch.zeros((h, w), device=T_cam_sensor.device, dtype=T_cam_sensor.dtype)
        aux_map = torch.zeros((h, w), device=T_cam_sensor.device, dtype=T_cam_sensor.dtype)
        return depth, aux_map

    xyz = pc_sensor[:, :3]
    ones = torch.ones((xyz.shape[0], 1), device=pc_sensor.device, dtype=pc_sensor.dtype)
    xyz1 = torch.cat([xyz, ones], dim=1)
    pc_cam = torch.matmul(xyz1, T_cam_sensor.t())

    x = pc_cam[:, 0]
    y = pc_cam[:, 1]
    z = pc_cam[:, 2]
    aux = pc_sensor[:, 3] if pc_sensor.shape[1] > 3 else torch.zeros_like(z)
    h, w = int(image_hw[0]), int(image_hw[1])

    valid = (z > 1e-5) & (z < max_depth)
    if not torch.any(valid):
        depth = torch.zeros((h, w), device=pc_sensor.device, dtype=pc_sensor.dtype)
        aux_map = torch.zeros((h, w), device=pc_sensor.device, dtype=pc_sensor.dtype)
        return depth, aux_map

    x = x[valid]
    y = y[valid]
    z = z[valid]
    aux = aux[valid]

    u = torch.round((calib[0, 0] * x / z) + calib[0, 2]).long()
    v = torch.round((calib[1, 1] * y / z) + calib[1, 2]).long()
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not torch.any(in_img):
        depth = torch.zeros((h, w), device=pc_sensor.device, dtype=pc_sensor.dtype)
        aux_map = torch.zeros((h, w), device=pc_sensor.device, dtype=pc_sensor.dtype)
        return depth, aux_map

    u = u[in_img]
    v = v[in_img]
    z = z[in_img]
    aux = aux[in_img]

    depth = torch.zeros((h, w), device=pc_sensor.device, dtype=pc_sensor.dtype)
    aux_map = torch.zeros((h, w), device=pc_sensor.device, dtype=pc_sensor.dtype)
    depth[v, u] = z
    aux_map[v, u] = aux
    depth = depth / max_depth
    return depth, aux_map


def quaternion_from_matrix(matrix):
    """
    Convert rotation matrix/matrices to quaternion(s).
    Args:
        matrix (torch.Tensor): [4,4], [3,3], [B,4,4], or [B,3,3]

    Returns:
        torch.Tensor: [4] for single input, or [B,4] for batched input
    """
    if matrix.ndim not in (2, 3):
        raise TypeError("Not a valid rotation matrix")

    batched = matrix.ndim == 3
    if batched:
        if matrix.shape[-2:] == (4, 4):
            R = matrix[:, :3, :3]
        elif matrix.shape[-2:] == (3, 3):
            R = matrix
        else:
            raise TypeError("Not a valid rotation matrix")
    else:
        if matrix.shape == (4, 4):
            R = matrix[:3, :3].unsqueeze(0)
        elif matrix.shape == (3, 3):
            R = matrix.unsqueeze(0)
        else:
            raise TypeError("Not a valid rotation matrix")

    q = torch.zeros((R.shape[0], 4), device=R.device, dtype=R.dtype)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    mask1 = tr > 0.0
    if torch.any(mask1):
        S = torch.sqrt(tr[mask1] + 1.0) * 2.0
        q[mask1, 0] = 0.25 * S
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S

    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if torch.any(mask2):
        S = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S
        q[mask2, 1] = 0.25 * S
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S

    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if torch.any(mask3):
        S = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S
        q[mask3, 2] = 0.25 * S
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S

    mask4 = (~mask1) & (~mask2) & (~mask3)
    if torch.any(mask4):
        S = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S
        q[mask4, 3] = 0.25 * S

    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return q if batched else q[0]


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    t_norm = t.norm()
    if t_norm < 1e-12:
        identity = torch.zeros(4, device=q.device, dtype=q.dtype)
        identity[0] = 1.
        return identity
    return t / t_norm


def quat2mat(q):
    """
    Convert quaternion(s) to homogeneous rotation matrix/matrices.
    Args:
        q (torch.Tensor): [4] or [B,4]

    Returns:
        torch.Tensor: [4,4] or [B,4,4]
    """
    if q.ndim not in (1, 2):
        raise AssertionError("Not a valid quaternion")
    batched = q.ndim == 2
    if not batched and q.shape != torch.Size([4]):
        raise AssertionError("Not a valid quaternion")
    if batched and q.shape[1] != 4:
        raise AssertionError("Not a valid quaternion")

    q_in = q if batched else q.unsqueeze(0)
    q_norm = q_in.norm(dim=1, keepdim=True)
    q_safe = q_in / q_norm.clamp(min=1e-12)

    w, x, y, z = q_safe[:, 0], q_safe[:, 1], q_safe[:, 2], q_safe[:, 3]
    mats = torch.zeros((q_safe.shape[0], 4, 4), device=q_safe.device, dtype=q_safe.dtype)
    mats[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    mats[:, 0, 1] = 2 * x * y - 2 * z * w
    mats[:, 0, 2] = 2 * x * z + 2 * y * w
    mats[:, 1, 0] = 2 * x * y + 2 * z * w
    mats[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    mats[:, 1, 2] = 2 * y * z - 2 * x * w
    mats[:, 2, 0] = 2 * x * z - 2 * y * w
    mats[:, 2, 1] = 2 * y * z + 2 * x * w
    mats[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    mats[:, 3, 3] = 1.0

    zero_norm_mask = (q_norm.squeeze(1) < 1e-12)
    if torch.any(zero_norm_mask):
        mats[zero_norm_mask] = torch.eye(4, device=q_safe.device, dtype=q_safe.dtype)
    return mats if batched else mats[0]


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    #io.imshow(blended_img)
    #io.show()
    #plt.figure()
    #plt.imshow(blended_img)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img
