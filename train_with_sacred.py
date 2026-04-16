# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/main_visibility_CALIB.py

import math
import os
import random
import time

# import apex
try:
    import mathutils
except ImportError:
    mathutils = None
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import ConcatDataset
import torch.nn as nn
import os.path as osp

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetLidarCamera import DatasetLidarCameraKittiOdometry, DatasetLidarCameraHercules
from DatasetCameraRadar import DatasetCameraRadarHercules
from DatasetLGInnotek import (
    DatasetLidarCameraLGInnotek,
    DatasetCameraRadarLGInnotek,
    DatasetTriModalLGInnotek,
    _load_point_cloud,
)
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
from losses_tri import TriModalPairwiseLoss
from models.LCCNet import LCCNet
from models.tri_calib.model import TriModalCalibNet

from quaternion_distances import quaternion_distance

from tensorboardX import SummaryWriter
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")

# Disable git info to avoid errors in Docker containers
# In Docker, git repository may not work properly, so disable it
save_git_info = False
ex = Experiment("LCCNet", save_git_info=save_git_info)
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = '/workspace/data/checkpoints/LCCNet/'
    dataset = 'hercules' # 'kitti/odom', 'kitti/raw', 'hercules', 'lg_innotek'
    sensor_mode = 'lidar'  # For Hercules/LG_Innotek: 'lidar', 'radar', 'both', or 'tri'
    data_folder = '/workspace/data/hercules'
    use_reflectance = False
    val_sequence = 0  # For KITTI
    val_scene = ['library_1']  # For Hercules (None = use first scene, list = use multiple scenes)
    train_scene = ['SC_1', 'SC_3', 'island_1']  # For Hercules (None = use all scenes except val_scene, list = use specific scenes for training)
    checkpoint_name = 'test'  # For Hercules: custom checkpoint name for saving (None = auto-generate from val_scene and sensor_mode)
    lg_train_scene = ['afternoon_parking_lot_1', 'afternoon_campus_1']
    lg_val_scene = ['afternoon_campus_2']
    lg_val_frame_limit = 3000
    epochs = 120
    BASE_LEARNING_RATE = 1e-4  # 1e-4
    loss = 'combined'
    max_t = 0.5 # 1.5, 1.0,  0.5,  0.2,  0.1
    max_r = 5.0 # 20.0, 10.0, 5.0,  2.0,  1.0
    batch_size = 120  # 120
    num_worker = 0
    network = 'Res_f1'
    optimizer = 'adam'
    resume = True
    weights = 'None'  # '/workspace/data/Checkpoint/LCCNet/kitti_iter5.tar'  # Set to None to start from scratch for Hercules
    rescale_rot = 1.0
    rescale_transl = 2.0
    precision = "O0"
    norm = 'bn'
    dropout = 0.0
    max_depth = 80.
    weight_point_cloud = 0.5
    log_frequency = 10
    print_frequency = 50
    starting_epoch = -1
    wandb_enabled = False
    wandb_project = 'LCCNet_TriModal'
    wandb_entity = 'LGIT_calib'
    wandb_name = 'TriBaseline'
    wandb_mode = 'online'  # 'online', 'offline', 'disabled'
    wandb_log_images = True
    debug_timing = False
    use_dataparallel = True
    tri_run_one_batch = True
    tri_debug_shapes = True
    tri_loss_w_t = 1.0
    tri_loss_w_q = 1.0


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv


def _resize_tensor_chw(tensor, size, mode="bilinear"):
    align_corners = False if mode in ["bilinear", "bicubic"] else None
    tensor = tensor.unsqueeze(0)
    if align_corners is None:
        tensor = F.interpolate(tensor, size=size, mode=mode)
    else:
        tensor = F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
    return tensor.squeeze(0)


def preprocess_projected_inputs(rgb, depth_img, depth_gt, img_shape, input_size):
    target_h, target_w = img_shape
    src_h, src_w = rgb.shape[1], rgb.shape[2]

    # Keep the full frame visible: shrink oversized inputs to fit the canvas
    # before padding, instead of relying on negative padding which crops the
    # right/bottom region and leaves only the top-left content.
    resize_scale = min(target_h / src_h, target_w / src_w, 1.0)
    if resize_scale < 1.0:
        resized_h = max(1, int(round(src_h * resize_scale)))
        resized_w = max(1, int(round(src_w * resize_scale)))
        resize_size = (resized_h, resized_w)
        rgb = _resize_tensor_chw(rgb, resize_size, mode="bilinear")
        depth_img = _resize_tensor_chw(depth_img, resize_size, mode="bilinear")
        depth_gt = _resize_tensor_chw(depth_gt, resize_size, mode="bilinear")

    shape_pad = [0, 0, 0, 0]
    shape_pad[3] = target_h - rgb.shape[1]
    shape_pad[1] = target_w - rgb.shape[2]

    rgb = F.pad(rgb, shape_pad)
    depth_img = F.pad(depth_img, shape_pad)
    depth_gt = F.pad(depth_gt, shape_pad)

    return rgb, depth_img, depth_gt, shape_pad


def fit_tensor_to_canvas(tensor, img_shape):
    target_h, target_w = img_shape
    src_h, src_w = tensor.shape[1], tensor.shape[2]

    resize_scale = min(target_h / src_h, target_w / src_w, 1.0)
    if resize_scale < 1.0:
        resized_h = max(1, int(round(src_h * resize_scale)))
        resized_w = max(1, int(round(src_w * resize_scale)))
        tensor = _resize_tensor_chw(tensor, (resized_h, resized_w), mode="bilinear")

    shape_pad = [0, 0, 0, 0]
    shape_pad[3] = target_h - tensor.shape[1]
    shape_pad[1] = target_w - tensor.shape[2]
    tensor = F.pad(tensor, shape_pad)
    return tensor


def resize_model_inputs(rgb_batch, lidar_batch, input_size):
    rgb_batch = F.interpolate(rgb_batch, size=input_size, mode="bilinear")
    lidar_batch = F.interpolate(lidar_batch, size=input_size, mode="bilinear")
    return rgb_batch, lidar_batch


def _tensor_to_wandb_image(tensor):
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu()
        if array.dim() == 3:
            array = array.permute(1, 2, 0).numpy()
        else:
            array = array.numpy()
    else:
        array = tensor
    return wandb.Image(array)


def _wandb_log(enabled, data, step=None, commit=None):
    if enabled and wandb is not None and wandb.run is not None:
        if commit is None:
            wandb.log(data)
        else:
            wandb.log(data, commit=commit)


# CCN training
@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss,
          debug_batch_idx=None, debug_timing=False):
    model.train()

    optimizer.zero_grad()
    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        print(f"[TrainFn] Batch {debug_batch_idx}: entering model forward")
        torch.cuda.synchronize()
        stage_start = time.time()

    # Run model
    transl_err, rot_err = model(rgb_img, refl_img)
    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        torch.cuda.synchronize()
        forward_time = time.time() - stage_start
        print(f"[TrainFn] Batch {debug_batch_idx}: model forward done")
    
    # Check for NaN in model outputs
    if torch.isnan(transl_err).any() or torch.isnan(rot_err).any():
        print("Warning: NaN detected in model outputs")
        return {'total_loss': torch.tensor(0.0, device=rgb_img.device, requires_grad=True)}, rot_err, transl_err

    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        print(f"[TrainFn] Batch {debug_batch_idx}: entering loss computation")
        torch.cuda.synchronize()
        stage_start = time.time()
    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)
    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        torch.cuda.synchronize()
        loss_time = time.time() - stage_start
        print(f"[TrainFn] Batch {debug_batch_idx}: loss computation done")
    
    # Check for NaN in loss before backward
    if torch.isnan(losses['total_loss']):
        print("Warning: NaN detected in loss, skipping backward")
        return losses, rot_err, transl_err

    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        print(f"[TrainFn] Batch {debug_batch_idx}: entering backward")
        torch.cuda.synchronize()
        stage_start = time.time()
    losses['total_loss'].backward()
    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        torch.cuda.synchronize()
        backward_time = time.time() - stage_start
        print(f"[TrainFn] Batch {debug_batch_idx}: backward done")
    
    # Gradient clipping to prevent NaN
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        print(f"[TrainFn] Batch {debug_batch_idx}: optimizer step")
        torch.cuda.synchronize()
        stage_start = time.time()
    optimizer.step()
    if debug_timing and debug_batch_idx is not None and debug_batch_idx < 3:
        torch.cuda.synchronize()
        optimizer_time = time.time() - stage_start
        print(f"[TrainFn] Batch {debug_batch_idx}: optimizer done")
        print(
            f"[TrainFn] Batch {debug_batch_idx}: timings "
            f"forward={forward_time:.2f}s, "
            f"loss={loss_time:.2f}s, "
            f"backward={backward_time:.2f}s, "
            f"optimizer={optimizer_time:.2f}s"
        )

    return losses, rot_err, transl_err


# CNN test
@ex.capture
def val(model, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.eval()

    # Run model
    with torch.no_grad():
        transl_err, rot_err = model(rgb_img, refl_img)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    # if loss != 'points_distance':
    #     total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    # else:
    #     total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    # Initialize on the same device as target_transl
    total_trasl_error = torch.tensor(0.0, device=target_transl.device)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    # # output image: The overlay image of the input rgb image and the projected lidar pointcloud depth image
    # cam_intrinsic = camera_model[0]
    # rotated_point_cloud =
    # R_predicted = quat2mat(R_predicted[0])
    # T_predicted = tvector2mat(T_predicted[0])
    # RT_predicted = torch.mm(T_predicted, R_predicted)
    # rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

    return losses, total_trasl_error.item(), total_rot_error.sum().item(), rot_err, transl_err


@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run, but no GPU is available.")

    if _config['dataset'] in ['hercules', 'lg_innotek']:
        sensor_mode = _config.get('sensor_mode', 'radar').lower()  # Default to 'radar' for backward compatibility
        if sensor_mode not in ['lidar', 'radar', 'both', 'tri']:
            raise ValueError(f"Invalid sensor_mode: {sensor_mode}. Must be 'lidar', 'radar', 'both', or 'tri'")

        dataset_label = 'Hercules' if _config['dataset'] == 'hercules' else 'LG_Innotek'
        if sensor_mode == 'both':
            print(f"Using {dataset_label} Camera-LIDAR and Camera-RADAR datasets (both)")
        elif sensor_mode == 'tri':
            print(f"Using {dataset_label} tri-modal Camera-LIDAR-RADAR dataset")
        else:
            print(f"Using {dataset_label} Camera-{sensor_mode.upper()} dataset")

        if _config['dataset'] == 'hercules':
            val_scene = _config['val_scene']
            train_scene = _config.get('train_scene')
        else:
            val_scene = _config.get('lg_val_scene', ['afternoon_campus_2'])
            train_scene = _config.get('lg_train_scene', ['afternoon_parking_lot_1', 'afternoon_campus_1'])

        if _config['dataset'] == 'hercules' and val_scene is None:
            # Get all scene directories
            scene_list = [d for d in os.listdir(_config['data_folder']) 
                         if os.path.isdir(os.path.join(_config['data_folder'], d))]
            scene_list.sort()
            
            # Find scenes with calibration.yaml (could be in scene or subfolder like CMRNext)
            valid_scenes = []
            for scene in scene_list:
                scene_path = os.path.join(_config['data_folder'], scene)
                # Check if calibration.yaml is directly in scene folder
                if os.path.exists(os.path.join(scene_path, 'calibration.yaml')):
                    valid_scenes.append(scene)
                else:
                    subdir_path = os.path.join(scene_path, 'CMRNext')
                    if os.path.isdir(subdir_path):
                        if os.path.exists(os.path.join(subdir_path, 'calibration.yaml')):
                            valid_scenes.append(scene)
            
            if len(valid_scenes) > 0:
                val_scene = valid_scenes[0]
                print(f"Found {len(valid_scenes)} valid scenes: {valid_scenes}")
            else:
                raise ValueError(f"No valid scenes found in Hercules dataset at {_config['data_folder']}")
        
        # Convert single scene to list if needed
        if isinstance(val_scene, str):
            val_scene = [val_scene]
        
        if train_scene is not None:
            if isinstance(train_scene, str):
                train_scene = [train_scene]
            # Ensure no overlap between train_scene and val_scene
            train_scene = [s for s in train_scene if s not in val_scene]
            if len(train_scene) == 0:
                print("Warning: All train_scene scenes are in val_scene. Using all scenes except val_scene for training.")
                train_scene = None
        
        print("Val Scenes: ", val_scene)
        if train_scene is not None:
            print("Train Scenes: ", train_scene)
        else:
            print("Train Scenes: All scenes except val_scene")
        print(f"Sensor Mode: {sensor_mode.upper()}")
        
        # Select dataset class(es) based on sensor mode
        if sensor_mode == 'lidar':
            dataset_class = DatasetLidarCameraHercules if _config['dataset'] == 'hercules' else DatasetLidarCameraLGInnotek
            dataset_class_val = None  # Same as train
        elif sensor_mode == 'radar':
            dataset_class = DatasetCameraRadarHercules if _config['dataset'] == 'hercules' else DatasetCameraRadarLGInnotek
            dataset_class_val = None  # Same as train
        elif sensor_mode == 'both':
            if _config['dataset'] == 'hercules':
                dataset_class = [DatasetLidarCameraHercules, DatasetCameraRadarHercules]
            else:
                dataset_class = [DatasetLidarCameraLGInnotek, DatasetCameraRadarLGInnotek]
            dataset_class_val = None  # Same as train
        else:  # tri
            if _config['dataset'] != 'lg_innotek':
                raise ValueError("tri sensor_mode is currently implemented only for lg_innotek")
            dataset_class = DatasetTriModalLGInnotek
            dataset_class_val = None
    else:
        val_sequence = _config['val_sequence']
        if val_sequence is None:
            raise TypeError('val_sequences cannot be None')
        else:
            val_sequence = f"{val_sequence:02d}"
            print("Val Sequence: ", val_sequence)
            if _config['dataset'] == 'kitti/odom':
                dataset_class = DatasetLidarCameraKittiOdometry
            elif _config['dataset'] == 'kitti/raw':
                from DatasetLidarCamera import DatasetLidarCameraKittiRaw
                dataset_class = DatasetLidarCameraKittiRaw
            else:
                raise ValueError(f"Unknown dataset: {_config['dataset']}")
    img_shape = (720, 1280)
    input_size = (288, 512)
    checkpoints_dir = os.path.join(_config["checkpoints"], _config['dataset'])

    if _config['dataset'] in ['hercules', 'lg_innotek']:
        if sensor_mode == 'both':
            # Create both lidar and radar datasets
            common_kwargs = {}
            if _config['dataset'] == 'lg_innotek':
                common_kwargs['val_frame_limit'] = _config.get('lg_val_frame_limit', 3000)

            dataset_train_lidar = dataset_class[0](_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                   split='train', use_reflectance=_config['use_reflectance'],
                                                   val_scene=val_scene, train_scene=train_scene, **common_kwargs)
            dataset_train_radar = dataset_class[1](_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                   split='train', use_reflectance=_config['use_reflectance'],
                                                   val_scene=val_scene, train_scene=train_scene, **common_kwargs)
            dataset_train = ConcatDataset([dataset_train_lidar, dataset_train_radar])
            
            dataset_val_lidar = dataset_class[0](_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                 split='val', use_reflectance=_config['use_reflectance'],
                                                 val_scene=val_scene, train_scene=train_scene, **common_kwargs)
            dataset_val_radar = dataset_class[1](_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                 split='val', use_reflectance=_config['use_reflectance'],
                                                 val_scene=val_scene, train_scene=train_scene, **common_kwargs)
            dataset_val = ConcatDataset([dataset_val_lidar, dataset_val_radar])
        elif sensor_mode == 'tri':
            common_kwargs = {}
            if _config['dataset'] == 'lg_innotek':
                common_kwargs['val_frame_limit'] = _config.get('lg_val_frame_limit', 3000)
            common_kwargs['input_size'] = input_size
            common_kwargs['max_depth'] = _config['max_depth']
            dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                          split='train', use_reflectance=_config['use_reflectance'],
                                          val_scene=val_scene, train_scene=train_scene, **common_kwargs)
            dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                        split='val', use_reflectance=_config['use_reflectance'],
                                        val_scene=val_scene, train_scene=train_scene, **common_kwargs)
        else:
            common_kwargs = {}
            if _config['dataset'] == 'lg_innotek':
                common_kwargs['val_frame_limit'] = _config.get('lg_val_frame_limit', 3000)

            dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                          split='train', use_reflectance=_config['use_reflectance'],
                                          val_scene=val_scene, train_scene=train_scene, **common_kwargs)
            dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                        split='val', use_reflectance=_config['use_reflectance'],
                                        val_scene=val_scene, train_scene=train_scene, **common_kwargs)
    else:
        dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                      split='train', use_reflectance=_config['use_reflectance'],
                                      val_sequence=val_sequence)
        dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                    split='val', use_reflectance=_config['use_reflectance'],
                                    val_sequence=val_sequence)
    if _config['dataset'] in ['hercules', 'lg_innotek']:
        # Use checkpoint_name from config if provided, otherwise auto-generate
        if _config.get('checkpoint_name') is not None:
            checkpoint_name = _config['checkpoint_name']
        else:
            # Auto-generate from val_scene list and sensor_mode
            if isinstance(val_scene, list):
                checkpoint_name = f"{'_'.join(val_scene)}_{sensor_mode}"
            else:
                checkpoint_name = f"{val_scene}_{sensor_mode}"
        model_savepath = os.path.join(checkpoints_dir, checkpoint_name, 'models')
    else:
        model_savepath = os.path.join(checkpoints_dir, 'val_seq_' + val_sequence, 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    if _config['dataset'] in ['hercules', 'lg_innotek']:
        # Use the same checkpoint_name for log path
        log_savepath = os.path.join(checkpoints_dir, checkpoint_name, 'log')
    else:
        log_savepath = os.path.join(checkpoints_dir, 'val_seq_' + val_sequence, 'log')
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    train_writer = SummaryWriter(os.path.join(log_savepath, 'train'))
    val_writer = SummaryWriter(os.path.join(log_savepath, 'val'))

    wandb_enabled = bool(_config.get('wandb_enabled', False))
    if wandb_enabled and wandb is None:
        print("Warning: wandb is not installed. Disabling wandb logging.")
        wandb_enabled = False
    if wandb_enabled:
        wandb_run_name = _config.get('wandb_name', 'TriBaseline')
        if _config.get('network', '').startswith('Tri') and wandb_run_name == 'TriBaseline':
            wandb_run_name = f"TriBaseline_{_config.get('dataset', 'unknown')}_{_config.get('sensor_mode', 'tri')}"
        wandb.init(
            project=_config.get('wandb_project', 'LCCNet_TriModal'),
            entity=_config.get('wandb_entity', 'LGIT_calib'),
            name=wandb_run_name,
            dir=log_savepath,
            mode=_config.get('wandb_mode', 'online'),
            config=dict(_config),
            reinit=True,
        )
        if wandb.run is not None:
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val_iter/step")
            wandb.define_metric("val_iter/*", step_metric="val_iter/step")
            wandb.define_metric("epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("best/*", step_metric="epoch")
            wandb.run.summary['checkpoint_dir'] = model_savepath
            wandb.run.summary['log_dir'] = log_savepath

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)

    print(len(TrainImgLoader))
    print(len(ValImgLoader))

    # loss function choice
    if _config['network'].startswith('Tri'):
        loss_fn = TriModalPairwiseLoss(_config.get('tri_loss_w_t', 1.0), _config.get('tri_loss_w_q', 1.0))
    else:
        if _config['loss'] == 'simple':
            loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
        elif _config['loss'] == 'geometric':
            loss_fn = GeometricLoss()
            loss_fn = loss_fn.to(device)
        elif _config['loss'] == 'points_distance':
            loss_fn = DistancePoints3D()
        elif _config['loss'] == 'L1':
            loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
        elif _config['loss'] == 'combined':
            loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
        else:
            raise ValueError("Unknown Loss Function")

    #runs = datetime.now().strftime('%b%d_%H-%M-%S') + "/"
    # train_writer = SummaryWriter('./logs/' + runs)
    #ex.info["tensorflow"] = {}
    #ex.info["tensorflow"]["logdirs"] = ['./logs/' + runs]

    # network choice and settings
    if _config['network'].startswith('Res'):
        feat = 1
        md = 4
        split = _config['network'].split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"
        model = LCCNet(input_size, use_feat_from=feat, md=md,
                         use_reflectance=_config['use_reflectance'], dropout=_config['dropout'],
                         Action_Func='leakyrelu', attention=False, res_num=18)
    elif _config['network'].startswith('Tri'):
        model = TriModalCalibNet(
            camera_pretrained=False,
            activation='leakyrelu',
            head_hidden_dim=256,
            dropout=_config['dropout'],
            debug_shapes=_config.get('tri_debug_shapes', True),
        )
    else:
        raise TypeError("Network unknown")
    if _config['weights'] is not None and os.path.exists(_config['weights']):
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
    elif _config['weights'] is not None:
        print(f"Warning: Weights file not found: {_config['weights']}. Starting from scratch.")

        # original saved file with DataParallel
        # state_dict = torch.load(model_path)
        # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)

    # model = model.to(device)
    use_dataparallel = _config.get('use_dataparallel', True)
    visible_gpus = [gpu.strip() for gpu in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if gpu.strip() != '']
    if use_dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1 and len(visible_gpus) > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume'] and os.path.exists(_config['weights']):
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        if 'optimizer' in checkpoint:
            opt_state_dict = checkpoint['optimizer']
            optimizer.load_state_dict(opt_state_dict)
        if starting_epoch != 0 and 'epoch' in checkpoint:
            starting_epoch = checkpoint['epoch']

    if _config['network'].startswith('Tri') and _config.get('tri_run_one_batch', True):
        model.train()
        sample = next(iter(TrainImgLoader))
        rgb = sample['rgb'].to(device, non_blocking=True)
        lidar_proj = sample['lidar_proj'].to(device, non_blocking=True)
        radar_proj = sample['radar_proj'].to(device, non_blocking=True)
        gt_batch = {
            'T_CL_t_gt': sample['T_CL_t_gt'].to(device, non_blocking=True),
            'T_CL_q_gt': sample['T_CL_q_gt'].to(device, non_blocking=True),
            'T_CR_t_gt': sample['T_CR_t_gt'].to(device, non_blocking=True),
            'T_CR_q_gt': sample['T_CR_q_gt'].to(device, non_blocking=True),
            'T_LR_t_gt': sample['T_LR_t_gt'].to(device, non_blocking=True),
            'T_LR_q_gt': sample['T_LR_q_gt'].to(device, non_blocking=True),
        }
        print(f"[Tri Debug] rgb={tuple(rgb.shape)} lidar_proj={tuple(lidar_proj.shape)} radar_proj={tuple(radar_proj.shape)}")
        assert rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[2] == 288 and rgb.shape[3] == 512
        assert lidar_proj.ndim == 4 and lidar_proj.shape[1] == 1 and lidar_proj.shape[2] == 288 and lidar_proj.shape[3] == 512
        assert radar_proj.ndim == 4 and radar_proj.shape[1] == 2 and radar_proj.shape[2] == 288 and radar_proj.shape[3] == 512

        optimizer.zero_grad()
        pred = model(rgb, lidar_proj, radar_proj)
        print(f"[Tri Debug] T_CL_t={tuple(pred['T_CL_t'].shape)} T_CL_q={tuple(pred['T_CL_q'].shape)}")
        print(f"[Tri Debug] T_CR_t={tuple(pred['T_CR_t'].shape)} T_CR_q={tuple(pred['T_CR_q'].shape)}")
        print(f"[Tri Debug] T_LR_t={tuple(pred['T_LR_t'].shape)} T_LR_q={tuple(pred['T_LR_q'].shape)}")
        assert pred['T_CL_t'].shape[1] == 3 and pred['T_CL_q'].shape[1] == 4
        assert pred['T_CR_t'].shape[1] == 3 and pred['T_CR_q'].shape[1] == 4
        assert pred['T_LR_t'].shape[1] == 3 and pred['T_LR_q'].shape[1] == 4

        losses = loss_fn(pred, gt_batch)
        print(
            f"[Tri Debug] total_loss={losses['total_loss'].item():.6f} "
            f"CL={losses['loss_cl'].item():.6f} "
            f"CR={losses['loss_cr'].item():.6f} "
            f"LR={losses['loss_lr'].item():.6f}"
        )
        losses['total_loss'].backward()
        optimizer.step()
        print("[Tri Debug] one-batch dataset -> model -> loss -> backward succeeded.")
        return losses['total_loss'].item()

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    starting_epoch = 0

    def run_tri_validation(epoch, train_epoch_loss=None):
        nonlocal BEST_VAL_LOSS, old_save_filename, val_iter
        model.eval()
        total_val_loss = 0.0
        total_input_t_cl = 0.0
        total_input_t_cr = 0.0
        total_input_t_lr = 0.0
        total_input_r_cl = 0.0
        total_input_r_cr = 0.0
        total_input_r_lr = 0.0
        total_val_t_cl = 0.0
        total_val_t_cr = 0.0
        total_val_t_lr = 0.0
        total_val_r_cl = 0.0
        total_val_r_cr = 0.0
        total_val_r_lr = 0.0

        def _pose_to_matrix(t_vec, q_vec):
            T = torch.eye(4, device=t_vec.device, dtype=t_vec.dtype)
            T[:3, :3] = quat2mat(q_vec)[:3, :3]
            T[:3, 3] = t_vec
            return T.detach().cpu().numpy().astype(np.float32)

        def _matrix_batch_to_quaternion(T_batch):
            quats = [quaternion_from_matrix(T_batch[i]) for i in range(T_batch.shape[0])]
            return torch.stack(quats, dim=0)

        with torch.no_grad():
            for batch_idx, sample in enumerate(ValImgLoader):
                rgb = sample['rgb'].to(device, non_blocking=True)
                lidar_proj = sample['lidar_proj'].to(device, non_blocking=True)
                radar_proj = sample['radar_proj'].to(device, non_blocking=True)
                T_cl_input = sample['T_CL_input'].to(device, non_blocking=True)
                T_cr_input = sample['T_CR_input'].to(device, non_blocking=True)
                T_lr_input = torch.linalg.inv(T_cl_input) @ T_cr_input
                gt_batch = {
                    'T_CL_t_gt': sample['T_CL_t_gt'].to(device, non_blocking=True),
                    'T_CL_q_gt': sample['T_CL_q_gt'].to(device, non_blocking=True),
                    'T_CR_t_gt': sample['T_CR_t_gt'].to(device, non_blocking=True),
                    'T_CR_q_gt': sample['T_CR_q_gt'].to(device, non_blocking=True),
                    'T_LR_t_gt': sample['T_LR_t_gt'].to(device, non_blocking=True),
                    'T_LR_q_gt': sample['T_LR_q_gt'].to(device, non_blocking=True),
                }
                input_q_cl = _matrix_batch_to_quaternion(T_cl_input)
                input_q_cr = _matrix_batch_to_quaternion(T_cr_input)
                input_q_lr = _matrix_batch_to_quaternion(T_lr_input)
                pred = model(rgb, lidar_proj, radar_proj)
                loss = loss_fn(pred, gt_batch)
                total_val_loss += loss['total_loss'].item() * rgb.shape[0]

                total_input_t_cl += torch.norm(T_cl_input[:, :3, 3] - gt_batch['T_CL_t_gt'], dim=1).sum().item() * 100.0
                total_input_t_cr += torch.norm(T_cr_input[:, :3, 3] - gt_batch['T_CR_t_gt'], dim=1).sum().item() * 100.0
                total_input_t_lr += torch.norm(T_lr_input[:, :3, 3] - gt_batch['T_LR_t_gt'], dim=1).sum().item() * 100.0
                total_input_r_cl += (quaternion_distance(input_q_cl, gt_batch['T_CL_q_gt'], device) * 180.0 / math.pi).sum().item()
                total_input_r_cr += (quaternion_distance(input_q_cr, gt_batch['T_CR_q_gt'], device) * 180.0 / math.pi).sum().item()
                total_input_r_lr += (quaternion_distance(input_q_lr, gt_batch['T_LR_q_gt'], device) * 180.0 / math.pi).sum().item()

                total_val_t_cl += torch.norm(pred['T_CL_t'] - gt_batch['T_CL_t_gt'], dim=1).sum().item() * 100.0
                total_val_t_cr += torch.norm(pred['T_CR_t'] - gt_batch['T_CR_t_gt'], dim=1).sum().item() * 100.0
                total_val_t_lr += torch.norm(pred['T_LR_t'] - gt_batch['T_LR_t_gt'], dim=1).sum().item() * 100.0
                total_val_r_cl += (quaternion_distance(pred['T_CL_q'], gt_batch['T_CL_q_gt'], device) * 180.0 / math.pi).sum().item()
                total_val_r_cr += (quaternion_distance(pred['T_CR_q'], gt_batch['T_CR_q_gt'], device) * 180.0 / math.pi).sum().item()
                total_val_r_lr += (quaternion_distance(pred['T_LR_q'], gt_batch['T_LR_q_gt'], device) * 180.0 / math.pi).sum().item()

                if batch_idx % _config['log_frequency'] == 0:
                    print(
                        f"[Tri Val] Iter {batch_idx}/{len(ValImgLoader)} "
                        f"loss={loss['total_loss'].item():.4f} "
                        f"CL={loss['loss_cl'].item():.4f} "
                        f"CR={loss['loss_cr'].item():.4f} "
                        f"LR={loss['loss_lr'].item():.4f}"
                    )
                    if wandb_enabled and wandb is not None and _config.get('wandb_log_images', True):
                        show_idx = 0
                        rgb_show = rgb[show_idx].detach().cpu()
                        lidar_show = lidar_proj[show_idx].detach().cpu()
                        radar_show = radar_proj[show_idx].detach().cpu()
                        radar_vis = radar_show[0:1, :, :]
                        dataset_show_idx = batch_idx * _config['batch_size'] + show_idx
                        dataset_item = ValImgLoader.dataset.all_files[dataset_show_idx]
                        rgb_img, calib = ValImgLoader.dataset._load_image(dataset_item['image_path'])
                        orig_hw = (rgb_img.height, rgb_img.width)
                        lidar_pc = _load_point_cloud(dataset_item['lidar_path'], ValImgLoader.dataset.pcd_reader)
                        radar_pc = _load_point_cloud(dataset_item['radar_path'], ValImgLoader.dataset.pcd_reader)

                        T_cl_pred = _pose_to_matrix(pred['T_CL_t'][show_idx], pred['T_CL_q'][show_idx])
                        T_cr_pred = _pose_to_matrix(pred['T_CR_t'][show_idx], pred['T_CR_q'][show_idx])
                        T_lr_pred = _pose_to_matrix(pred['T_LR_t'][show_idx], pred['T_LR_q'][show_idx])
                        T_cl_gt = _pose_to_matrix(gt_batch['T_CL_t_gt'][show_idx], gt_batch['T_CL_q_gt'][show_idx])
                        T_cr_gt = _pose_to_matrix(gt_batch['T_CR_t_gt'][show_idx], gt_batch['T_CR_q_gt'][show_idx])

                        lidar_depth_pred, _ = ValImgLoader.dataset._project_to_image(lidar_pc, T_cl_pred, calib, orig_hw)
                        radar_depth_pred, _ = ValImgLoader.dataset._project_to_image(radar_pc, T_cr_pred, calib, orig_hw)
                        lidar_depth_gt, _ = ValImgLoader.dataset._project_to_image(lidar_pc, T_cl_gt, calib, orig_hw)
                        radar_depth_gt, _ = ValImgLoader.dataset._project_to_image(radar_pc, T_cr_gt, calib, orig_hw)

                        lidar_pred_vis = torch.from_numpy(lidar_depth_pred).unsqueeze(0).unsqueeze(0)
                        lidar_gt_vis = torch.from_numpy(lidar_depth_gt).unsqueeze(0).unsqueeze(0)
                        radar_pred_vis = torch.from_numpy(radar_depth_pred).unsqueeze(0).unsqueeze(0)
                        radar_gt_vis = torch.from_numpy(radar_depth_gt).unsqueeze(0).unsqueeze(0)
                        lidar_pred_vis = F.interpolate(lidar_pred_vis, size=ValImgLoader.dataset.input_size, mode='bilinear', align_corners=False)
                        lidar_gt_vis = F.interpolate(lidar_gt_vis, size=ValImgLoader.dataset.input_size, mode='bilinear', align_corners=False)
                        radar_pred_vis = F.interpolate(radar_pred_vis, size=ValImgLoader.dataset.input_size, mode='bilinear', align_corners=False)
                        radar_gt_vis = F.interpolate(radar_gt_vis, size=ValImgLoader.dataset.input_size, mode='bilinear', align_corners=False)

                        wandb_data = {
                            "epoch": epoch,
                            "val/rgb": _tensor_to_wandb_image(rgb_show),
                            "val/rgb_lidar_overlay_input": wandb.Image(overlay_imgs(rgb_show, lidar_show.unsqueeze(0))),
                            "val/rgb_radar_overlay_input": wandb.Image(overlay_imgs(rgb_show, radar_vis.unsqueeze(0))),
                            "val/rgb_lidar_overlay_gt": wandb.Image(overlay_imgs(rgb_show, lidar_gt_vis)),
                            "val/rgb_radar_overlay_gt": wandb.Image(overlay_imgs(rgb_show, radar_gt_vis)),
                            "val/rgb_lidar_overlay_pred": wandb.Image(overlay_imgs(rgb_show, lidar_pred_vis)),
                            "val/rgb_radar_overlay_pred": wandb.Image(overlay_imgs(rgb_show, radar_pred_vis)),
                        }
                        _wandb_log(wandb_enabled, wandb_data)

        val_loss = total_val_loss / len(dataset_val)
        input_t_cl = total_input_t_cl / len(dataset_val)
        input_t_cr = total_input_t_cr / len(dataset_val)
        input_t_lr = total_input_t_lr / len(dataset_val)
        input_r_cl = total_input_r_cl / len(dataset_val)
        input_r_cr = total_input_r_cr / len(dataset_val)
        input_r_lr = total_input_r_lr / len(dataset_val)
        val_t_cl = total_val_t_cl / len(dataset_val)
        val_t_cr = total_val_t_cr / len(dataset_val)
        val_t_lr = total_val_t_lr / len(dataset_val)
        val_r_cl = total_val_r_cl / len(dataset_val)
        val_r_cr = total_val_r_cr / len(dataset_val)
        val_r_lr = total_val_r_lr / len(dataset_val)
        print("------------------------------------")
        print(f"total val loss = {val_loss:.3f}")
        print(f"val input translation cm: CL={input_t_cl:.3f} CR={input_t_cr:.3f} LR={input_t_lr:.3f}")
        print(f"val input rotation deg:    CL={input_r_cl:.3f} CR={input_r_cr:.3f} LR={input_r_lr:.3f}")
        print(f"val translation cm: CL={val_t_cl:.3f} CR={val_t_cr:.3f} LR={val_t_lr:.3f}")
        print(f"val rotation deg:    CL={val_r_cl:.3f} CR={val_r_cr:.3f} LR={val_r_lr:.3f}")
        print("------------------------------------")
        _run.log_scalar("Val_Loss", val_loss, epoch)
        _run.log_scalar("Val_input_t_CL", input_t_cl, epoch)
        _run.log_scalar("Val_input_t_CR", input_t_cr, epoch)
        _run.log_scalar("Val_input_t_LR", input_t_lr, epoch)
        _run.log_scalar("Val_input_r_CL", input_r_cl, epoch)
        _run.log_scalar("Val_input_r_CR", input_r_cr, epoch)
        _run.log_scalar("Val_input_r_LR", input_r_lr, epoch)
        _run.log_scalar("Val_t_CL", val_t_cl, epoch)
        _run.log_scalar("Val_t_CR", val_t_cr, epoch)
        _run.log_scalar("Val_t_LR", val_t_lr, epoch)
        _run.log_scalar("Val_r_CL", val_r_cl, epoch)
        _run.log_scalar("Val_r_CR", val_r_cr, epoch)
        _run.log_scalar("Val_r_LR", val_r_lr, epoch)
        _wandb_log(wandb_enabled, {
            "epoch": epoch,
            "val/loss": val_loss,
            "val/input_t_CL_cm": input_t_cl,
            "val/input_t_CR_cm": input_t_cr,
            "val/input_t_LR_cm": input_t_lr,
            "val/input_r_CL_deg": input_r_cl,
            "val/input_r_CR_deg": input_r_cr,
            "val/input_r_LR_deg": input_r_lr,
        })

        if train_epoch_loss is not None and val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            _run.result = val_loss
            savefilename = f'{model_savepath}/checkpoint_tri_e{epoch}_{val_loss:.3f}.tar'
            torch.save({
                'config': _config,
                'epoch': epoch,
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_epoch_loss,
                'val_loss': val_loss,
                'val_t_cl': val_t_cl,
                'val_t_cr': val_t_cr,
                'val_t_lr': val_t_lr,
                'val_r_cl': val_r_cl,
                'val_r_cr': val_r_cr,
                'val_r_lr': val_r_lr,
            }, savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None and os.path.exists(old_save_filename):
                os.remove(old_save_filename)
            old_save_filename = savefilename

        return val_loss

    if _config['network'].startswith('Tri'):
        print("Running initial validation before training...")
        run_tri_validation(epoch=-1, train_epoch_loss=None)

    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        if _config['optimizer'] != 'adam':
            _run.log_scalar("LR", _config['BASE_LEARNING_RATE'] *
                            math.exp((1 - epoch) * 4e-2), epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = _config['BASE_LEARNING_RATE'] * \
                                    math.exp((1 - epoch) * 4e-2)
        else:
            #scheduler.step(epoch%100)
            _run.log_scalar("LR", scheduler.get_lr()[0])
        current_lr = optimizer.param_groups[0]['lr']
        _wandb_log(wandb_enabled, {"epoch": epoch, "train/lr": current_lr})

        if _config['network'].startswith('Tri'):
            # Minimal tri-modal baseline loop (no cost-volume path, no recurrence/reliability).
            model.train()
            for batch_idx, sample in enumerate(TrainImgLoader):
                rgb = sample['rgb'].to(device, non_blocking=True)
                lidar_proj = sample['lidar_proj'].to(device, non_blocking=True)
                radar_proj = sample['radar_proj'].to(device, non_blocking=True)
                gt_batch = {
                    'T_CL_t_gt': sample['T_CL_t_gt'].to(device, non_blocking=True),
                    'T_CL_q_gt': sample['T_CL_q_gt'].to(device, non_blocking=True),
                    'T_CR_t_gt': sample['T_CR_t_gt'].to(device, non_blocking=True),
                    'T_CR_q_gt': sample['T_CR_q_gt'].to(device, non_blocking=True),
                    'T_LR_t_gt': sample['T_LR_t_gt'].to(device, non_blocking=True),
                    'T_LR_q_gt': sample['T_LR_q_gt'].to(device, non_blocking=True),
                }

                optimizer.zero_grad()
                pred = model(rgb, lidar_proj, radar_proj)
                loss = loss_fn(pred, gt_batch)
                loss['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss['total_loss'].item() * rgb.shape[0]
                if batch_idx % _config['log_frequency'] == 0:
                    print(
                        f"[Tri Train] Iter {batch_idx}/{len(TrainImgLoader)} "
                        f"loss={loss['total_loss'].item():.4f} "
                        f"CL={loss['loss_cl'].item():.4f} "
                        f"CR={loss['loss_cr'].item():.4f} "
                        f"LR={loss['loss_lr'].item():.4f}"
                    )

            train_epoch_loss = total_train_loss / len(dataset_train)
            print("------------------------------------")
            print(f"epoch {epoch} total training loss = {train_epoch_loss:.3f}")
            print(f"Total epoch time = {time.time() - epoch_start_time:.2f}")
            print("------------------------------------")
            _run.log_scalar("Total training loss", train_epoch_loss, epoch)
            _wandb_log(wandb_enabled, {"epoch": epoch, "train/epoch_loss": train_epoch_loss})
            if (epoch + 1) % 20 == 0:
                run_tri_validation(epoch=epoch, train_epoch_loss=train_epoch_loss)
            continue


        ## Training ##
        time_for_50ep = time.time()
        debug_timing = _config.get('debug_timing', False)
        for batch_idx, sample in enumerate(TrainImgLoader):
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            if debug_timing and batch_idx < 3:
                print(f"[Train] Batch {batch_idx} fetched: batch_size={len(sample['rgb'])}")
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            start_preprocess = time.time()
            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose
                real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

                sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda() # 变换到相机坐标系下的激光雷达点云
                pc_lidar = sample['point_cloud'][idx].clone()

                if _config['max_depth'] < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

                depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']

                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T * R

                pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc

                if _config['max_depth'] < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

                depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']

                rgb = sample['rgb'][idx].cuda()
                rgb, depth_img, depth_gt, shape_pad = preprocess_projected_inputs(
                    rgb, depth_img, depth_gt, img_shape, input_size
                )

                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input, lidar_input = resize_model_inputs(rgb_input, lidar_input, input_size)
            end_preprocess = time.time()
            if debug_timing and batch_idx < 3:
                print(f"[Train] Batch {batch_idx} preprocess done in {end_preprocess - start_preprocess:.2f}s")
                print(f"[Train] Batch {batch_idx} resize/pad done, starting forward")
            loss, R_predicted,  T_predicted = train(model, optimizer, rgb_input, lidar_input,
                                                   sample['tr_error'], sample['rot_error'],
                                                   loss_fn, sample['point_cloud'], _config['loss'],
                                                   debug_batch_idx=batch_idx,
                                                   debug_timing=debug_timing)
            if debug_timing and batch_idx < 3:
                print(f"[Train] Batch {batch_idx} forward/backward done in {time.time() - end_preprocess:.2f}s")

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            if batch_idx % _config['log_frequency'] == 0:
                show_idx = 0
                # output image: The overlay image of the input rgb image
                # and the projected lidar pointcloud depth image
                rotated_point_cloud = pc_rotated_input[show_idx]
                R_predicted = quat2mat(R_predicted[show_idx])
                T_predicted = tvector2mat(T_predicted[show_idx])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                depth_pred, uv = lidar_project_depth(rotated_point_cloud,
                                                    sample['calib'][show_idx],
                                                    real_shape_input[show_idx]) # or image_shape
                depth_pred /= _config['max_depth']
                depth_pred = fit_tensor_to_canvas(depth_pred, img_shape)

                pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
                input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
                gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0))

                pred_show = torch.from_numpy(pred_show)
                pred_show = pred_show.permute(2, 0, 1)
                input_show = torch.from_numpy(input_show)
                input_show = input_show.permute(2, 0, 1)
                gt_show = torch.from_numpy(gt_show)
                gt_show = gt_show.permute(2, 0, 1)

                train_writer.add_image("input_proj_lidar", input_show, train_iter)
                train_writer.add_image("gt_proj_lidar", gt_show, train_iter)
                train_writer.add_image("pred_proj_lidar", pred_show, train_iter)

                train_writer.add_scalar("Loss_Total", loss['total_loss'].item(), train_iter)
                train_writer.add_scalar("Loss_Translation", loss['transl_loss'].item(), train_iter)
                train_writer.add_scalar("Loss_Rotation", loss['rot_loss'].item(), train_iter)
                if _config['loss'] == 'combined':
                    train_writer.add_scalar("Loss_Point_clouds", loss['point_clouds_loss'].item(), train_iter)
                wandb_data = {
                    "train/step": train_iter,
                    "train/loss_total": loss['total_loss'].item(),
                    "train/loss_translation": loss['transl_loss'].item(),
                    "train/loss_rotation": loss['rot_loss'].item(),
                }
                if _config['loss'] == 'combined':
                    wandb_data["train/loss_point_clouds"] = loss['point_clouds_loss'].item()
                if wandb_enabled and wandb is not None and _config.get('wandb_log_images', True):
                    wandb_data["train/input_proj_lidar"] = _tensor_to_wandb_image(input_show)
                    wandb_data["train/gt_proj_lidar"] = _tensor_to_wandb_image(gt_show)
                    wandb_data["train/pred_proj_lidar"] = _tensor_to_wandb_image(pred_show)
                _wandb_log(wandb_enabled, wandb_data)

            local_loss += loss['total_loss'].item()

            if batch_idx % 10 == 0:
                avg_loss = local_loss / 10 if batch_idx != 0 else local_loss

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {avg_loss:.3f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      #f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
                      f'time for recent iters: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                _run.log_scalar("Loss", avg_loss, train_iter)
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset_train), epoch)
        _wandb_log(wandb_enabled, {
            "epoch": epoch,
            "train/epoch_loss": total_train_loss / len(dataset_train),
            "train/epoch_time_sec": time.time() - epoch_start_time,
        })

        ## Validation ##
        total_val_loss = 0.
        total_val_t = 0.
        total_val_r = 0.

        local_loss = 0.0
        for batch_idx, sample in enumerate(ValImgLoader):
            #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose
                real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

                sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda() # 变换到相机坐标系下的激光雷达点云
                pc_lidar = sample['point_cloud'][idx].clone()

                if _config['max_depth'] < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

                depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']

                reflectance = None
                if _config['use_reflectance']:
                    reflectance = sample['reflectance'][idx].cuda()

                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T * R

                pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc

                if _config['max_depth'] < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

                depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']

                if _config['use_reflectance']:
                    # This need to be checked
                    # cam_params = sample['calib'][idx].cuda()
                    # cam_model = CameraModel()
                    # cam_model.focal_length = cam_params[:2]
                    # cam_model.principal_point = cam_params[2:]
                    # uv, depth, _, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance)
                    # uv = uv.long()
                    # indexes = depth_img[uv[:,1], uv[:,0]] == depth
                    # refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                    # refl_img[uv[indexes, 1], uv[indexes, 0]] = refl[0, indexes]
                    refl_img = None

                # if not _config['use_reflectance']:
                #     depth_img = depth_img.unsqueeze(0)
                # else:
                #     depth_img = torch.stack((depth_img, refl_img))

                rgb = sample['rgb'][idx].cuda()
                rgb, depth_img, depth_gt, shape_pad = preprocess_projected_inputs(
                    rgb, depth_img, depth_gt, img_shape, input_size
                )

                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input, lidar_input = resize_model_inputs(rgb_input, lidar_input, input_size)

            loss, trasl_e, rot_e, R_predicted,  T_predicted = val(model, rgb_input, lidar_input,
                                                                  sample['tr_error'], sample['rot_error'],
                                                                  loss_fn, sample['point_cloud'], _config['loss'])

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            if batch_idx % _config['log_frequency'] == 0:
                show_idx = 0
                # output image: The overlay image of the input rgb image
                # and the projected lidar pointcloud depth image
                rotated_point_cloud = pc_rotated_input[show_idx]
                R_predicted = quat2mat(R_predicted[show_idx])
                T_predicted = tvector2mat(T_predicted[show_idx])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                depth_pred, uv = lidar_project_depth(rotated_point_cloud,
                                                    sample['calib'][show_idx],
                                                    real_shape_input[show_idx]) # or image_shape
                depth_pred /= _config['max_depth']
                depth_pred = fit_tensor_to_canvas(depth_pred, img_shape)

                pred_show = overlay_imgs(rgb_show[show_idx], depth_pred.unsqueeze(0))
                input_show = overlay_imgs(rgb_show[show_idx], lidar_show[show_idx].unsqueeze(0))
                gt_show = overlay_imgs(rgb_show[show_idx], lidar_gt[show_idx].unsqueeze(0))

                pred_show = torch.from_numpy(pred_show)
                pred_show = pred_show.permute(2, 0, 1)
                input_show = torch.from_numpy(input_show)
                input_show = input_show.permute(2, 0, 1)
                gt_show = torch.from_numpy(gt_show)
                gt_show = gt_show.permute(2, 0, 1)

                val_writer.add_image("input_proj_lidar", input_show, val_iter)
                val_writer.add_image("gt_proj_lidar", gt_show, val_iter)
                val_writer.add_image("pred_proj_lidar", pred_show, val_iter)

                val_writer.add_scalar("Loss_Total", loss['total_loss'].item(), val_iter)
                val_writer.add_scalar("Loss_Translation", loss['transl_loss'].item(), val_iter)
                val_writer.add_scalar("Loss_Rotation", loss['rot_loss'].item(), val_iter)
                if _config['loss'] == 'combined':
                    val_writer.add_scalar("Loss_Point_clouds", loss['point_clouds_loss'].item(), val_iter)
                wandb_data = {
                    "val_iter/step": val_iter,
                    "val_iter/loss_total": loss['total_loss'].item(),
                    "val_iter/loss_translation": loss['transl_loss'].item(),
                    "val_iter/loss_rotation": loss['rot_loss'].item(),
                }
                if _config['loss'] == 'combined':
                    wandb_data["val_iter/loss_point_clouds"] = loss['point_clouds_loss'].item()
                if wandb_enabled and wandb is not None and _config.get('wandb_log_images', True):
                    wandb_data["val/input_proj_lidar"] = _tensor_to_wandb_image(input_show)
                    wandb_data["val/gt_proj_lidar"] = _tensor_to_wandb_image(gt_show)
                    wandb_data["val/pred_proj_lidar"] = _tensor_to_wandb_image(pred_show)
                _wandb_log(wandb_enabled, wandb_data)


            total_val_t += trasl_e
            total_val_r += rot_e
            local_loss += loss['total_loss'].item()

            if batch_idx % 10 == 0:
                avg_loss = local_loss / 10 if batch_idx != 0 else local_loss
                print('Iter %d val loss = %.3f , time = %.2f' % (batch_idx, avg_loss,
                                                                  (time.time() - start_time)/lidar_input.shape[0]))
                local_loss = 0.0
            total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
            val_iter += 1

        print("------------------------------------")
        print('total val loss = %.3f' % (total_val_loss / len(dataset_val)))
        print(f'total traslation error: {total_val_t / len(dataset_val)} cm')
        print(f'total rotation error: {total_val_r / len(dataset_val)} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
        _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
        _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)
        _wandb_log(wandb_enabled, {
            "epoch": epoch,
            "val/loss": total_val_loss / len(dataset_val),
            "val/t_error_cm": total_val_t / len(dataset_val),
            "val/r_error_deg": total_val_r / len(dataset_val),
        })

        # SAVE
        val_loss = total_val_loss / len(dataset_val)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            #_run.result = BEST_VAL_LOSS
            if _config['rescale_transl'] > 0:
                _run.result = total_val_t / len(dataset_val)
            else:
                _run.result = total_val_r / len(dataset_val)
            savefilename = f'{model_savepath}/checkpoint_r{_config["max_r"]:.2f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.tar'
            torch.save({
                'config': _config,
                'epoch': epoch,
                # 'state_dict': model.state_dict(), # single gpu
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), # multi gpu
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_train),
                'val_loss': total_val_loss / len(dataset_val),
            }, savefilename)
            print(f'Model saved as {savefilename}')
            _wandb_log(wandb_enabled, {
                "epoch": epoch,
                "best/val_loss": val_loss,
                "best/checkpoint_path": savefilename,
                "best/epoch": epoch,
            })
            if wandb_enabled and wandb.run is not None:
                wandb.run.summary['best_val_loss'] = val_loss
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_checkpoint_path'] = savefilename
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    if wandb_enabled and wandb is not None and wandb.run is not None:
        wandb.run.summary['full_training_time_hr'] = (time.time() - start_full_time) / 3600
        wandb.finish()
    return _run.result
