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
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import ConcatDataset
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetLidarCamera import DatasetLidarCameraKittiOdometry, DatasetLidarCameraHercules
from DatasetCameraRadar import DatasetCameraRadarHercules
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
from models.LCCNet import LCCNet

from quaternion_distances import quaternion_distance

from tensorboardX import SummaryWriter
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

# Disable git info to avoid errors in Docker containers
# In Docker, git repository may not work properly, so disable it
save_git_info = False
ex = Experiment("LCCNet", save_git_info=save_git_info)
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = '/workspace/data/checkpoints/LCCNet/'
    dataset = 'hercules' # 'kitti/odom', 'kitti/raw', 'hercules'
    sensor_mode = 'lidar'  # For Hercules: 'lidar', 'radar', or 'both'
    data_folder = '/workspace/data/hercules'
    use_reflectance = False
    val_sequence = 0  # For KITTI
    val_scene = ['library_1']  # For Hercules (None = use first scene, list = use multiple scenes)
    train_scene = ['SC_1', 'SC_3', 'island_1']  # For Hercules (None = use all scenes except val_scene, list = use specific scenes for training)
    checkpoint_name = 'test'  # For Hercules: custom checkpoint name for saving (None = auto-generate from val_scene and sensor_mode)
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


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


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


# CCN training
@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.train()

    optimizer.zero_grad()

    # Run model
    transl_err, rot_err = model(rgb_img, refl_img)
    
    # Check for NaN in model outputs
    if torch.isnan(transl_err).any() or torch.isnan(rot_err).any():
        print("Warning: NaN detected in model outputs")
        return {'total_loss': torch.tensor(0.0, device=rgb_img.device, requires_grad=True)}, rot_err, transl_err

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)
    
    # Check for NaN in loss before backward
    if torch.isnan(losses['total_loss']):
        print("Warning: NaN detected in loss, skipping backward")
        return losses, rot_err, transl_err

    losses['total_loss'].backward()
    
    # Gradient clipping to prevent NaN
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

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

    if _config['dataset'] == 'hercules':
        sensor_mode = _config.get('sensor_mode', 'radar').lower()  # Default to 'radar' for backward compatibility
        if sensor_mode not in ['lidar', 'radar', 'both']:
            raise ValueError(f"Invalid sensor_mode: {sensor_mode}. Must be 'lidar', 'radar', or 'both'")
        
        if sensor_mode == 'both':
            print(f"Using Hercules Camera-LIDAR and Camera-RADAR datasets (both)")
        else:
            print(f"Using Hercules Camera-{sensor_mode.upper()} dataset")
        val_scene = _config['val_scene']
        if val_scene is None:
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
        
        train_scene = _config.get('train_scene')
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
            dataset_class = DatasetLidarCameraHercules
            dataset_class_val = None  # Same as train
        elif sensor_mode == 'radar':
            dataset_class = DatasetCameraRadarHercules
            dataset_class_val = None  # Same as train
        else:  # both
            dataset_class = [DatasetLidarCameraHercules, DatasetCameraRadarHercules]
            dataset_class_val = None  # Same as train
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
    img_shape = (384, 1280) # 网络的输入尺度
    input_size = (256, 512)
    checkpoints_dir = os.path.join(_config["checkpoints"], _config['dataset'])

    if _config['dataset'] == 'hercules':
        if sensor_mode == 'both':
            # Create both lidar and radar datasets
            dataset_train_lidar = DatasetLidarCameraHercules(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                             split='train', use_reflectance=_config['use_reflectance'],
                                                             val_scene=val_scene, train_scene=train_scene)
            dataset_train_radar = DatasetCameraRadarHercules(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                             split='train', use_reflectance=_config['use_reflectance'],
                                                             val_scene=val_scene, train_scene=train_scene)
            dataset_train = ConcatDataset([dataset_train_lidar, dataset_train_radar])
            
            dataset_val_lidar = DatasetLidarCameraHercules(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                          split='val', use_reflectance=_config['use_reflectance'],
                                                          val_scene=val_scene)
            dataset_val_radar = DatasetCameraRadarHercules(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                                          split='val', use_reflectance=_config['use_reflectance'],
                                                          val_scene=val_scene)
            dataset_val = ConcatDataset([dataset_val_lidar, dataset_val_radar])
        else:
            dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                          split='train', use_reflectance=_config['use_reflectance'],
                                          val_scene=val_scene, train_scene=train_scene)
            dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                        split='val', use_reflectance=_config['use_reflectance'],
                                        val_scene=val_scene)
    else:
        dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                      split='train', use_reflectance=_config['use_reflectance'],
                                      val_sequence=val_sequence)
        dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                    split='val', use_reflectance=_config['use_reflectance'],
                                    val_sequence=val_sequence)
    if _config['dataset'] == 'hercules':
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
    if _config['dataset'] == 'hercules':
        # Use the same checkpoint_name for log path
        log_savepath = os.path.join(checkpoints_dir, checkpoint_name, 'log')
    else:
        log_savepath = os.path.join(checkpoints_dir, 'val_seq_' + val_sequence, 'log')
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    train_writer = SummaryWriter(os.path.join(log_savepath, 'train'))
    val_writer = SummaryWriter(os.path.join(log_savepath, 'val'))

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
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss()
        loss_fn = loss_fn.cuda()
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
    model = nn.DataParallel(model)
    model = model.cuda()

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

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    starting_epoch = 0
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


        ## Training ##
        time_for_50ep = time.time()
        for batch_idx, sample in enumerate(TrainImgLoader):
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

                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                rgb = sample['rgb'][idx].cuda()
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)

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
            rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")
            end_preprocess = time.time()
            loss, R_predicted,  T_predicted = train(model, optimizer, rgb_input, lidar_input,
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
                depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

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

            local_loss += loss['total_loss'].item()

            if batch_idx % 50 == 0 and batch_idx != 0:

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      #f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
                      f'time for 50 iter: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                _run.log_scalar("Loss", local_loss/50, train_iter)
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])
            break

        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
        _run.log_scalar("Total training loss", total_train_loss / len(dataset_train), epoch)

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

                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                rgb = sample['rgb'][idx].cuda()
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)

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
            rgb_input = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
            lidar_input = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")

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
                depth_pred = F.pad(depth_pred, shape_pad_input[show_idx])

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


            total_val_t += trasl_e
            total_val_r += rot_e
            local_loss += loss['total_loss'].item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('Iter %d val loss = %.3f , time = %.2f' % (batch_idx, local_loss/50.,
                                                                  (time.time() - start_time)/lidar_input.shape[0]))
                local_loss = 0.0
            total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
            val_iter += 1
            break

        print("------------------------------------")
        print('total val loss = %.3f' % (total_val_loss / len(dataset_val)))
        print(f'total traslation error: {total_val_t / len(dataset_val)} cm')
        print(f'total rotation error: {total_val_r / len(dataset_val)} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
        _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
        _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)

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
                'state_dict': model.module.state_dict(), # multi gpu
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_train),
                'val_loss': total_val_loss / len(dataset_val),
            }, savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result