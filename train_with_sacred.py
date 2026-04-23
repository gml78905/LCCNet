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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetLGInnotek import (
    DatasetTriModalLGInnotek,
    DatasetTriModalHercules,
    TriSequenceDataset,
    _load_point_cloud,
)
from losses_tri import TriModalPairwiseLoss
from models.tri_joint.model import TriModalJointCalibNetV2
from models.tri_joint_v3.model import TriModalJointCalibNetV3Lite

from quaternion_distances import quaternion_distance

from tensorboardX import SummaryWriter
from utils import (
    merge_inputs,
    overlay_imgs,
    quat2mat,
    project_pointcloud_to_image_torch,
    project_pointclouds_to_image_torch_batched,
    quaternion_from_matrix,
)

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
    dataset = 'hercules'
    sensor_mode = 'tri'
    data_folder = '/workspace/data/hercules'
    use_reflectance = False
    val_sequence = 0  # For KITTI
    val_scene = ['library_1']  # For Hercules (None = use first scene, list = use multiple scenes)
    train_scene = ['SC_1', 'SC_3', 'island_1']  # For Hercules (None = use all scenes except val_scene, list = use specific scenes for training)
    val_frame_limit = None  # For Hercules tri/val split frame cap
    checkpoint_name = 'test'  # For Hercules: custom checkpoint name for saving (None = auto-generate from val_scene and sensor_mode)
    lg_train_scene = ['afternoon_parking_lot_1', 'afternoon_campus_1']
    lg_val_scene = ['afternoon_campus_2']
    lg_val_frame_limit = 3000
    epochs = 120
    BASE_LEARNING_RATE = 1e-4  # 1e-4
    loss = 'tri_pairwise'
    max_t = 0.5 # 1.5, 1.0,  0.5,  0.2,  0.1
    max_r = 5.0 # 20.0, 10.0, 5.0,  2.0,  1.0
    batch_size = 120  # 120
    num_worker = 8
    network = 'TriJointV2'
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
    tri_loss_w_t = 1.0
    tri_loss_w_q = 1.0
    tri_lambda_loop = 0.0
    tri_use_sequence = False
    tri_seq_len = 4
    tri_seq_stride = 1
    tri_use_recurrence = False
    tri_recurrent_hidden_dim = 256
    tri_frame_cache_size = 32
    tri_project_on_gpu = True
    tri_pointcloud_cache = True
    tri_pointcloud_cache_write = True
    tri_use_amp = True
    tri_amp_dtype = 'fp16'  # 'fp16' or 'bf16'
    tri_use_compile = True
    tri_compile_mode = 'reduce-overhead'
    tri_joint_debug_return_aux = True
    tri_sync_batchnorm = True
    tri_freeze_bn_after = 100  # Freeze BN running stats late in training to reduce DDP eval jitter.
    loader_persistent_workers = True
    loader_prefetch_factor = 4


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

@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run, but no GPU is available.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    is_main_process = (rank == 0)
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")

    if _config['network'] not in ['TriJointV2', 'TriJointV3Lite']:
        raise ValueError(
            f"Only network in ['TriJointV2', 'TriJointV3Lite'] is supported now, got {_config['network']}"
        )
    if _config['sensor_mode'].lower() != 'tri':
        raise ValueError(f"Only sensor_mode='tri' is supported now, got {_config['sensor_mode']}")
    if _config['dataset'] not in ['hercules', 'lg_innotek']:
        raise ValueError(f"Only dataset in ['hercules', 'lg_innotek'] is supported now, got {_config['dataset']}")

    sensor_mode = 'tri'
    dataset_label = 'Hercules' if _config['dataset'] == 'hercules' else 'LG_Innotek'
    print(f"Using {dataset_label} tri-modal Camera-LIDAR-RADAR dataset")

    if _config['dataset'] == 'hercules':
        val_scene = _config['val_scene']
        train_scene = _config.get('train_scene')
    else:
        val_scene = _config.get('lg_val_scene', ['afternoon_campus_2'])
        train_scene = _config.get('lg_train_scene', ['afternoon_parking_lot_1', 'afternoon_campus_1'])

    if _config['dataset'] == 'hercules' and val_scene is None:
        scene_list = [d for d in os.listdir(_config['data_folder']) if os.path.isdir(os.path.join(_config['data_folder'], d))]
        scene_list.sort()
        valid_scenes = []
        for scene in scene_list:
            scene_path = os.path.join(_config['data_folder'], scene)
            if os.path.exists(os.path.join(scene_path, 'calibration.yaml')):
                valid_scenes.append(scene)
            else:
                subdir_path = os.path.join(scene_path, 'CMRNext')
                if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, 'calibration.yaml')):
                    valid_scenes.append(scene)
        if not valid_scenes:
            raise ValueError(f"No valid scenes found in Hercules dataset at {_config['data_folder']}")
        val_scene = valid_scenes[0]
        print(f"Found {len(valid_scenes)} valid scenes: {valid_scenes}")

    if isinstance(val_scene, str):
        val_scene = [val_scene]
    if train_scene is not None:
        if isinstance(train_scene, str):
            train_scene = [train_scene]
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

    if _config['dataset'] == 'lg_innotek':
        dataset_class = DatasetTriModalLGInnotek
    else:
        dataset_class = DatasetTriModalHercules
    img_shape = (720, 1280)
    input_size = (288, 512)
    checkpoints_dir = os.path.join(_config["checkpoints"], _config['dataset'])

    common_kwargs = {}
    if _config['dataset'] == 'lg_innotek':
        common_kwargs['val_frame_limit'] = _config.get('lg_val_frame_limit', 3000)
    else:
        common_kwargs['val_frame_limit'] = _config.get('val_frame_limit')
    common_kwargs['input_size'] = input_size
    common_kwargs['max_depth'] = _config['max_depth']
    common_kwargs['project_on_gpu'] = _config.get('tri_project_on_gpu', False)
    common_kwargs['pointcloud_cache'] = _config.get('tri_pointcloud_cache', True)
    common_kwargs['pointcloud_cache_write'] = _config.get('tri_pointcloud_cache_write', True)

    dataset_train = dataset_class(
        _config['data_folder'],
        max_r=_config['max_r'],
        max_t=_config['max_t'],
        split='train',
        use_reflectance=_config['use_reflectance'],
        val_scene=val_scene,
        train_scene=train_scene,
        **common_kwargs,
    )
    dataset_val = dataset_class(
        _config['data_folder'],
        max_r=_config['max_r'],
        max_t=_config['max_t'],
        split='val',
        use_reflectance=_config['use_reflectance'],
        val_scene=val_scene,
        train_scene=train_scene,
        **common_kwargs,
    )
    if _config.get('tri_use_sequence', False):
        dataset_train = TriSequenceDataset(
            dataset_train,
            seq_len=_config.get('tri_seq_len', 4),
            stride=_config.get('tri_seq_stride', 1),
            cache_size=_config.get('tri_frame_cache_size', 8),
            project_on_gpu=_config.get('tri_project_on_gpu', False),
        )
        dataset_val = TriSequenceDataset(
            dataset_val,
            seq_len=_config.get('tri_seq_len', 4),
            stride=_config.get('tri_seq_stride', 1),
            cache_size=_config.get('tri_frame_cache_size', 8),
            project_on_gpu=_config.get('tri_project_on_gpu', False),
        )

    if _config.get('checkpoint_name') is not None:
        checkpoint_name = _config['checkpoint_name']
    else:
        checkpoint_name = f"{'_'.join(val_scene)}_{sensor_mode}" if isinstance(val_scene, list) else f"{val_scene}_{sensor_mode}"
    model_savepath = os.path.join(checkpoints_dir, checkpoint_name, 'models')
    os.makedirs(model_savepath, exist_ok=True)
    log_savepath = os.path.join(checkpoints_dir, checkpoint_name, 'log')
    os.makedirs(log_savepath, exist_ok=True)
    train_writer = SummaryWriter(os.path.join(log_savepath, 'train')) if is_main_process else None
    val_writer = SummaryWriter(os.path.join(log_savepath, 'val')) if is_main_process else None

    wandb_enabled = bool(_config.get('wandb_enabled', False)) and is_main_process
    if wandb_enabled and wandb is None:
        print("Warning: wandb is not installed. Disabling wandb logging.")
        wandb_enabled = False
    if wandb_enabled:
        wandb_run_name = _config.get('wandb_name', 'TriBaseline')
        if wandb_run_name == 'TriBaseline':
            wandb_run_name = f"{_config.get('network', 'TriJoint')}_{_config.get('dataset', 'unknown')}_tri"
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

    np.random.seed(seed + rank)
    torch.random.manual_seed(seed + rank)
    random.seed(seed + rank)

    def init_fn(x): return _init_fn(x, seed)

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    persistent_workers = bool(_config.get('loader_persistent_workers', True)) and num_worker > 0
    prefetch_factor = int(_config.get('loader_prefetch_factor', 4))
    loader_kwargs = {}
    if num_worker > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor
    train_sampler = None
    val_sampler = None
    if distributed and _config.get('network', '').startswith('Tri') and _config.get('use_dataparallel', True):
        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
        # Validation in tri path is executed on rank0 only, so keep full val set there.
        val_sampler = None
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=(train_sampler is None),
                                                 sampler=train_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 **loader_kwargs)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                sampler=val_sampler,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True,
                                                **loader_kwargs)

    print(len(TrainImgLoader))
    print(len(ValImgLoader))

    loss_fn = TriModalPairwiseLoss(
        _config.get('tri_loss_w_t', 1.0),
        _config.get('tri_loss_w_q', 1.0),
        _config.get('tri_lambda_loop', 0.0),
    )

    #runs = datetime.now().strftime('%b%d_%H-%M-%S') + "/"
    # train_writer = SummaryWriter('./logs/' + runs)
    #ex.info["tensorflow"] = {}
    #ex.info["tensorflow"]["logdirs"] = ['./logs/' + runs]

    if _config['network'] == 'TriJointV2':
        model = TriModalJointCalibNetV2(
            camera_pretrained=False,
            activation='leakyrelu',
            head_hidden_dim=256,
            head_dropout=_config['dropout'],
        )
    else:
        model = TriModalJointCalibNetV3Lite(
            camera_pretrained=False,
            activation='leakyrelu',
            head_hidden_dim=256,
            head_dropout=_config['dropout'],
        )

    if distributed and _config.get('network', '').startswith('Tri') and _config.get('tri_sync_batchnorm', True):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process:
            print("[Tri] SyncBatchNorm enabled for distributed training")

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

    if _config.get('tri_use_compile', True):
        compile_mode = _config.get('tri_compile_mode', 'reduce-overhead')
        try:
            model = torch.compile(model, mode=compile_mode)
            if is_main_process:
                print(f"[Tri] torch.compile enabled (mode={compile_mode})")
        except Exception as compile_error:
            if is_main_process:
                print(f"[Tri] torch.compile disabled due to error: {compile_error}")

    # model = model.to(device)
    use_dataparallel = _config.get('use_dataparallel', True)
    visible_gpus = [gpu.strip() for gpu in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if gpu.strip() != '']
    if distributed and use_dataparallel:
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        if use_dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1 and len(visible_gpus) > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
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

    tri_amp_enabled = bool(_config.get('tri_use_amp', True)) and device.type == 'cuda'
    tri_amp_dtype_name = str(_config.get('tri_amp_dtype', 'fp16')).lower()
    tri_amp_dtype = torch.bfloat16 if tri_amp_dtype_name in ('bf16', 'bfloat16') else torch.float16
    tri_use_grad_scaler = tri_amp_enabled and tri_amp_dtype == torch.float16
    tri_grad_scaler = torch.cuda.amp.GradScaler(enabled=tri_use_grad_scaler)

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    train_iter = 0
    val_iter = 0
    starting_epoch = 0

    def _batch_item_count(rgb_tensor):
        if rgb_tensor.ndim == 5:
            return rgb_tensor.shape[0] * rgb_tensor.shape[1]
        return rgb_tensor.shape[0]

    def _flatten_pose_tensor(tensor):
        if tensor.ndim <= 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])

    def _flatten_matrix_tensor(tensor):
        if tensor.ndim <= 3:
            return tensor
        return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])

    def _matrix_to_pose_batch(T_batch):
        flat = _flatten_matrix_tensor(T_batch)
        t_batch = flat[:, :3, 3]
        q_batch = quaternion_from_matrix(flat)
        return t_batch, q_batch

    def _pose_batch_to_matrix(t_batch, q_batch):
        flat_t = _flatten_pose_tensor(t_batch)
        flat_q = _flatten_pose_tensor(q_batch)
        T = quat2mat(flat_q)
        T[:, :3, 3] = flat_t
        return T

    def _apply_delta_to_input(pred_t, pred_q, input_T):
        delta_T = _pose_batch_to_matrix(pred_t, pred_q)
        flat_input = _flatten_matrix_tensor(input_T)
        corrected_T = torch.bmm(delta_T, flat_input)
        corrected_t, corrected_q = _matrix_to_pose_batch(corrected_T)
        return corrected_t, corrected_q, corrected_T

    def _tri_model_forward(forward_model, rgb_batch, lidar_batch, radar_batch, return_aux=False):
        """
        TriJointV2 / TriJointV3Lite:
          out = (pred_dict, new_state) or (pred_dict, new_state, aux_dict)
        """
        if return_aux:
            out = forward_model(rgb_batch, lidar_batch, radar_batch, state=None, return_aux=True)
        else:
            out = forward_model(rgb_batch, lidar_batch, radar_batch)
        if isinstance(out, tuple):
            if len(out) == 3:
                pred_out, new_state_out, aux_out = out
                return pred_out, new_state_out, aux_out
            if len(out) == 2:
                pred_out, new_state_out = out
                return pred_out, new_state_out, {}
        return out, None, {}

    def _set_batchnorm_eval(module):
        # Keep affine BN parameters trainable, but stop running-stat updates.
        for child in module.modules():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                child.eval()

    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    def _prepare_rgb_batch(rgb_batch):
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        if rgb_batch.ndim == 4:
            rgb_batch = rgb_batch.unsqueeze(1)
            squeeze_seq = True
        else:
            squeeze_seq = False

        rgb_batch = rgb_batch.float()
        if _config.get('tri_project_on_gpu', False):
            rgb_batch = (rgb_batch - rgb_mean) / rgb_std

        if squeeze_seq:
            rgb_batch = rgb_batch.squeeze(1)
        return rgb_batch

    def _project_batch_with_optional_vectorization(pc_tensor, pc_mask, T_tensor, calib_tensor, image_hw_tensor):
        flat_hw = image_hw_tensor.reshape(-1, 2)
        same_hw = bool(torch.all(flat_hw == flat_hw[:1]).item())
        if same_hw:
            return project_pointclouds_to_image_torch_batched(
                pc_tensor,
                pc_mask,
                T_tensor,
                calib_tensor,
                image_hw_tensor,
                _config['max_depth'],
            )

        leading_shape = pc_tensor.shape[:-2]
        flat_pc = pc_tensor.reshape(-1, pc_tensor.shape[-2], pc_tensor.shape[-1])
        flat_mask = pc_mask.reshape(-1, pc_mask.shape[-1])
        flat_T = T_tensor.reshape(-1, 4, 4)
        flat_calib = calib_tensor.reshape(-1, 3, 3)
        flat_hw = image_hw_tensor.reshape(-1, 2)

        depth_list = []
        aux_list = []
        for i in range(flat_pc.shape[0]):
            points = flat_pc[i][flat_mask[i]]
            depth, aux = project_pointcloud_to_image_torch(
                points,
                flat_T[i],
                flat_calib[i],
                flat_hw[i].tolist(),
                _config['max_depth'],
            )
            depth_list.append(depth)
            aux_list.append(aux)

        h_max = max(depth.shape[0] for depth in depth_list)
        w_max = max(depth.shape[1] for depth in depth_list)
        depth_tensor = torch.zeros((len(depth_list), h_max, w_max), device=device, dtype=flat_pc.dtype)
        aux_tensor = torch.zeros_like(depth_tensor)
        for i, (depth, aux) in enumerate(zip(depth_list, aux_list)):
            h, w = depth.shape
            depth_tensor[i, :h, :w] = depth
            aux_tensor[i, :h, :w] = aux

        return (
            depth_tensor.reshape(*leading_shape, h_max, w_max),
            aux_tensor.reshape(*leading_shape, h_max, w_max),
        )

    def _build_tri_projection_batch(sample_batch):
        if 'lidar_proj' in sample_batch and 'radar_proj' in sample_batch:
            return (
                sample_batch['lidar_proj'].to(device, non_blocking=True),
                sample_batch['radar_proj'].to(device, non_blocking=True),
            )

        input_size = dataset_train.input_size if getattr(dataset_train, 'input_size', None) is not None else (288, 512)
        if 'lidar_pc_seq' in sample_batch:
            lidar_pc_seq = sample_batch['lidar_pc_seq'].to(device, non_blocking=True)
            radar_pc_seq = sample_batch['radar_pc_seq'].to(device, non_blocking=True)
            lidar_mask_seq = sample_batch['lidar_pc_seq_mask'].to(device, non_blocking=True)
            radar_mask_seq = sample_batch['radar_pc_seq_mask'].to(device, non_blocking=True)
            calib_seq = sample_batch['calib_seq'].to(device, non_blocking=True)
            image_hw_seq = sample_batch['image_hw_seq'].to(device, non_blocking=True)
            T_cl_input_seq = sample_batch['T_CL_input'].to(device, non_blocking=True)
            T_cr_input_seq = sample_batch['T_CR_input'].to(device, non_blocking=True)
            lidar_depth, _ = _project_batch_with_optional_vectorization(
                lidar_pc_seq,
                lidar_mask_seq,
                T_cl_input_seq,
                calib_seq,
                image_hw_seq,
            )
            radar_depth, radar_aux = _project_batch_with_optional_vectorization(
                radar_pc_seq,
                radar_mask_seq,
                T_cr_input_seq,
                calib_seq,
                image_hw_seq,
            )
            lidar_tensor = lidar_depth.unsqueeze(2)
            radar_tensor = torch.stack([radar_depth, radar_aux], dim=2)
            bsz, seq_len = lidar_tensor.shape[:2]
            if input_size is not None:
                lidar_tensor = F.interpolate(
                    lidar_tensor.reshape(bsz * seq_len, 1, lidar_tensor.shape[-2], lidar_tensor.shape[-1]),
                    size=input_size,
                    mode='bilinear',
                    align_corners=False,
                ).reshape(bsz, seq_len, 1, input_size[0], input_size[1])
                radar_tensor = F.interpolate(
                    radar_tensor.reshape(bsz * seq_len, 2, radar_tensor.shape[-2], radar_tensor.shape[-1]),
                    size=input_size,
                    mode='bilinear',
                    align_corners=False,
                ).reshape(bsz, seq_len, 2, input_size[0], input_size[1])
            return lidar_tensor, radar_tensor

        calib_batch = sample_batch['calib'].to(device, non_blocking=True)
        image_hw_batch = sample_batch['image_hw'].to(device, non_blocking=True)
        T_cl_input_batch = sample_batch['T_CL_input'].to(device, non_blocking=True)
        T_cr_input_batch = sample_batch['T_CR_input'].to(device, non_blocking=True)
        lidar_pc = sample_batch['lidar_pc'].to(device, non_blocking=True)
        radar_pc = sample_batch['radar_pc'].to(device, non_blocking=True)
        lidar_mask = sample_batch['lidar_pc_mask'].to(device, non_blocking=True)
        radar_mask = sample_batch['radar_pc_mask'].to(device, non_blocking=True)
        lidar_depth, _ = _project_batch_with_optional_vectorization(
            lidar_pc,
            lidar_mask,
            T_cl_input_batch,
            calib_batch,
            image_hw_batch,
        )
        radar_depth, radar_aux = _project_batch_with_optional_vectorization(
            radar_pc,
            radar_mask,
            T_cr_input_batch,
            calib_batch,
            image_hw_batch,
        )
        lidar_tensor = lidar_depth.unsqueeze(1)
        radar_tensor = torch.stack([radar_depth, radar_aux], dim=1)
        if input_size is not None:
            lidar_tensor = F.interpolate(lidar_tensor, size=input_size, mode='bilinear', align_corners=False)
            radar_tensor = F.interpolate(radar_tensor, size=input_size, mode='bilinear', align_corners=False)
        return lidar_tensor, radar_tensor

    if _config['network'].startswith('Tri') and _config.get('tri_run_one_batch', True):
        is_tri_joint = _config['network'].startswith('TriJoint')
        model.train()
        sample = next(iter(TrainImgLoader))
        rgb = _prepare_rgb_batch(sample['rgb'])
        lidar_proj, radar_proj = _build_tri_projection_batch(sample)
        gt_batch = {
            'T_CL_t_gt': sample['T_CL_t_gt'].to(device, non_blocking=True),
            'T_CL_q_gt': sample['T_CL_q_gt'].to(device, non_blocking=True),
            'T_CR_t_gt': sample['T_CR_t_gt'].to(device, non_blocking=True),
            'T_CR_q_gt': sample['T_CR_q_gt'].to(device, non_blocking=True),
            'T_LR_t_gt': sample['T_LR_t_gt'].to(device, non_blocking=True),
            'T_LR_q_gt': sample['T_LR_q_gt'].to(device, non_blocking=True),
            'T_CL_input': sample['T_CL_input'].to(device, non_blocking=True),
            'T_CR_input': sample['T_CR_input'].to(device, non_blocking=True),
        }
        gt_batch['T_LR_input'] = torch.linalg.inv(gt_batch['T_CL_input']) @ gt_batch['T_CR_input']
        if is_main_process:
            print(f"[Tri Debug] rgb={tuple(rgb.shape)} lidar_proj={tuple(lidar_proj.shape)} radar_proj={tuple(radar_proj.shape)}")
        if rgb.ndim == 5:
            assert rgb.shape[2] == 3 and rgb.shape[3] == 288 and rgb.shape[4] == 512
            assert lidar_proj.ndim == 5 and lidar_proj.shape[2] == 1 and lidar_proj.shape[3] == 288 and lidar_proj.shape[4] == 512
            assert radar_proj.ndim == 5 and radar_proj.shape[2] == 2 and radar_proj.shape[3] == 288 and radar_proj.shape[4] == 512
        else:
            assert rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[2] == 288 and rgb.shape[3] == 512
            assert lidar_proj.ndim == 4 and lidar_proj.shape[1] == 1 and lidar_proj.shape[2] == 288 and lidar_proj.shape[3] == 512
            assert radar_proj.ndim == 4 and radar_proj.shape[1] == 2 and radar_proj.shape[2] == 288 and radar_proj.shape[3] == 512

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=tri_amp_dtype, enabled=tri_amp_enabled):
            debug_return_aux = bool(_config.get('tri_joint_debug_return_aux', True)) and is_tri_joint
            pred, new_state_debug, aux_debug = _tri_model_forward(model, rgb, lidar_proj, radar_proj, return_aux=debug_return_aux)
            losses = loss_fn(pred, gt_batch)
        if is_main_process:
            print(f"[Tri Debug] T_CL_t={tuple(pred['T_CL_t'].shape)} T_CL_q={tuple(pred['T_CL_q'].shape)}")
            print(f"[Tri Debug] T_CR_t={tuple(pred['T_CR_t'].shape)} T_CR_q={tuple(pred['T_CR_q'].shape)}")
            print(f"[Tri Debug] T_LR_t={tuple(pred['T_LR_t'].shape)} T_LR_q={tuple(pred['T_LR_q'].shape)}")
            if is_tri_joint and new_state_debug is not None:
                state_shapes = " ".join(f"{k}={tuple(v.shape)}" for k, v in new_state_debug.items())
                print(f"[TriJoint Debug] new_state {state_shapes}")
            if is_tri_joint and debug_return_aux and len(aux_debug) > 0:
                aux_keys = [
                    'r_cl_coarse', 'w_c', 'w_cl', 'r_cl_ref', 'gate_cl', 'R_cam', 'r_cl0',
                    'F_joint_map', 'E_joint_map', 'align_cl', 'z_lid_valid', 'z_rad_valid',
                    'fusion_map_summary', 'summary_bias', 'z_joint_ref',
                ]
                for k in aux_keys:
                    if k in aux_debug:
                        print(f"[TriJoint Debug] aux[{k}]={tuple(aux_debug[k].shape)}")
                if 'R_cam' in aux_debug:
                    print(
                        f"[TriJoint Debug] R_cam stats "
                        f"mean={aux_debug['R_cam'].mean().item():.4f} "
                        f"min={aux_debug['R_cam'].min().item():.4f} "
                        f"max={aux_debug['R_cam'].max().item():.4f}"
                    )
        assert pred['T_CL_t'].shape[-1] == 3 and pred['T_CL_q'].shape[-1] == 4
        assert pred['T_CR_t'].shape[-1] == 3 and pred['T_CR_q'].shape[-1] == 4
        assert pred['T_LR_t'].shape[-1] == 3 and pred['T_LR_q'].shape[-1] == 4

        if is_main_process:
            print(
                f"[Tri Debug] total_loss={losses['total_loss'].item():.6f} "
                f"CL={losses['loss_cl'].item():.6f} "
                f"CR={losses['loss_cr'].item():.6f} "
                f"LR={losses['loss_lr'].item():.6f} "
                f"LOOP={losses['loss_loop'].item():.6f}"
            )
        if tri_use_grad_scaler:
            tri_grad_scaler.scale(losses['total_loss']).backward()
            tri_grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            tri_grad_scaler.step(optimizer)
            tri_grad_scaler.update()
        else:
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if distributed:
            dist.barrier()
        if is_main_process:
            print("[Tri Debug] one-batch dataset -> model -> loss -> backward succeeded.")
            return losses['total_loss'].item()
        return None

    def run_tri_validation(epoch, train_epoch_loss=None):
        nonlocal BEST_VAL_LOSS, val_iter
        model.eval()
        eval_model = model.module if isinstance(model, DDP) else model
        eval_model.eval()
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
        total_eval_count = 0

        def _pose_to_matrix(t_vec, q_vec):
            T = torch.eye(4, device=t_vec.device, dtype=t_vec.dtype)
            T[:3, :3] = quat2mat(q_vec)[:3, :3]
            T[:3, 3] = t_vec
            return T.detach().cpu().numpy().astype(np.float32)

        def _matrix_batch_to_quaternion(T_batch):
            flat = _flatten_matrix_tensor(T_batch)
            return quaternion_from_matrix(flat)

        with torch.no_grad():
            for batch_idx, sample in enumerate(ValImgLoader):
                rgb = _prepare_rgb_batch(sample['rgb'])
                lidar_proj, radar_proj = _build_tri_projection_batch(sample)
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
                    'T_CL_input': T_cl_input,
                    'T_CR_input': T_cr_input,
                    'T_LR_input': T_lr_input,
                }
                input_q_cl = _matrix_batch_to_quaternion(T_cl_input)
                input_q_cr = _matrix_batch_to_quaternion(T_cr_input)
                input_q_lr = _matrix_batch_to_quaternion(T_lr_input)
                with torch.autocast(device_type='cuda', dtype=tri_amp_dtype, enabled=tri_amp_enabled):
                    pred, _, _ = _tri_model_forward(eval_model, rgb, lidar_proj, radar_proj, return_aux=False)
                    loss = loss_fn(pred, gt_batch)
                batch_item_count = _batch_item_count(rgb)
                total_eval_count += batch_item_count
                total_val_loss += loss['total_loss'].item() * batch_item_count

                flat_t_cl_input = _flatten_matrix_tensor(T_cl_input)[:, :3, 3]
                flat_t_cr_input = _flatten_matrix_tensor(T_cr_input)[:, :3, 3]
                flat_t_lr_input = _flatten_matrix_tensor(T_lr_input)[:, :3, 3]
                flat_t_cl_gt = _flatten_pose_tensor(gt_batch['T_CL_t_gt'])
                flat_t_cr_gt = _flatten_pose_tensor(gt_batch['T_CR_t_gt'])
                flat_t_lr_gt = _flatten_pose_tensor(gt_batch['T_LR_t_gt'])
                flat_q_cl_gt = _flatten_pose_tensor(gt_batch['T_CL_q_gt'])
                flat_q_cr_gt = _flatten_pose_tensor(gt_batch['T_CR_q_gt'])
                flat_q_lr_gt = _flatten_pose_tensor(gt_batch['T_LR_q_gt'])
                flat_t_cl_pred, flat_q_cl_pred, flat_T_cl_pred = _apply_delta_to_input(pred['T_CL_t'], pred['T_CL_q'], T_cl_input)
                flat_t_cr_pred, flat_q_cr_pred, flat_T_cr_pred = _apply_delta_to_input(pred['T_CR_t'], pred['T_CR_q'], T_cr_input)
                flat_t_lr_pred, flat_q_lr_pred, flat_T_lr_pred = _apply_delta_to_input(pred['T_LR_t'], pred['T_LR_q'], T_lr_input)

                total_input_t_cl += torch.norm(flat_t_cl_input - flat_t_cl_gt, dim=1).sum().item() * 100.0
                total_input_t_cr += torch.norm(flat_t_cr_input - flat_t_cr_gt, dim=1).sum().item() * 100.0
                total_input_t_lr += torch.norm(flat_t_lr_input - flat_t_lr_gt, dim=1).sum().item() * 100.0
                total_input_r_cl += (quaternion_distance(input_q_cl, flat_q_cl_gt, device) * 180.0 / math.pi).sum().item()
                total_input_r_cr += (quaternion_distance(input_q_cr, flat_q_cr_gt, device) * 180.0 / math.pi).sum().item()
                total_input_r_lr += (quaternion_distance(input_q_lr, flat_q_lr_gt, device) * 180.0 / math.pi).sum().item()

                total_val_t_cl += torch.norm(flat_t_cl_pred - flat_t_cl_gt, dim=1).sum().item() * 100.0
                total_val_t_cr += torch.norm(flat_t_cr_pred - flat_t_cr_gt, dim=1).sum().item() * 100.0
                total_val_t_lr += torch.norm(flat_t_lr_pred - flat_t_lr_gt, dim=1).sum().item() * 100.0
                total_val_r_cl += (quaternion_distance(flat_q_cl_pred, flat_q_cl_gt, device) * 180.0 / math.pi).sum().item()
                total_val_r_cr += (quaternion_distance(flat_q_cr_pred, flat_q_cr_gt, device) * 180.0 / math.pi).sum().item()
                total_val_r_lr += (quaternion_distance(flat_q_lr_pred, flat_q_lr_gt, device) * 180.0 / math.pi).sum().item()

                if batch_idx % _config['log_frequency'] == 0:
                    print(
                        f"[Tri Val] Iter {batch_idx}/{len(ValImgLoader)} "
                        f"loss={loss['total_loss'].item():.4f} "
                        f"CL={loss['loss_cl'].item():.4f} "
                        f"CR={loss['loss_cr'].item():.4f} "
                        f"LR={loss['loss_lr'].item():.4f} "
                        f"LOOP={loss['loss_loop'].item():.4f}"
                    )
                    if wandb_enabled and wandb is not None and _config.get('wandb_log_images', True):
                        show_idx = 0
                        time_idx = -1 if rgb.ndim == 5 else None
                        rgb_show = rgb[show_idx, time_idx].detach().cpu() if rgb.ndim == 5 else rgb[show_idx].detach().cpu()
                        lidar_show = lidar_proj[show_idx, time_idx].detach().cpu() if lidar_proj.ndim == 5 else lidar_proj[show_idx].detach().cpu()
                        radar_show = radar_proj[show_idx, time_idx].detach().cpu() if radar_proj.ndim == 5 else radar_proj[show_idx].detach().cpu()
                        radar_vis = radar_show[0:1, :, :]
                        if 'frame_index' in sample:
                            frame_index = int(sample['frame_index'][show_idx, time_idx].item()) if rgb.ndim == 5 else int(sample['frame_index'][show_idx].item())
                            base_dataset = ValImgLoader.dataset.base_dataset if hasattr(ValImgLoader.dataset, 'base_dataset') else ValImgLoader.dataset
                            dataset_item = base_dataset.all_files[frame_index]
                        else:
                            dataset_show_idx = batch_idx * _config['batch_size'] + show_idx
                            base_dataset = ValImgLoader.dataset.base_dataset if hasattr(ValImgLoader.dataset, 'base_dataset') else ValImgLoader.dataset
                            dataset_item = base_dataset.all_files[dataset_show_idx]
                        loaded_img = base_dataset._load_image(dataset_item['image_path'])
                        if isinstance(loaded_img, tuple):
                            rgb_img, calib = loaded_img
                        else:
                            rgb_img = loaded_img
                            scene_name = dataset_item.get('scene')
                            if hasattr(base_dataset, 'scene_info') and scene_name in base_dataset.scene_info:
                                calib = base_dataset.scene_info[scene_name]['K']
                            else:
                                raise ValueError("Could not resolve camera intrinsic for tri validation visualization.")
                        orig_hw = (rgb_img.height, rgb_img.width)
                        lidar_pc = _load_point_cloud(dataset_item['lidar_path'], base_dataset.pcd_reader)
                        radar_pc = _load_point_cloud(dataset_item['radar_path'], base_dataset.pcd_reader)

                        if rgb.ndim == 5:
                            pred_flat_index = show_idx * rgb.shape[1] + (time_idx % rgb.shape[1])
                            T_cl_pred = flat_T_cl_pred[pred_flat_index].detach().cpu().numpy().astype(np.float32)
                            T_cr_pred = flat_T_cr_pred[pred_flat_index].detach().cpu().numpy().astype(np.float32)
                            T_cl_gt = _pose_to_matrix(gt_batch['T_CL_t_gt'][show_idx, time_idx], gt_batch['T_CL_q_gt'][show_idx, time_idx])
                            T_cr_gt = _pose_to_matrix(gt_batch['T_CR_t_gt'][show_idx, time_idx], gt_batch['T_CR_q_gt'][show_idx, time_idx])
                        else:
                            T_cl_pred = flat_T_cl_pred[show_idx].detach().cpu().numpy().astype(np.float32)
                            T_cr_pred = flat_T_cr_pred[show_idx].detach().cpu().numpy().astype(np.float32)
                            T_cl_gt = _pose_to_matrix(gt_batch['T_CL_t_gt'][show_idx], gt_batch['T_CL_q_gt'][show_idx])
                            T_cr_gt = _pose_to_matrix(gt_batch['T_CR_t_gt'][show_idx], gt_batch['T_CR_q_gt'][show_idx])

                        lidar_depth_pred, _ = base_dataset._project_to_image(lidar_pc, T_cl_pred, calib, orig_hw)
                        radar_depth_pred, _ = base_dataset._project_to_image(radar_pc, T_cr_pred, calib, orig_hw)
                        lidar_depth_gt, _ = base_dataset._project_to_image(lidar_pc, T_cl_gt, calib, orig_hw)
                        radar_depth_gt, _ = base_dataset._project_to_image(radar_pc, T_cr_gt, calib, orig_hw)

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

        denom = max(total_eval_count, 1)
        val_loss = total_val_loss / denom
        input_t_cl = total_input_t_cl / denom
        input_t_cr = total_input_t_cr / denom
        input_t_lr = total_input_t_lr / denom
        input_r_cl = total_input_r_cl / denom
        input_r_cr = total_input_r_cr / denom
        input_r_lr = total_input_r_lr / denom
        val_t_cl = total_val_t_cl / denom
        val_t_cr = total_val_t_cr / denom
        val_t_lr = total_val_t_lr / denom
        val_r_cl = total_val_r_cl / denom
        val_r_cr = total_val_r_cr / denom
        val_r_lr = total_val_r_lr / denom
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

        if train_epoch_loss is not None:
            checkpoint_payload = {
                'config': _config,
                'epoch': epoch,
                'state_dict': model.module.state_dict() if isinstance(model, (nn.DataParallel, DDP)) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_epoch_loss,
                'val_loss': val_loss,
                'val_t_cl': val_t_cl,
                'val_t_cr': val_t_cr,
                'val_t_lr': val_t_lr,
                'val_r_cl': val_r_cl,
                'val_r_cr': val_r_cr,
                'val_r_lr': val_r_lr,
            }
            latest_savefilename = f'{model_savepath}/checkpoint_tri_latest.tar'
            torch.save(checkpoint_payload, latest_savefilename)
            print(f'Latest model saved as {latest_savefilename}')

            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                _run.result = val_loss
                savefilename = f'{model_savepath}/checkpoint_tri_best.tar'
                torch.save(checkpoint_payload, savefilename)
                print(f'Best model saved as {savefilename}')

        return val_loss

    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process:
            print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        total_train_count = 0
        local_loss = 0.
        _run.log_scalar("LR", scheduler.get_last_lr()[0])
        current_lr = optimizer.param_groups[0]['lr']
        _wandb_log(wandb_enabled, {"epoch": epoch, "train/lr": current_lr})

        model.train()
        freeze_bn_after = _config.get('tri_freeze_bn_after')
        if freeze_bn_after is not None and epoch >= int(freeze_bn_after):
            _set_batchnorm_eval(model)
            if is_main_process and epoch == int(freeze_bn_after):
                print(f"[Tri] BatchNorm running stats frozen from epoch {epoch}")
        for batch_idx, sample in enumerate(TrainImgLoader):
            rgb = _prepare_rgb_batch(sample['rgb'])
            lidar_proj, radar_proj = _build_tri_projection_batch(sample)
            gt_batch = {
                'T_CL_t_gt': sample['T_CL_t_gt'].to(device, non_blocking=True),
                'T_CL_q_gt': sample['T_CL_q_gt'].to(device, non_blocking=True),
                'T_CR_t_gt': sample['T_CR_t_gt'].to(device, non_blocking=True),
                'T_CR_q_gt': sample['T_CR_q_gt'].to(device, non_blocking=True),
                'T_LR_t_gt': sample['T_LR_t_gt'].to(device, non_blocking=True),
                'T_LR_q_gt': sample['T_LR_q_gt'].to(device, non_blocking=True),
                'T_CL_input': sample['T_CL_input'].to(device, non_blocking=True),
                'T_CR_input': sample['T_CR_input'].to(device, non_blocking=True),
            }
            gt_batch['T_LR_input'] = torch.linalg.inv(gt_batch['T_CL_input']) @ gt_batch['T_CR_input']

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=tri_amp_dtype, enabled=tri_amp_enabled):
                pred, _, _ = _tri_model_forward(model, rgb, lidar_proj, radar_proj, return_aux=False)
                loss = loss_fn(pred, gt_batch)
            if tri_use_grad_scaler:
                tri_grad_scaler.scale(loss['total_loss']).backward()
                tri_grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                tri_grad_scaler.step(optimizer)
                tri_grad_scaler.update()
            else:
                loss['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_item_count = _batch_item_count(rgb)
            total_train_loss += loss['total_loss'].item() * batch_item_count
            total_train_count += batch_item_count
            if is_main_process and batch_idx % _config['log_frequency'] == 0:
                print(
                    f"[Tri Train] Iter {batch_idx}/{len(TrainImgLoader)} "
                    f"loss={loss['total_loss'].item():.4f} "
                    f"CL={loss['loss_cl'].item():.4f} "
                    f"CR={loss['loss_cr'].item():.4f} "
                    f"LR={loss['loss_lr'].item():.4f} "
                    f"LOOP={loss['loss_loop'].item():.4f}"
                )

        if distributed:
            total_train_loss_tensor = torch.tensor(total_train_loss, device=device)
            total_train_count_tensor = torch.tensor(total_train_count, device=device)
            dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_train_count_tensor, op=dist.ReduceOp.SUM)
            total_train_loss = total_train_loss_tensor.item()
            total_train_count = int(total_train_count_tensor.item())
        train_epoch_loss = total_train_loss / max(total_train_count, 1)
        if is_main_process:
            print("------------------------------------")
            print(f"epoch {epoch} total training loss = {train_epoch_loss:.3f}")
            print(f"Total epoch time = {time.time() - epoch_start_time:.2f}")
            print("------------------------------------")
            _run.log_scalar("Total training loss", train_epoch_loss, epoch)
            _wandb_log(wandb_enabled, {"epoch": epoch, "train/epoch_loss": train_epoch_loss})
        should_run_val = ((epoch + 1) % 20 == 0) or ((epoch + 1) > (_config['epochs'] - 10))
        if should_run_val and is_main_process:
            run_tri_validation(epoch=epoch, train_epoch_loss=train_epoch_loss)
        scheduler.step()
        if distributed:
            dist.barrier()

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    if wandb_enabled and wandb is not None and wandb.run is not None:
        wandb.run.summary['full_training_time_hr'] = (time.time() - start_full_time) / 3600
        wandb.finish()
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return _run.result
