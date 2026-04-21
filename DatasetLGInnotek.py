import csv
import os
import re
from collections import OrderedDict

import cv2
try:
    import mathutils
except ImportError:
    mathutils = None
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TTF
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, quaternion_from_matrix

import open3d as o3d


class ReadOpen3d:
    def __call__(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)


def _extract_floats(text):
    return [float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', text)]


def _load_intrinsic_file(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        content = f.read()

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    keyed_values = {}
    for line in lines:
        if ':' not in line and '=' not in line:
            continue
        key, value = re.split(r'[:=]', line, maxsplit=1)
        key = key.strip().lower()
        numbers = _extract_floats(value)
        if numbers:
            keyed_values[key] = numbers

    if all(key in keyed_values for key in ['fx', 'fy', 'cx', 'cy']):
        fx = keyed_values['fx'][0]
        fy = keyed_values['fy'][0]
        cx = keyed_values['cx'][0]
        cy = keyed_values['cy'][0]
        dist = []
        if 'distcoeffs' in keyed_values:
            dist.extend(keyed_values['distcoeffs'])
        for key in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']:
            if key in keyed_values:
                dist.append(keyed_values[key][0])
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return K, np.array(dist, dtype=np.float32)

    floats = _extract_floats(content)
    if len(floats) >= 9:
        first_nine = np.array(floats[:9], dtype=np.float32).reshape(3, 3)
        if np.isclose(first_nine[2, 2], 1.0, atol=1e-4):
            return first_nine, np.array(floats[9:17], dtype=np.float32)

    if len(floats) >= 4:
        fx, fy, cx, cy = floats[:4]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return K, np.array(floats[4:12], dtype=np.float32)

    raise ValueError(f"Could not parse intrinsic file: {intrinsic_path}")


def _scale_intrinsic_half(K):
    K = np.array(K, dtype=np.float32).copy()
    K[0, 0] *= 0.5
    K[1, 1] *= 0.5
    K[0, 2] *= 0.5
    K[1, 2] *= 0.5
    return K


def _compute_rectified_intrinsic(K, distortion, image_size):
    if distortion.size == 0:
        return np.array(K, dtype=np.float32).copy()
    width, height = image_size
    rectified_K, _ = cv2.getOptimalNewCameraMatrix(
        np.array(K, dtype=np.float32),
        np.array(distortion, dtype=np.float32),
        (width, height),
        0,
        (width, height),
    )
    return rectified_K.astype(np.float32)


def _euler_xyz_to_matrix(rotx, roty, rotz):
    cx, sx = np.cos(rotx), np.sin(rotx)
    cy, sy = np.cos(roty), np.sin(roty)
    cz, sz = np.cos(rotz), np.sin(rotz)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ], dtype=np.float32)
    Ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=np.float32)
    Rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return (Rx @ Ry @ Rz).astype(np.float32)


def _build_transform(tx, ty, tz, rotx, roty, rotz):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _euler_xyz_to_matrix(rotx, roty, rotz)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T


def _load_extrinsic_matrix(extrinsic_path, sensor_mode):
    with open(extrinsic_path, 'r') as f:
        data = yaml.safe_load(f)

    true_extrinsic = data.get('true_extrinsic', {}) if data else {}
    key = 'cam_to_lidar' if sensor_mode == 'lidar' else 'cam_to_radar'
    values = true_extrinsic.get(key)
    if values is None:
        raise KeyError(f"Missing true_extrinsic.{key} in {extrinsic_path}")

    values = np.array(values, dtype=np.float32).reshape(-1)
    if values.size == 16:
        matrix = values.reshape(4, 4)
        return matrix
    if values.size != 7:
        raise ValueError(f"Invalid extrinsic shape for {key}: {values.shape}")

    tx, ty, tz, qx, qy, qz, qw = values.tolist()
    q_norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_norm == 0:
        raise ValueError(f"Invalid quaternion norm for {key}")
    qx /= q_norm
    qy /= q_norm
    qz /= q_norm
    qw /= q_norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)
    matrix[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return matrix


def _index_files(directory):
    file_map = {}
    if not os.path.isdir(directory):
        return file_map

    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        file_map[stem] = path
        file_map[name] = path
        if stem.isdigit():
            file_map[str(int(stem))] = path
    return file_map


def _match_path(file_map, token):
    candidates = [token]
    stem = os.path.splitext(token)[0]
    if stem not in candidates:
        candidates.append(stem)
    if token.isdigit():
        candidates.append(str(int(token)))
    if stem.isdigit():
        candidates.append(str(int(stem)))

    for candidate in candidates:
        if candidate in file_map:
            return file_map[candidate]
    return None


def _load_pairs(pair_file, image_dir, sensor_dir, split, val_frame_limit=None):
    image_map = _index_files(image_dir)
    sensor_map = _index_files(sensor_dir)
    pairs = []

    with open(pair_file, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            tokens = re.split(r'[\s,]+', line)
            if len(tokens) < 2:
                continue

            # Skip the header row like: "image_Cam0 lidar_Hesai"
            if line_idx == 0 and (not tokens[0].isdigit() or not tokens[1].isdigit()):
                continue

            if not tokens[0].isdigit() or not tokens[1].isdigit():
                continue

            image_token, sensor_token = tokens[0], tokens[1]
            image_path = _match_path(image_map, image_token)
            sensor_path = _match_path(sensor_map, sensor_token)
            if image_path is None or sensor_path is None:
                continue

            pairs.append({
                'stamp': image_token,
                'image_path': image_path,
                'sensor_path': sensor_path,
                'image_name': os.path.basename(image_path),
            })

    if split in ['val', 'test'] and val_frame_limit is not None:
        pairs = pairs[:val_frame_limit]
    return pairs


def _load_triplets(pair_file, image_dir, lidar_dir, radar_dir, split, val_frame_limit=None):
    image_map = _index_files(image_dir)
    lidar_map = _index_files(lidar_dir)
    radar_map = _index_files(radar_dir)
    triplets = []

    with open(pair_file, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            tokens = re.split(r'[\s,]+', line)
            if len(tokens) < 3:
                continue

            # Skip header rows if present.
            if line_idx == 0 and (not tokens[0].isdigit() or not tokens[1].isdigit() or not tokens[2].isdigit()):
                continue
            if not tokens[0].isdigit() or not tokens[1].isdigit() or not tokens[2].isdigit():
                continue

            cam_token, lidar_token, radar_token = tokens[0], tokens[1], tokens[2]
            image_path = _match_path(image_map, cam_token)
            lidar_path = _match_path(lidar_map, lidar_token)
            radar_path = _match_path(radar_map, radar_token)
            if image_path is None or lidar_path is None or radar_path is None:
                continue

            triplets.append({
                'stamp': cam_token,
                'image_path': image_path,
                'lidar_path': lidar_path,
                'radar_path': radar_path,
                'image_name': os.path.basename(image_path),
            })

    if split in ['val', 'test'] and val_frame_limit is not None:
        triplets = triplets[:val_frame_limit]
    return triplets


def _load_point_cloud(file_path, pcd_reader):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pcd':
        points = pcd_reader(file_path)
    elif ext == '.bin':
        raw = np.fromfile(file_path, dtype=np.float32)
        if raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        elif raw.size % 5 == 0:
            points = raw.reshape(-1, 5)[:, :4]
        else:
            raise ValueError(f"Unsupported .bin point layout: {file_path}")
    elif ext in ['.npy', '.npz']:
        data = np.load(file_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            first_key = list(data.keys())[0]
            points = data[first_key]
        else:
            points = data
    elif ext in ['.txt', '.csv']:
        delimiter = ',' if ext == '.csv' else None
        points = np.loadtxt(file_path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported point cloud extension: {file_path}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] == 3:
        points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])
    elif points.shape[1] > 4:
        points = points[:, :4]
    return points


class DatasetTriModalLGInnotek(Dataset):
    """
    Minimal tri-modal dataset contract for baseline tri-modal calibration.
    Returns:
      rgb, lidar_proj, radar_proj,
      T_CL_t_gt, T_CL_q_gt,
      T_CR_t_gt, T_CR_q_gt,
      T_LR_t_gt, T_LR_q_gt
    """

    pair_filename = 'image_Cam0_lidar_Hesai_radar_Continental.txt'

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', train_scene=None,
                 val_scene=None, val_frame_limit=None, suf='.png', input_size=(288, 512), max_depth=80.0,
                 project_on_gpu=False):
        super().__init__()
        self.use_reflectance = use_reflectance
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.suf = suf
        self.train_scene = train_scene or []
        self.val_scene = val_scene or []
        self.val_frame_limit = val_frame_limit
        self.input_size = input_size
        self.max_depth = float(max_depth)
        self.project_on_gpu = bool(project_on_gpu)

        intrinsic_path = os.path.join(dataset_dir, 'intrinsic.txt')
        extrinsic_path = os.path.join(dataset_dir, 'lg_init_extrinsics.yaml')
        self.K_raw, self.distortion = _load_intrinsic_file(intrinsic_path)
        self.K_raw = _scale_intrinsic_half(self.K_raw)
        self.T_cam_lidar = _load_extrinsic_matrix(extrinsic_path, 'lidar').astype(np.float32)
        self.T_cam_radar = _load_extrinsic_matrix(extrinsic_path, 'radar').astype(np.float32)
        self.T_lidar_radar = np.linalg.inv(self.T_cam_lidar) @ self.T_cam_radar

        self.pcd_reader = ReadOpen3d()
        self.all_files = []
        self.val_RT_lidar = []
        self.val_RT_radar = []

        if split == 'train':
            selected_scenes = self.train_scene
        else:
            selected_scenes = self.val_scene

        for scene in selected_scenes:
            scene_root = os.path.join(dataset_dir, scene, 'offline', 'sensor_data')
            image_dir = os.path.join(scene_root, 'image_Cam0')
            lidar_dir = os.path.join(scene_root, 'lidar_Hesai')
            radar_dir = os.path.join(scene_root, 'radar_Continental')
            pair_file = os.path.join(dataset_dir, scene, 'offline', 'synced_stamps', self.pair_filename)
            if not os.path.exists(pair_file):
                raise FileNotFoundError(f"Triplet pair file not found: {pair_file}")

            frame_limit = self.val_frame_limit if split in ['val', 'test'] else None
            scene_triplets = _load_triplets(pair_file, image_dir, lidar_dir, radar_dir, split, frame_limit)
            for triplet in scene_triplets:
                triplet['scene'] = scene
                self.all_files.append(triplet)

        self._init_val_perturbations()

    def __len__(self):
        return len(self.all_files)

    def _sample_pose_error(self, rng=None):
        if rng is None:
            uniform = np.random.uniform
        else:
            uniform = rng.uniform
        rotz = uniform(-self.max_r, self.max_r) * (np.pi / 180.0)
        roty = uniform(-self.max_r, self.max_r) * (np.pi / 180.0)
        rotx = uniform(-self.max_r, self.max_r) * (np.pi / 180.0)
        tx = uniform(-self.max_t, self.max_t)
        ty = uniform(-self.max_t, self.max_t)
        tz = uniform(-self.max_t, self.max_t)
        return tx, ty, tz, rotx, roty, rotz

    def _init_val_perturbations(self):
        if self.split not in ['val', 'test']:
            return
        lidar_rng = np.random.RandomState(0)
        radar_rng = np.random.RandomState(1)
        for _ in range(len(self.all_files)):
            self.val_RT_lidar.append(self._sample_pose_error(lidar_rng))
            self.val_RT_radar.append(self._sample_pose_error(radar_rng))

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def _load_image(self, image_path):
        try:
            with Image.open(image_path) as image:
                image_rgb = image.convert('RGB')
                image_rgb = np.array(image_rgb)
        except (OSError, ValueError):
            raise OSError(f"Image not found or unreadable: {image_path}")
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        height, width = image_bgr.shape[:2]
        if self.distortion.size > 0:
            calib = _compute_rectified_intrinsic(self.K_raw, self.distortion, (width, height))
            image_bgr = cv2.undistort(image_bgr, self.K_raw, self.distortion, None, calib)
        else:
            calib = self.K_raw.copy()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb), calib

    def _project_to_image(self, pc_sensor, T_cam_sensor, calib, image_hw):
        if pc_sensor.shape[0] == 0:
            h, w = image_hw
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        xyz1 = np.ones((pc_sensor.shape[0], 4), dtype=np.float32)
        xyz1[:, :3] = pc_sensor[:, :3]
        pc_cam = (T_cam_sensor @ xyz1.T).T

        x = pc_cam[:, 0]
        y = pc_cam[:, 1]
        z = pc_cam[:, 2]
        aux = pc_sensor[:, 3] if pc_sensor.shape[1] > 3 else np.zeros_like(z)
        h, w = image_hw

        valid = z > 1e-5
        valid = valid & (z < self.max_depth)
        if not np.any(valid):
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        x = x[valid]
        y = y[valid]
        z = z[valid]
        aux = aux[valid]

        u = (calib[0, 0] * x / z) + calib[0, 2]
        v = (calib[1, 1] * y / z) + calib[1, 2]

        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        u = u[in_img]
        v = v[in_img]
        z = z[in_img]
        aux = aux[in_img]

        depth = np.zeros((h, w), dtype=np.float32)
        aux_map = np.zeros((h, w), dtype=np.float32)
        depth[v, u] = z
        aux_map[v, u] = aux
        depth /= self.max_depth
        return depth, aux_map

    @staticmethod
    def _matrix_to_t_q(T):
        T_torch = torch.from_numpy(T.astype(np.float32))
        t = T_torch[:3, 3].clone()
        q = quaternion_from_matrix(T_torch)
        return t, q

    def __getitem__(self, idx):
        item = self.all_files[idx]
        image_path = item['image_path']
        lidar_path = item['lidar_path']
        radar_path = item['radar_path']

        try:
            img, calib = self._load_image(image_path)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        rgb = self.custom_transform(img, img_rotation=0., flip=False)
        if self.input_size is not None:
            rgb = F.interpolate(rgb.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)

        lidar_pc = _load_point_cloud(lidar_path, self.pcd_reader)
        radar_pc = _load_point_cloud(radar_path, self.pcd_reader)
        image_hw = (img.height, img.width)

        if self.split == 'train':
            lidar_rt = self._sample_pose_error()
            radar_rt = self._sample_pose_error()
        else:
            lidar_rt = self.val_RT_lidar[idx]
            radar_rt = self.val_RT_radar[idx]

        T_err_lidar = _build_transform(*lidar_rt)
        T_err_radar = _build_transform(*radar_rt)
        T_cam_lidar_input = (T_err_lidar @ self.T_cam_lidar).astype(np.float32)
        T_cam_radar_input = (T_err_radar @ self.T_cam_radar).astype(np.float32)

        T_CL_t_gt, T_CL_q_gt = self._matrix_to_t_q(self.T_cam_lidar)
        T_CR_t_gt, T_CR_q_gt = self._matrix_to_t_q(self.T_cam_radar)
        T_LR_t_gt, T_LR_q_gt = self._matrix_to_t_q(self.T_lidar_radar)

        result = {
            'rgb': rgb.float(),
            'T_CL_t_gt': T_CL_t_gt.float(),
            'T_CL_q_gt': T_CL_q_gt.float(),
            'T_CR_t_gt': T_CR_t_gt.float(),
            'T_CR_q_gt': T_CR_q_gt.float(),
            'T_LR_t_gt': T_LR_t_gt.float(),
            'T_LR_q_gt': T_LR_q_gt.float(),
            'T_CL_input': torch.from_numpy(T_cam_lidar_input),
            'T_CR_input': torch.from_numpy(T_cam_radar_input),
        }
        if self.project_on_gpu:
            result['lidar_pc'] = torch.from_numpy(lidar_pc.astype(np.float32))
            result['radar_pc'] = torch.from_numpy(radar_pc.astype(np.float32))
            result['calib'] = torch.from_numpy(calib.astype(np.float32))
            result['image_hw'] = torch.tensor(image_hw, dtype=torch.long)
        else:
            lidar_depth, _ = self._project_to_image(lidar_pc, T_cam_lidar_input, calib, image_hw)
            radar_depth, radar_aux = self._project_to_image(radar_pc, T_cam_radar_input, calib, image_hw)

            lidar_proj = torch.from_numpy(lidar_depth).unsqueeze(0)
            radar_proj = torch.from_numpy(np.stack([radar_depth, radar_aux], axis=0))
            if self.input_size is not None:
                lidar_proj = F.interpolate(lidar_proj.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)
                radar_proj = F.interpolate(radar_proj.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)
            result['lidar_proj'] = lidar_proj.float()
            result['radar_proj'] = radar_proj.float()
        return result


class DatasetTriModalHercules(Dataset):
    """
    Tri-modal Hercules dataset with the same output contract as DatasetTriModalLGInnotek.
    Expects per-scene camera/lidar/radar folders and calibration.yaml.
    """

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', train_scene=None,
                 val_scene=None, val_frame_limit=None, suf='.png', input_size=(288, 512), max_depth=80.0,
                 project_on_gpu=False):
        super().__init__()
        self.use_reflectance = use_reflectance
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.suf = suf
        self.input_size = input_size
        self.max_depth = float(max_depth)
        self.project_on_gpu = bool(project_on_gpu)
        self.val_frame_limit = val_frame_limit

        # scene -> {"data_dir", "K", "T_cam_lidar", "T_cam_radar", "T_lidar_radar"}
        self.scene_info = {}
        self.all_files = []
        self.pcd_reader = ReadOpen3d()
        self.val_RT_lidar = []
        self.val_RT_radar = []
        self.global_calib = self._load_global_calibration(dataset_dir)

        scene_data_dirs = self._discover_scene_data_dirs(dataset_dir)
        if len(scene_data_dirs) == 0:
            raise ValueError(f"No Hercules scenes with calibration.yaml found in: {dataset_dir}")

        if val_scene is None:
            val_scene = [sorted(scene_data_dirs.keys())[0]]
        if isinstance(val_scene, str):
            val_scene = [val_scene]

        if train_scene is not None and isinstance(train_scene, str):
            train_scene = [train_scene]
        if train_scene is not None:
            train_scene = [s for s in train_scene if s not in val_scene]

        if split == 'train':
            if train_scene is None:
                selected_scenes = [s for s in scene_data_dirs.keys() if s not in val_scene]
            else:
                selected_scenes = [s for s in train_scene if s in scene_data_dirs]
        else:
            selected_scenes = [s for s in val_scene if s in scene_data_dirs]

        for scene in selected_scenes:
            data_dir = scene_data_dirs[scene]
            calib_path = os.path.join(data_dir, 'calibration.yaml')
            calib_data = {}
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    calib_data = yaml.safe_load(f) or {}

            K = self._extract_intrinsic(calib_data, self.global_calib)
            T_cam_lidar = self._extract_extrinsic(calib_data, self.global_calib, kind='lidar')
            T_cam_radar = self._extract_extrinsic(calib_data, self.global_calib, kind='radar')
            T_lidar_radar = np.linalg.inv(T_cam_lidar) @ T_cam_radar

            self.scene_info[scene] = {
                'data_dir': data_dir,
                'K': K.astype(np.float32),
                'T_cam_lidar': T_cam_lidar.astype(np.float32),
                'T_cam_radar': T_cam_radar.astype(np.float32),
                'T_lidar_radar': T_lidar_radar.astype(np.float32),
            }

            triplets = self._collect_triplets(scene, data_dir)
            if split in ['val', 'test'] and self.val_frame_limit is not None:
                triplets = triplets[:self.val_frame_limit]
            self.all_files.extend(triplets)

        if len(self.all_files) == 0:
            raise ValueError(f"No tri-modal matched samples found for split='{split}' in Hercules dataset.")

        self._init_val_perturbations()

    @staticmethod
    def _discover_scene_data_dirs(dataset_dir):
        scene_data_dirs = {}
        for scene in sorted(os.listdir(dataset_dir)):
            scene_path = os.path.join(dataset_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            calib_path = os.path.join(scene_path, 'calibration.yaml')
            if os.path.exists(calib_path):
                scene_data_dirs[scene] = scene_path
                continue
            subdir_path = os.path.join(scene_path, 'CMRNext')
            if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, 'calibration.yaml')):
                scene_data_dirs[scene] = subdir_path
                continue

            # Fallback: if scene has raw sensor folders, allow global calibration usage.
            camera_dir = os.path.join(scene_path, 'camera')
            lidar_dir = os.path.join(scene_path, 'lidar')
            radar_dir = os.path.join(scene_path, 'radar')
            if os.path.isdir(camera_dir) and os.path.isdir(lidar_dir) and os.path.isdir(radar_dir):
                scene_data_dirs[scene] = scene_path
        return scene_data_dirs

    @staticmethod
    def _load_global_calibration(dataset_dir):
        for name in ['rlc_calibration.yaml', 'calibration.yaml', 'calibration.yaml.save']:
            path = os.path.join(dataset_dir, name)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _extract_intrinsic(calib_data, global_calib=None):
        if 'camera' in calib_data and isinstance(calib_data['camera'], dict) and 'intrinsic' in calib_data['camera']:
            K = np.array(calib_data['camera']['intrinsic'], dtype=np.float32)
            if K.shape == (3, 3):
                return K
        if all(k in calib_data for k in ['fx', 'fy', 'cx', 'cy']):
            fx = float(calib_data['fx'])
            fy = float(calib_data['fy'])
            cx = float(calib_data['cx'])
            cy = float(calib_data['cy'])
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        if global_calib is not None and len(global_calib) > 0:
            if 'camera' in global_calib and isinstance(global_calib['camera'], dict) and 'intrinsic' in global_calib['camera']:
                K = np.array(global_calib['camera']['intrinsic'], dtype=np.float32)
                if K.shape == (3, 3):
                    return K
            if all(k in global_calib for k in ['fx', 'fy', 'cx', 'cy']):
                fx = float(global_calib['fx'])
                fy = float(global_calib['fy'])
                cx = float(global_calib['cx'])
                cy = float(global_calib['cy'])
                return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        raise ValueError("No valid camera intrinsic found in calibration.yaml")

    @staticmethod
    def _as_matrix4(value):
        arr = np.array(value, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
        return None

    @staticmethod
    def _extract_extrinsic(calib_data, global_calib, kind):
        extr = calib_data.get('extrinsic', {}) if isinstance(calib_data, dict) else {}
        keys = ['T_cam_lidar', 'T_cam2_velo'] if kind == 'lidar' else ['T_cam_radar']
        for key in keys:
            if key in extr:
                mat = DatasetTriModalHercules._as_matrix4(extr[key])
                if mat is not None:
                    return mat

        # Fallback for root rlc_calibration.yaml style keys.
        if isinstance(global_calib, dict):
            if kind == 'lidar':
                alt_keys = ['cam_to_lid_extrinsics', 'cam_to_lidar_extrinsics']
            else:
                alt_keys = ['cam_to_rad_extrinsics', 'cam_to_radar_extrinsics']
            for key in alt_keys:
                if key in global_calib:
                    mat = DatasetTriModalHercules._as_matrix4(global_calib[key])
                    if mat is not None:
                        return mat
        raise ValueError(f"No valid camera->{kind} extrinsic matrix found in calibration.yaml")

    @staticmethod
    def _collect_triplets(scene, data_dir):
        camera_dir = os.path.join(data_dir, 'camera')
        lidar_dir = os.path.join(data_dir, 'lidar')
        radar_dir = os.path.join(data_dir, 'radar')
        if not (os.path.isdir(camera_dir) and os.path.isdir(lidar_dir) and os.path.isdir(radar_dir)):
            return []

        image_map = {}
        for name in os.listdir(camera_dir):
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(name)[0]
                image_map[stem] = os.path.join(camera_dir, name)
        lidar_map = {}
        for name in os.listdir(lidar_dir):
            if name.lower().endswith('.pcd'):
                stem = os.path.splitext(name)[0]
                lidar_map[stem] = os.path.join(lidar_dir, name)
        radar_map = {}
        for name in os.listdir(radar_dir):
            if name.lower().endswith('.pcd'):
                stem = os.path.splitext(name)[0]
                radar_map[stem] = os.path.join(radar_dir, name)

        common_stems = sorted(set(image_map.keys()) & set(lidar_map.keys()) & set(radar_map.keys()))
        triplets = []
        for stem in common_stems:
            triplets.append({
                'scene': scene,
                'stamp': stem,
                'image_path': image_map[stem],
                'lidar_path': lidar_map[stem],
                'radar_path': radar_map[stem],
                'image_name': os.path.basename(image_map[stem]),
            })
        return triplets

    def __len__(self):
        return len(self.all_files)

    def _sample_pose_error(self, rng=None):
        if rng is None:
            uniform = np.random.uniform
        else:
            uniform = rng.uniform
        rotz = uniform(-self.max_r, self.max_r) * (np.pi / 180.0)
        roty = uniform(-self.max_r, self.max_r) * (np.pi / 180.0)
        rotx = uniform(-self.max_r, self.max_r) * (np.pi / 180.0)
        tx = uniform(-self.max_t, self.max_t)
        ty = uniform(-self.max_t, self.max_t)
        tz = uniform(-self.max_t, self.max_t)
        return tx, ty, tz, rotx, roty, rotz

    def _init_val_perturbations(self):
        if self.split not in ['val', 'test']:
            return
        lidar_rng = np.random.RandomState(0)
        radar_rng = np.random.RandomState(1)
        for _ in range(len(self.all_files)):
            self.val_RT_lidar.append(self._sample_pose_error(lidar_rng))
            self.val_RT_radar.append(self._sample_pose_error(radar_rng))

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def _load_image(self, image_path):
        try:
            with Image.open(image_path) as image:
                image_rgb = image.convert('RGB')
                image_rgb = np.array(image_rgb)
        except (OSError, ValueError):
            raise OSError(f"Image not found or unreadable: {image_path}")
        return Image.fromarray(image_rgb)

    def _project_to_image(self, pc_sensor, T_cam_sensor, calib, image_hw):
        if pc_sensor.shape[0] == 0:
            h, w = image_hw
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        xyz1 = np.ones((pc_sensor.shape[0], 4), dtype=np.float32)
        xyz1[:, :3] = pc_sensor[:, :3]
        pc_cam = (T_cam_sensor @ xyz1.T).T

        x = pc_cam[:, 0]
        y = pc_cam[:, 1]
        z = pc_cam[:, 2]
        aux = pc_sensor[:, 3] if pc_sensor.shape[1] > 3 else np.zeros_like(z)
        h, w = image_hw

        valid = (z > 1e-5) & (z < self.max_depth)
        if not np.any(valid):
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        x = x[valid]
        y = y[valid]
        z = z[valid]
        aux = aux[valid]

        u = (calib[0, 0] * x / z) + calib[0, 2]
        v = (calib[1, 1] * y / z) + calib[1, 2]

        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        u = u[in_img]
        v = v[in_img]
        z = z[in_img]
        aux = aux[in_img]

        depth = np.zeros((h, w), dtype=np.float32)
        aux_map = np.zeros((h, w), dtype=np.float32)
        depth[v, u] = z
        aux_map[v, u] = aux
        depth /= self.max_depth
        return depth, aux_map

    @staticmethod
    def _matrix_to_t_q(T):
        T_torch = torch.from_numpy(T.astype(np.float32))
        t = T_torch[:3, 3].clone()
        q = quaternion_from_matrix(T_torch)
        return t, q

    def __getitem__(self, idx):
        item = self.all_files[idx]
        scene = item['scene']
        scene_info = self.scene_info[scene]
        image_path = item['image_path']
        lidar_path = item['lidar_path']
        radar_path = item['radar_path']

        try:
            img = self._load_image(image_path)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        rgb = self.custom_transform(img, img_rotation=0., flip=False)
        if self.input_size is not None:
            rgb = F.interpolate(rgb.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)

        lidar_pc = _load_point_cloud(lidar_path, self.pcd_reader)
        radar_pc = _load_point_cloud(radar_path, self.pcd_reader)
        image_hw = (img.height, img.width)
        calib = scene_info['K']

        if self.split == 'train':
            lidar_rt = self._sample_pose_error()
            radar_rt = self._sample_pose_error()
        else:
            lidar_rt = self.val_RT_lidar[idx]
            radar_rt = self.val_RT_radar[idx]

        T_err_lidar = _build_transform(*lidar_rt)
        T_err_radar = _build_transform(*radar_rt)
        T_cam_lidar_input = (T_err_lidar @ scene_info['T_cam_lidar']).astype(np.float32)
        T_cam_radar_input = (T_err_radar @ scene_info['T_cam_radar']).astype(np.float32)

        T_CL_t_gt, T_CL_q_gt = self._matrix_to_t_q(scene_info['T_cam_lidar'])
        T_CR_t_gt, T_CR_q_gt = self._matrix_to_t_q(scene_info['T_cam_radar'])
        T_LR_t_gt, T_LR_q_gt = self._matrix_to_t_q(scene_info['T_lidar_radar'])

        result = {
            'rgb': rgb.float(),
            'T_CL_t_gt': T_CL_t_gt.float(),
            'T_CL_q_gt': T_CL_q_gt.float(),
            'T_CR_t_gt': T_CR_t_gt.float(),
            'T_CR_q_gt': T_CR_q_gt.float(),
            'T_LR_t_gt': T_LR_t_gt.float(),
            'T_LR_q_gt': T_LR_q_gt.float(),
            'T_CL_input': torch.from_numpy(T_cam_lidar_input),
            'T_CR_input': torch.from_numpy(T_cam_radar_input),
        }
        if self.project_on_gpu:
            result['lidar_pc'] = torch.from_numpy(lidar_pc.astype(np.float32))
            result['radar_pc'] = torch.from_numpy(radar_pc.astype(np.float32))
            result['calib'] = torch.from_numpy(calib.astype(np.float32))
            result['image_hw'] = torch.tensor(image_hw, dtype=torch.long)
        else:
            lidar_depth, _ = self._project_to_image(lidar_pc, T_cam_lidar_input, calib, image_hw)
            radar_depth, radar_aux = self._project_to_image(radar_pc, T_cam_radar_input, calib, image_hw)

            lidar_proj = torch.from_numpy(lidar_depth).unsqueeze(0)
            radar_proj = torch.from_numpy(np.stack([radar_depth, radar_aux], axis=0))
            if self.input_size is not None:
                lidar_proj = F.interpolate(lidar_proj.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)
                radar_proj = F.interpolate(radar_proj.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)
            result['lidar_proj'] = lidar_proj.float()
            result['radar_proj'] = radar_proj.float()
        return result


class TriSequenceDataset(Dataset):
    """
    Wrap a tri-modal frame dataset and expose sliding-window frame sequences.
    Each sequence shares one sampled extrinsic perturbation for LiDAR and radar.
    """

    def __init__(self, base_dataset, seq_len=4, stride=1, cache_size=8, project_on_gpu=None):
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.base_dataset = base_dataset
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.cache_size = max(int(cache_size), 0)
        self.project_on_gpu = self.base_dataset.project_on_gpu if project_on_gpu is None else bool(project_on_gpu)
        self._frame_cache = OrderedDict()
        self.sequence_indices = []
        self.val_RT_lidar_seq = []
        self.val_RT_radar_seq = []

        scene_to_indices = {}
        for idx, item in enumerate(self.base_dataset.all_files):
            scene = item.get('scene', 'default')
            scene_to_indices.setdefault(scene, []).append(idx)

        for scene in scene_to_indices:
            scene_to_indices[scene] = sorted(
                scene_to_indices[scene],
                key=lambda i: self.base_dataset.all_files[i].get('stamp', str(i))
            )
            scene_indices = scene_to_indices[scene]
            if len(scene_indices) < self.seq_len:
                continue
            for start in range(0, len(scene_indices) - self.seq_len + 1, self.stride):
                self.sequence_indices.append(scene_indices[start:start + self.seq_len])

        if self.base_dataset.split in ['val', 'test']:
            lidar_rng = np.random.RandomState(1000)
            radar_rng = np.random.RandomState(1001)
            for _ in range(len(self.sequence_indices)):
                self.val_RT_lidar_seq.append(self.base_dataset._sample_pose_error(lidar_rng))
                self.val_RT_radar_seq.append(self.base_dataset._sample_pose_error(radar_rng))

    def __len__(self):
        return len(self.sequence_indices)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def _cache_get(self, frame_idx):
        if self.cache_size <= 0:
            return None
        cached = self._frame_cache.get(frame_idx)
        if cached is not None:
            self._frame_cache.move_to_end(frame_idx)
        return cached

    def _cache_put(self, frame_idx, frame_data):
        if self.cache_size <= 0:
            return
        self._frame_cache[frame_idx] = frame_data
        self._frame_cache.move_to_end(frame_idx)
        while len(self._frame_cache) > self.cache_size:
            self._frame_cache.popitem(last=False)

    def _get_cached_frame_assets(self, frame_idx):
        cached = self._cache_get(frame_idx)
        if cached is not None:
            return cached

        item = self.base_dataset.all_files[frame_idx]
        image_path = item['image_path']
        lidar_path = item['lidar_path']
        radar_path = item['radar_path']

        try:
            loaded = self.base_dataset._load_image(image_path)
        except OSError:
            new_frame_idx = np.random.randint(0, len(self.base_dataset.all_files))
            return self._get_cached_frame_assets(new_frame_idx)

        if isinstance(loaded, tuple):
            img, calib = loaded
        else:
            img = loaded
            scene = item.get('scene')
            calib = self.base_dataset.scene_info[scene]['K']

        rgb = self.base_dataset.custom_transform(img, img_rotation=0., flip=False)
        if self.base_dataset.input_size is not None:
            rgb = F.interpolate(
                rgb.unsqueeze(0),
                size=self.base_dataset.input_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        if hasattr(self.base_dataset, 'scene_info'):
            scene_info = self.base_dataset.scene_info[item['scene']]
            T_cam_lidar_gt = scene_info['T_cam_lidar']
            T_cam_radar_gt = scene_info['T_cam_radar']
            T_lidar_radar_gt = scene_info['T_lidar_radar']
        else:
            T_cam_lidar_gt = self.base_dataset.T_cam_lidar
            T_cam_radar_gt = self.base_dataset.T_cam_radar
            T_lidar_radar_gt = self.base_dataset.T_lidar_radar

        frame_data = {
            'item': item,
            'rgb': rgb.float(),
            'calib': calib,
            'image_hw': (img.height, img.width),
            'lidar_pc': _load_point_cloud(lidar_path, self.base_dataset.pcd_reader),
            'radar_pc': _load_point_cloud(radar_path, self.base_dataset.pcd_reader),
            'T_cam_lidar_gt': T_cam_lidar_gt,
            'T_cam_radar_gt': T_cam_radar_gt,
            'T_lidar_radar_gt': T_lidar_radar_gt,
        }
        self._cache_put(frame_idx, frame_data)
        return frame_data

    def _load_frame(self, frame_idx, lidar_rt, radar_rt):
        frame_assets = self._get_cached_frame_assets(frame_idx)
        item = frame_assets['item']
        rgb = frame_assets['rgb']
        calib = frame_assets['calib']
        lidar_pc = frame_assets['lidar_pc']
        radar_pc = frame_assets['radar_pc']
        image_hw = frame_assets['image_hw']

        T_err_lidar = _build_transform(*lidar_rt)
        T_err_radar = _build_transform(*radar_rt)
        T_cam_lidar_gt = frame_assets['T_cam_lidar_gt']
        T_cam_radar_gt = frame_assets['T_cam_radar_gt']
        T_lidar_radar_gt = frame_assets['T_lidar_radar_gt']

        T_cam_lidar_input = (T_err_lidar @ T_cam_lidar_gt).astype(np.float32)
        T_cam_radar_input = (T_err_radar @ T_cam_radar_gt).astype(np.float32)

        T_CL_t_gt, T_CL_q_gt = self.base_dataset._matrix_to_t_q(T_cam_lidar_gt)
        T_CR_t_gt, T_CR_q_gt = self.base_dataset._matrix_to_t_q(T_cam_radar_gt)
        T_LR_t_gt, T_LR_q_gt = self.base_dataset._matrix_to_t_q(T_lidar_radar_gt)

        result = {
            'rgb': rgb.float(),
            'T_CL_t_gt': T_CL_t_gt.float(),
            'T_CL_q_gt': T_CL_q_gt.float(),
            'T_CR_t_gt': T_CR_t_gt.float(),
            'T_CR_q_gt': T_CR_q_gt.float(),
            'T_LR_t_gt': T_LR_t_gt.float(),
            'T_LR_q_gt': T_LR_q_gt.float(),
            'T_CL_input': torch.from_numpy(T_cam_lidar_input),
            'T_CR_input': torch.from_numpy(T_cam_radar_input),
            'frame_index': torch.tensor(frame_idx, dtype=torch.long),
        }
        if self.project_on_gpu:
            result['lidar_pc'] = torch.from_numpy(lidar_pc.astype(np.float32))
            result['radar_pc'] = torch.from_numpy(radar_pc.astype(np.float32))
            result['calib'] = torch.from_numpy(calib.astype(np.float32))
            result['image_hw'] = torch.tensor(image_hw, dtype=torch.long)
        else:
            lidar_depth, _ = self.base_dataset._project_to_image(lidar_pc, T_cam_lidar_input, calib, image_hw)
            radar_depth, radar_aux = self.base_dataset._project_to_image(radar_pc, T_cam_radar_input, calib, image_hw)

            lidar_proj = torch.from_numpy(lidar_depth).unsqueeze(0)
            radar_proj = torch.from_numpy(np.stack([radar_depth, radar_aux], axis=0))
            if self.base_dataset.input_size is not None:
                lidar_proj = F.interpolate(
                    lidar_proj.unsqueeze(0),
                    size=self.base_dataset.input_size,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
                radar_proj = F.interpolate(
                    radar_proj.unsqueeze(0),
                    size=self.base_dataset.input_size,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            result['lidar_proj'] = lidar_proj.float()
            result['radar_proj'] = radar_proj.float()
        return result

    def __getitem__(self, idx):
        if self.base_dataset.split == 'train':
            lidar_rt = self.base_dataset._sample_pose_error()
            radar_rt = self.base_dataset._sample_pose_error()
        else:
            lidar_rt = self.val_RT_lidar_seq[idx]
            radar_rt = self.val_RT_radar_seq[idx]

        frames = []
        for frame_idx in self.sequence_indices[idx]:
            frame = self._load_frame(frame_idx, lidar_rt, radar_rt)
            if isinstance(frame, dict) and 'frame_index' in frame:
                frames.append(frame)
            else:
                return frame

        keys = [
            'rgb',
            'T_CL_t_gt', 'T_CL_q_gt',
            'T_CR_t_gt', 'T_CR_q_gt',
            'T_LR_t_gt', 'T_LR_q_gt',
            'T_CL_input', 'T_CR_input', 'frame_index',
        ]
        out = {key: torch.stack([frame[key] for frame in frames], dim=0) for key in keys}
        if self.project_on_gpu:
            out['lidar_pc_seq'] = [frame['lidar_pc'] for frame in frames]
            out['radar_pc_seq'] = [frame['radar_pc'] for frame in frames]
            out['calib_seq'] = torch.stack([frame['calib'] for frame in frames], dim=0)
            out['image_hw_seq'] = torch.stack([frame['image_hw'] for frame in frames], dim=0)
        else:
            out['lidar_proj'] = torch.stack([frame['lidar_proj'] for frame in frames], dim=0)
            out['radar_proj'] = torch.stack([frame['radar_proj'] for frame in frames], dim=0)
        return out
