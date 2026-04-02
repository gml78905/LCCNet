import csv
import os
import re

import cv2
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose

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


class _DatasetLGInnotekBase(Dataset):
    sensor_folder = None
    pair_filename = None
    extrinsic_mode = None

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', train_scene=None,
                 val_scene=None, val_frame_limit=None, suf='.png'):
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

        intrinsic_path = os.path.join(dataset_dir, 'intrinsic.txt')
        extrinsic_path = os.path.join(dataset_dir, 'lg_init_extrinsics.yaml')
        self.K_raw, self.distortion = _load_intrinsic_file(intrinsic_path)
        self.K_raw = _scale_intrinsic_half(self.K_raw)
        self.extrinsic = _load_extrinsic_matrix(extrinsic_path, self.extrinsic_mode).astype(np.float32)

        self.pcd_reader = ReadOpen3d()
        self.all_files = []
        self.val_RT = []

        if split == 'train':
            selected_scenes = self.train_scene
        else:
            selected_scenes = self.val_scene

        for scene in selected_scenes:
            scene_root = os.path.join(dataset_dir, scene, 'offline', 'sensor_data')
            image_dir = os.path.join(scene_root, 'image_Cam0')
            sensor_dir = os.path.join(scene_root, self.sensor_folder)
            pair_file = os.path.join(dataset_dir, scene, 'offline', 'synced_stamps', self.pair_filename)
            if not os.path.exists(pair_file):
                raise FileNotFoundError(f"Pair file not found: {pair_file}")

            frame_limit = self.val_frame_limit if split in ['val', 'test'] else None
            scene_pairs = _load_pairs(pair_file, image_dir, sensor_dir, split, frame_limit)
            for pair in scene_pairs:
                pair['scene'] = scene
                self.all_files.append(pair)

        self._init_val_perturbations()

    def _init_val_perturbations(self):
        if self.split not in ['val', 'test']:
            return

        val_scene_name = '_'.join(self.val_scene) if isinstance(self.val_scene, list) else str(self.val_scene)
        limit_suffix = f"_first{self.val_frame_limit}" if self.val_frame_limit is not None else ""
        val_rt_file = os.path.join(
            self.root_dir,
            f"val_RT_{self.extrinsic_mode}_scene{val_scene_name}{limit_suffix}_{self.max_r:.2f}_{self.max_t:.2f}.csv"
        )

        if os.path.exists(val_rt_file):
            print(f'VAL SET: Using this file: {val_rt_file}')
            df_test_rt = pd.read_csv(val_rt_file, sep=',')
            for _, row in df_test_rt.iterrows():
                self.val_RT.append(list(row))
        else:
            print(f'VAL SET - Not found: {val_rt_file}')
            print('Generating a new one')
            with open(val_rt_file, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-self.max_t, self.max_t)
                    transl_y = np.random.uniform(-self.max_t, self.max_t)
                    transl_z = np.random.uniform(-self.max_t, self.max_t)
                    writer.writerow([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                        float(rotx), float(roty), float(rotz)])

        assert len(self.val_RT) == len(self.all_files), "Something wrong with validation RTs"

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

    def __len__(self):
        return len(self.all_files)

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

    def _prepare_point_cloud(self, sensor_path):
        pc = _load_point_cloud(sensor_path, self.pcd_reader)
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        filtered_pc = pc[valid_indices].copy()
        # Radar can be sparse enough that the KITTI-style center filter removes
        # every point; keep the original cloud in that case to avoid empty-point
        # losses producing NaNs.
        if filtered_pc.shape[0] > 0:
            pc = filtered_pc
        pc_org = torch.from_numpy(pc.astype(np.float32))

        if pc_org.shape[1] in [3, 4]:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_rot = np.matmul(self.extrinsic, pc_org.numpy()).astype(np.float32).copy()
        return torch.from_numpy(pc_rot)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        image_path = item['image_path']
        sensor_path = item['sensor_path']
        scene = item['scene']

        pc_in = self._prepare_point_cloud(sensor_path)
        try:
            img, calib = self._load_image(image_path)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)
        img_rotation = 0.

        try:
            img = self.custom_transform(img, img_rotation, False)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        if self.split == 'train':
            rotz = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            roty = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            rotx = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            initial_rt = None
        else:
            initial_rt = self.val_RT[idx]
            rotz = initial_rt[6]
            roty = initial_rt[5]
            rotx = initial_rt[4]
            transl_x = initial_rt[1]
            transl_y = initial_rt[2]
            transl_z = initial_rt[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))
        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)
        sample = {
            'rgb': img,
            'point_cloud': pc_in,
            'calib': calib,
            'tr_error': T,
            'rot_error': R,
            'scene': scene,
            'rgb_name': item['image_name'],
            'item': item['stamp'],
            'extrin': self.extrinsic.astype(np.float32),
        }
        if initial_rt is not None:
            sample['initial_RT'] = initial_rt
        return sample


class DatasetLidarCameraLGInnotek(_DatasetLGInnotekBase):
    sensor_folder = 'lidar_Hesai'
    pair_filename = 'image_Cam0_lidar_Hesai.txt'
    extrinsic_mode = 'lidar'


class DatasetCameraRadarLGInnotek(_DatasetLGInnotekBase):
    sensor_folder = 'radar_Continental'
    pair_filename = 'image_Cam0_radar_Continental.txt'
    extrinsic_mode = 'radar'
