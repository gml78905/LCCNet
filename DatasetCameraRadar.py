# -------------------------------------------------------------------
# Dataset for Hercules Camera-Radar Calibration
# -------------------------------------------------------------------

import csv
import os
from math import radians
import cv2
import yaml
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, rotate_forward, quaternion_from_matrix
from quaternion_distances import quaternion_distance
import open3d as o3d

class ReadOpen3d:
    def __call__(self, file):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        return points

class DatasetCameraRadarHercules(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_scene=None, train_scene=None, suf='.png'):
        super(DatasetCameraRadarHercules, self).__init__()
        self.use_reflectance = use_reflectance
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.suf = suf
        
        self.GTs_T_cam_radar = {}  # Ground truth transformation from camera to radar
        self.K = {}  # Camera intrinsic parameters
        self.scene_data_dirs = {}  # {scene: actual_data_dir} - stores where calibration.yaml is located
        
        self.all_files = []
        
        # Get all scene directories
        scene_list = [d for d in os.listdir(dataset_dir) 
                     if os.path.isdir(os.path.join(dataset_dir, d))]
        scene_list.sort()
        
        # Find calibration.yaml - could be directly in scene or in subfolder
        for scene in scene_list:
            scene_path = os.path.join(dataset_dir, scene)
            data_dir = None
            
            # Check if calibration.yaml is directly in scene folder
            calib_file = os.path.join(scene_path, 'calibration.yaml')
            if os.path.exists(calib_file):
                data_dir = scene_path
            else:
                subdir_path = os.path.join(scene_path, 'CMRNext')
                if os.path.isdir(subdir_path):
                    calib_file = os.path.join(subdir_path, 'calibration.yaml')
                    if os.path.exists(calib_file):
                            data_dir = subdir_path
            
            if data_dir is None:
                continue
            
            self.scene_data_dirs[scene] = data_dir
        
        if val_scene is None:
            val_scene = list(self.scene_data_dirs.keys())[0] if len(self.scene_data_dirs) > 0 else None
        
        # Convert single scene to list if needed
        if isinstance(val_scene, str):
            val_scene = [val_scene]
        
        # Handle train_scene
        if train_scene is not None:
            if isinstance(train_scene, str):
                train_scene = [train_scene]
            # Remove any overlap with val_scene
            train_scene = [s for s in train_scene if s not in val_scene]
        
        self.val_scene = val_scene
        self.train_scene = train_scene
        print(f"Validation scenes: {val_scene}")
        if train_scene is not None:
            print(f"Train scenes: {train_scene}")
        else:
            print(f"Train scenes: All scenes except val_scene")
        print(f"Total scenes: {len(self.scene_data_dirs)}")
        
        # Load calibration data for each scene
        for scene, data_dir in self.scene_data_dirs.items():
            calib_file = os.path.join(data_dir, 'calibration.yaml')
            if not os.path.exists(calib_file):
                continue
                
            # Load calibration.yaml
            with open(calib_file, 'r') as f:
                calib_data = yaml.safe_load(f)
            
            # Extract camera intrinsic (K matrix)
            # Support multiple formats
            K_matrix = None
            
            # Format 1: Direct 3x3 matrix in camera.intrinsic
            if calib_data and 'camera' in calib_data and 'intrinsic' in calib_data['camera']:
                K_matrix = np.array(calib_data['camera']['intrinsic'], dtype=np.float32)
                if K_matrix.shape != (3, 3):
                    K_matrix = None
            
            # Format 2: Individual fx, fy, cx, cy fields
            if K_matrix is None and calib_data:
                if all(key in calib_data for key in ['fx', 'fy', 'cx', 'cy']):
                    fx = float(calib_data['fx'])
                    fy = float(calib_data['fy'])
                    cx = float(calib_data['cx'])
                    cy = float(calib_data['cy'])
                    K_matrix = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]], dtype=np.float32)
            
            if K_matrix is None or K_matrix.shape != (3, 3):
                print(f"Warning: No valid camera intrinsic found for scene {scene}")
                continue
            
            self.K[scene] = K_matrix
            
            # Extract camera-radar extrinsic (T_cam_radar)
            T_cam_radar = None
            
            # Format 1: Direct 4x4 matrix in extrinsic.T_cam_radar
            if calib_data and 'extrinsic' in calib_data:
                if 'T_cam_radar' in calib_data['extrinsic']:
                    T_cam_radar = np.array(calib_data['extrinsic']['T_cam_radar'], dtype=np.float32)
                    if T_cam_radar.shape != (4, 4):
                        T_cam_radar = None
                elif 'rotation' in calib_data['extrinsic'] and 'translation' in calib_data['extrinsic']:
                    # Build transformation matrix from rotation and translation
                    R = np.array(calib_data['extrinsic']['rotation'], dtype=np.float32)
                    t = np.array(calib_data['extrinsic']['translation'], dtype=np.float32)
                    T_cam_radar = np.eye(4, dtype=np.float32)
                    T_cam_radar[:3, :3] = R
                    T_cam_radar[:3, 3] = t
            
            # Format 2: initial_extrinsic as flattened list (16 elements)
            if T_cam_radar is None and calib_data:
                if 'initial_extrinsic' in calib_data:
                    ext_list = calib_data['initial_extrinsic']
                    if isinstance(ext_list, list) and len(ext_list) == 16:
                        T_cam_radar = np.array(ext_list, dtype=np.float32).reshape(4, 4)
            
            if T_cam_radar is None or T_cam_radar.shape != (4, 4):
                print(f"Warning: No valid extrinsic found for scene {scene}")
                continue
            
            self.GTs_T_cam_radar[scene] = T_cam_radar
            
            # Get image and radar file lists
            camera_dir = os.path.join(data_dir, 'camera')
            radar_dir = os.path.join(data_dir, 'radar')
            
            if not os.path.exists(camera_dir) or not os.path.exists(radar_dir):
                continue
            
            # Get image files
            image_list = [f for f in os.listdir(camera_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_list.sort()
            
            # Get radar files and create a set of base names for fast lookup (.pcd format only)
            radar_base_names = set()
            if os.path.exists(radar_dir):
                radar_files = os.listdir(radar_dir)
                for radar_file in radar_files:
                    if radar_file.endswith('.pcd'):
                        base_name = os.path.splitext(radar_file)[0]
                        radar_base_names.add(base_name)
            
            # Match image files with radar files (only keep pairs that exist)
            for image_name in image_list:
                base_name = os.path.splitext(image_name)[0]
                
                # Check if corresponding radar file exists
                if base_name not in radar_base_names:
                    continue
                
                # Add to file list based on split
                if scene in val_scene:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(scene, base_name))
                elif split == 'train':
                    # If train_scene is specified, only use those scenes
                    if self.train_scene is not None:
                        if scene in self.train_scene:
                            self.all_files.append(os.path.join(scene, base_name))
                    else:
                        # Otherwise, use all scenes except val_scene
                        if scene not in val_scene:
                            self.all_files.append(os.path.join(scene, base_name))
        
        # Test open3d with first pcd file if available
        self.pcd_reader = None
        first_scan_file = None
        for scene, data_dir in self.scene_data_dirs.items():
            radar_dir = os.path.join(data_dir, 'radar')
            if os.path.exists(radar_dir):
                radar_files = [f for f in os.listdir(radar_dir) if f.endswith('.pcd')]
                if radar_files:
                    first_scan_file = os.path.join(radar_dir, radar_files[0])
                    break
        
        if first_scan_file:
            try_pcd = o3d.io.read_point_cloud(first_scan_file)
            if try_pcd.is_empty():
                # open3d binding does not raise an exception if file is unreadable or extension is not supported
                raise Exception("Generic Dataloader| Open3d PointCloud file is empty")
            self.pcd_reader = ReadOpen3d()
        
        # Generate validation RT perturbations
        self.val_RT = []
        if split == 'val' or split == 'test':
            # Create filename from val_scene list
            if isinstance(val_scene, list):
                val_scene_name = '_'.join(val_scene)
            else:
                val_scene_name = str(val_scene)
            val_RT_file = os.path.join(dataset_dir, 
                                       f'val_RT_scene{val_scene_name}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file_handle = open(val_RT_file, 'w')
                val_RT_writer = csv.writer(val_RT_file_handle, delimiter=',')
                val_RT_writer.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    val_RT_writer.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])
                val_RT_file_handle.close()
            
            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

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

    def load_radar_data(self, radar_path):
        """Load radar point cloud data (.pcd format only)"""
        if self.pcd_reader is None:
            raise Exception("Generic Dataloader| Open3d PointCloud reader not initialized")
        
        pts = self.pcd_reader(radar_path)
        if pts is None or pts.size == 0:
            raise ValueError(f"Empty or unreadable .pcd file: {radar_path}")
        pc = pts.astype(np.float32)
        if len(pc.shape) == 1:
            pc = pc.reshape(1, -1)
        if pc.shape[1] == 3:
            pc = np.hstack([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)])
        elif pc.shape[1] > 4:
            pc = pc[:, :4]
        
        return pc

    def __getitem__(self, idx):
        item = self.all_files[idx]
        scene = str(item.split('/')[0])
        base_name = str(item.split('/')[1])
        
        # Get the actual data directory for this scene (from initialization)
        if scene in self.scene_data_dirs:
            data_dir = self.scene_data_dirs[scene]
        else:
            # Fallback: try to find it
            scene_path = os.path.join(self.root_dir, scene)
            data_dir = scene_path
            if not os.path.exists(os.path.join(scene_path, 'calibration.yaml')):
                for subdir in os.listdir(scene_path):
                    subdir_path = os.path.join(scene_path, subdir)
                    if os.path.isdir(subdir_path):
                        if os.path.exists(os.path.join(subdir_path, 'calibration.yaml')):
                            data_dir = subdir_path
                            break
        
        # Paths
        camera_dir = os.path.join(data_dir, 'camera')
        radar_dir = os.path.join(data_dir, 'radar')
        
        # Find image file
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(camera_dir, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {base_name} in scene {scene}")
        
        # Find radar file (.pcd format only)
        radar_path = os.path.join(radar_dir, base_name + '.pcd')
        if not os.path.exists(radar_path):
            raise FileNotFoundError(f"Radar data not found for {base_name} in scene {scene}")
        
        # Load radar point cloud
        pc = self.load_radar_data(radar_path)
        
        # Filter points (similar to KITTI)
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        
        # Get ground truth transformation (T_cam_radar: radar to camera)
        RT = self.GTs_T_cam_radar[scene].astype(np.float32)
        
        # Convert point cloud to homogeneous coordinates
        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        # Transform radar points to camera coordinate system
        pc_rot = np.matmul(RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)
        
        # Load and transform image
        img = Image.open(img_path)
        img_rotation = 0.
        try:
            img = self.custom_transform(img, img_rotation, False)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)
        
        # Generate perturbation for training/validation
        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]
        
        # Create rotation and translation
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))
        
        # Invert pose for training
        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)
        
        # Get camera intrinsic
        calib = self.K[scene]
        
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                  'tr_error': T, 'rot_error': R, 'scene': scene,
                  'rgb_name': base_name + self.suf, 'item': item, 'extrin': RT}
        
        return sample

