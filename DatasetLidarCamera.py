# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/DatasetVisibilityKitti.py

import csv
import os
from math import radians
import cv2

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, rotate_forward, quaternion_from_matrix
from pykitti import odometry
import pykitti
import yaml
import open3d as o3d

class ReadOpen3d:
    def __call__(self, file):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        return points


class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_sequence='00', suf='.png'):
        super(DatasetLidarCameraKittiOdometry, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf

        self.all_files = []
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.model = CameraModel()
        # self.model.focal_length = [7.18856e+02, 7.18856e+02]
        # self.model.principal_point = [6.071928e+02, 1.852157e+02]
        # for seq in ['00', '03', '05', '06', '07', '08', '09']:
        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.K[seq] = calib.K_cam2 # 3x3
            # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
            # GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
            # GT_T = T_cam02_velo[3:, :3]
            # self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            # self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GTs_T_cam02_velo[seq] = T_cam02_velo_np #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

            image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0])+suf)):
                    continue
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train':
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            # val_RT_file = os.path.join(dataset_dir, 'sequences',
            #                            f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_file = os.path.join(dataset_dir, 'sequences',
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2', rgb_name+self.suf)
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        RT = self.GTs_T_cam02_velo[seq].astype(np.float32)

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
        
        pc_rot = np.matmul(RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        # pc_rot = np.matmul(RT, pc.T)
        # pc_rot = pc_rot.astype(np.float).T.copy()
        # pc_in = torch.from_numpy(pc_rot.astype(np.float32))#.float()

        # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        #     pc_in = pc_in.t()
        # if pc_in.shape[0] == 3:
        #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        #     pc_in = torch.cat((pc_in, homogeneous), 0)
        # elif pc_in.shape[0] == 4:
        #      if not torch.all(pc_in[3,:] == 1.):
        #         pc_in[3,:] = 1.
        # else:
        #     raise TypeError("Wrong PointCloud shape")

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[1, :] *= -1

        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq), 'img_path': img_path,
                      'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT,
                      'initial_RT': initial_RT}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq),
                      'rgb_name': rgb_name, 'item': item, 'extrin': RT}

        return sample


class DatasetLidarCameraKittiRaw(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='2011_09_26_drive_0117_sync'):
        super(DatasetLidarCameraKittiRaw, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.max_depth = 80
        self.K_list = {}

        self.all_files = []
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        data_drive_list = ['0001', '0002', '0004', '0016', '0027']
        self.calib_date = {}

        for i in range(len(date_list)):
            date = date_list[i]
            data_drive = data_drive_list[i]
            data = pykitti.raw(self.root_dir, date, data_drive)
            calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                     'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
            self.calib_date[date] = calib

        # date = val_sequence[:10]
        # seq = val_sequence
        # image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
        # image_list.sort()
        #
        # for image_name in image_list:
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
        #                                        str(image_name.split('.')[0]) + '.bin')):
        #         continue
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
        #                                        str(image_name.split('.')[0]) + '.jpg')):  # png
        #         continue
        #     self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        date = val_sequence[:10]
        test_list = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync', '2011_10_03_drive_0027_sync']
        seq_list = os.listdir(os.path.join(self.root_dir, date))

        for seq in seq_list:
            if not os.path.isdir(os.path.join(dataset_dir, date, seq)):
                continue
            image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
                                                   str(image_name.split('.')[0])+'.jpg')): # png
                    continue
                if seq == val_sequence and (not split == 'train'):
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq not in test_list:
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            # color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    # self.all_files.append(os.path.join(date, seq, 'image_2/data', image_name.split('.')[0]))
    def __getitem__(self, idx):
        item = self.all_files[idx]
        date = str(item.split('/')[0])
        seq = str(item.split('/')[1])
        rgb_name = str(item.split('/')[4])
        img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name+'.jpg') # png
        lidar_path = os.path.join(self.root_dir, date, seq, 'velodyne_points/data', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_lidar = pc.copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        calib = self.calib_date[date]
        RT_cam02 = calib['RT2'].astype(np.float32)
        # camera intrinsic parameter
        calib_cam02 = calib['K2']  # 3x3

        E_RT = RT_cam02

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

        pc_rot = np.matmul(E_RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        img = Image.open(img_path)
        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = calib_cam02
        # calib = get_calib_kitti_odom(int(seq))
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        # sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
        #           'tr_error': T, 'rot_error': R, 'seq': int(seq), 'rgb_name': rgb_name, 'item': item,
        #           'extrin': E_RT, 'initial_RT': initial_RT}
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': rgb_name + '.png', 'item': item,
                  'extrin': E_RT, 'initial_RT': initial_RT, 'pc_lidar': pc_lidar}

        return sample


class DatasetLidarCameraHercules(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_scene=None, train_scene=None, suf='.png'):
        super(DatasetLidarCameraHercules, self).__init__()
        self.use_reflectance = use_reflectance
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.suf = suf
        
        self.GTs_T_cam_lidar = {}  # Ground truth transformation from camera to lidar
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
            
            # Extract camera-lidar extrinsic (T_cam_lidar)
            T_cam_lidar = None
            
            # Format 1: Direct 4x4 matrix in extrinsic.T_cam_lidar or T_cam2_velo
            if calib_data and 'extrinsic' in calib_data:
                if 'T_cam_lidar' in calib_data['extrinsic']:
                    T_cam_lidar = np.array(calib_data['extrinsic']['T_cam_lidar'], dtype=np.float32)
                    if T_cam_lidar.shape != (4, 4):
                        T_cam_lidar = None
                elif 'T_cam2_velo' in calib_data['extrinsic']:
                    T_cam_lidar = np.array(calib_data['extrinsic']['T_cam2_velo'], dtype=np.float32)
                    if T_cam_lidar.shape != (4, 4):
                        T_cam_lidar = None
                elif 'rotation' in calib_data['extrinsic'] and 'translation' in calib_data['extrinsic']:
                    # Build transformation matrix from rotation and translation
                    R = np.array(calib_data['extrinsic']['rotation'], dtype=np.float32)
                    t = np.array(calib_data['extrinsic']['translation'], dtype=np.float32)
                    T_cam_lidar = np.eye(4, dtype=np.float32)
                    T_cam_lidar[:3, :3] = R
                    T_cam_lidar[:3, 3] = t
            
            # Format 2: initial_extrinsic as flattened list (16 elements)
            if T_cam_lidar is None and calib_data:
                if 'initial_extrinsic' in calib_data:
                    ext_list = calib_data['initial_extrinsic']
                    if isinstance(ext_list, list) and len(ext_list) == 16:
                        T_cam_lidar = np.array(ext_list, dtype=np.float32).reshape(4, 4)
            
            if T_cam_lidar is None or T_cam_lidar.shape != (4, 4):
                print(f"Warning: No valid extrinsic found for scene {scene}")
                continue
            
            self.GTs_T_cam_lidar[scene] = T_cam_lidar
            
            # Get image and lidar file lists
            camera_dir = os.path.join(data_dir, 'camera')
            lidar_dir = os.path.join(data_dir, 'lidar')
            
            if not os.path.exists(camera_dir) or not os.path.exists(lidar_dir):
                continue
            
            # Get image files
            image_list = [f for f in os.listdir(camera_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_list.sort()
            
            # Get lidar files and create a set of base names for fast lookup (.pcd format only)
            lidar_base_names = set()
            if os.path.exists(lidar_dir):
                lidar_files = os.listdir(lidar_dir)
                for lidar_file in lidar_files:
                    if lidar_file.endswith('.pcd'):
                        base_name = os.path.splitext(lidar_file)[0]
                        lidar_base_names.add(base_name)
            
            # Match image files with lidar files (only keep pairs that exist)
            for image_name in image_list:
                base_name = os.path.splitext(image_name)[0]
                
                # Check if corresponding lidar file exists
                if base_name not in lidar_base_names:
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
            lidar_dir = os.path.join(data_dir, 'lidar')
            if os.path.exists(lidar_dir):
                lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.pcd')]
                if lidar_files:
                    first_scan_file = os.path.join(lidar_dir, lidar_files[0])
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
                                       f'val_RT_left_scene{val_scene_name}_{max_r:.2f}_{max_t:.2f}.csv')
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

    def load_lidar_data(self, lidar_path):
        """Load lidar point cloud data (.pcd format only)"""
        if self.pcd_reader is None:
            raise Exception("Generic Dataloader| Open3d PointCloud reader not initialized")
        
        pts = self.pcd_reader(lidar_path)
        if pts is None or pts.size == 0:
            raise ValueError(f"Empty or unreadable .pcd file: {lidar_path}")
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
        lidar_dir = os.path.join(data_dir, 'lidar')
        
        # Find image file
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(camera_dir, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {base_name} in scene {scene}")
        
        # Find lidar file (.pcd format only)
        lidar_path = os.path.join(lidar_dir, base_name + '.pcd')
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"Lidar data not found for {base_name} in scene {scene}")
        
        # Load lidar point cloud
        pc = self.load_lidar_data(lidar_path)
        
        # Filter points (similar to KITTI)
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        
        if self.use_reflectance:
            if pc.shape[1] >= 4:
                reflectance = pc[:, 3].copy()
                reflectance = torch.from_numpy(reflectance).float()
        
        # Get ground truth transformation (T_cam_lidar: lidar to camera)
        RT = self.GTs_T_cam_lidar[scene].astype(np.float32)
        
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
        
        # Transform lidar points to camera coordinate system
        pc_rot = np.matmul(RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)
        
        h_mirror = False
        
        # Load and transform image
        img = Image.open(img_path)
        img_rotation = 0.
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)
        
        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)
        
        # Generate perturbation for training/validation
        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            initial_RT = None
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
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]
        
        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'scene': scene, 'img_path': img_path,
                      'rgb_name': base_name + self.suf, 'item': item, 'extrin': RT,
                      'initial_RT': initial_RT}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'scene': scene,
                      'rgb_name': base_name + self.suf, 'item': item, 'extrin': RT}
        
        return sample

