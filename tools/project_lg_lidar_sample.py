import os

import cv2
import numpy as np

from DatasetLGInnotek import (
    _compute_rectified_intrinsic,
    _load_extrinsic_matrix,
    _load_intrinsic_file,
)


DEFAULT_ROOTS = [
    "/workspace/data/LG_Innotek/CustomDataset/LG_Innotek",
    "/media/TrainDataset/LG_Innotek/CustomDataset/LG_Innotek",
]
SCENE = "afternoon_campus_2"
PAIR_INDEX = 0
USE_UNDISTORT = os.environ.get("USE_UNDISTORT", "1") != "0"
OUTPUT_PATH = f"/workspace/LCCNet/tmp/lg_innotek_lidar_projection_pair0_{'undistort' if USE_UNDISTORT else 'raw'}.png"


def main():
    dataset_root = None
    for candidate in DEFAULT_ROOTS:
        if os.path.exists(candidate):
            dataset_root = candidate
            break
    if dataset_root is None:
        raise FileNotFoundError(f"Could not find dataset root in: {DEFAULT_ROOTS}")

    intrinsic_path = os.path.join(dataset_root, "intrinsic.txt")
    extrinsic_path = os.path.join(dataset_root, "lg_init_extrinsics.yaml")
    pair_file = os.path.join(dataset_root, SCENE, "offline", "synced_stamps", "image_Cam0_lidar_Hesai.txt")
    image_dir = os.path.join(dataset_root, SCENE, "offline", "sensor_data", "image_Cam0")
    lidar_dir = os.path.join(dataset_root, SCENE, "offline", "sensor_data", "lidar_Hesai")

    with open(pair_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    pairs = [line.split() for line in lines[1:]]
    image_stamp, lidar_stamp = pairs[PAIR_INDEX][0], pairs[PAIR_INDEX][1]

    image_path = os.path.join(image_dir, f"{image_stamp}.png")
    lidar_path = os.path.join(lidar_dir, f"{lidar_stamp}.bin")

    K_full, dist = _load_intrinsic_file(intrinsic_path)
    K_half = K_full.copy()
    K_half[0, 0] *= 0.5
    K_half[1, 1] *= 0.5
    K_half[0, 2] *= 0.5
    K_half[1, 2] *= 0.5

    T_lidar_cam = _load_extrinsic_matrix(extrinsic_path, "lidar").astype(np.float32)

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    h, w = image_bgr.shape[:2]
    if USE_UNDISTORT:
        K_proj = _compute_rectified_intrinsic(K_half, dist, (w, h))
        image_bgr = cv2.undistort(image_bgr, K_half, dist, None, K_proj)
    else:
        K_proj = K_half

    raw = np.fromfile(lidar_path, dtype=np.float32)
    if raw.size % 4 == 0:
        pc = raw.reshape(-1, 4)
    elif raw.size % 5 == 0:
        pc = raw.reshape(-1, 5)[:, :4]
    else:
        raise ValueError(f"Unexpected lidar layout: {lidar_path}")

    xyz = pc[:, :3]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz1 = np.concatenate([xyz, ones], axis=1)
    cam_xyz1 = (T_lidar_cam @ xyz1.T).T
    cam_xyz = cam_xyz1[:, :3]

    z = cam_xyz[:, 2]
    valid = z > 0.1
    cam_xyz = cam_xyz[valid]
    z = z[valid]

    uvw = (K_proj @ cam_xyz.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]

    valid = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    uv = uv[valid]
    z = z[valid]

    z_min = float(np.min(z)) if len(z) > 0 else 0.0
    z_max = float(np.max(z)) if len(z) > 0 else 1.0
    z_range = max(z_max - z_min, 1e-6)

    overlay = image_bgr.copy()
    for point, depth in zip(uv.astype(np.int32), z):
        normalized = (depth - z_min) / z_range
        color = cv2.applyColorMap(np.array([[int((1.0 - normalized) * 255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        cv2.circle(overlay, tuple(point), 2, color.tolist(), -1, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, overlay)

    print(f"image_path={image_path}")
    print(f"lidar_path={lidar_path}")
    print(f"dataset_root={dataset_root}")
    print(f"output_path={OUTPUT_PATH}")
    print(f"use_undistort={USE_UNDISTORT}")
    print(f"K_proj={K_proj.tolist()}")
    print(f"projected_points={len(uv)}")
    print(f"image_size={w}x{h}")
    print(f"depth_range=({z_min:.3f}, {z_max:.3f})")


if __name__ == "__main__":
    main()
