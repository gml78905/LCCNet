import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DatasetLGInnotek import ReadOpen3d, _load_point_cloud


POINTCLOUD_EXTS = {".pcd", ".bin", ".txt", ".csv"}
POINTCLOUD_DIR_HINTS = ("lidar", "radar", "Hesai", "Aeva", "Continental")


def iter_pointcloud_files(root_dir):
    for current_root, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in POINTCLOUD_EXTS:
                full_path = os.path.join(current_root, name)
                path_parts = full_path.split(os.sep)
                if ext in {".txt", ".csv"}:
                    if not any(hint in part for part in path_parts for hint in POINTCLOUD_DIR_HINTS):
                        continue
                yield full_path


def main():
    parser = argparse.ArgumentParser(description="Precompute sidecar .npy caches for point clouds.")
    parser.add_argument("--data-root", required=True, help="Dataset root to scan recursively")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing cache files")
    args = parser.parse_args()

    pcd_reader = ReadOpen3d()
    created = 0
    skipped = 0
    failed = 0

    for file_path in iter_pointcloud_files(args.data_root):
        cache_path = f"{file_path}.npy"
        if os.path.exists(cache_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            points = _load_point_cloud(file_path, pcd_reader)
            np.save(cache_path, points.astype(np.float32))
            created += 1
            if created % 100 == 0:
                print(f"[cache_pointclouds] created={created} skipped={skipped} failed={failed}")
        except Exception as exc:
            failed += 1
            print(f"[cache_pointclouds] failed: {file_path} ({exc})")

    print(
        f"[cache_pointclouds] done created={created} skipped={skipped} "
        f"failed={failed} root={args.data_root}"
    )


if __name__ == "__main__":
    main()
