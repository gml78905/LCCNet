import argparse
import os


POINTCLOUD_CACHE_SUFFIXES = (
    ".pcd.npy",
    ".bin.npy",
    ".txt.npy",
    ".csv.npy",
)


def iter_cache_files(root_dir):
    for current_root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(POINTCLOUD_CACHE_SUFFIXES):
                yield os.path.join(current_root, name)


def main():
    parser = argparse.ArgumentParser(description="Delete sidecar point cloud cache .npy files.")
    parser.add_argument("--data-root", required=True, help="Dataset root to scan recursively")
    parser.add_argument("--dry-run", action="store_true", help="Only print files that would be deleted")
    args = parser.parse_args()

    deleted = 0
    total_size = 0

    for file_path in iter_cache_files(args.data_root):
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            file_size = 0

        if args.dry_run:
            print(file_path)
        else:
            os.remove(file_path)
            if deleted % 1000 == 0 and deleted > 0:
                print(f"[delete_pointcloud_caches] deleted={deleted}")

        deleted += 1
        total_size += file_size

    size_mb = total_size / (1024 * 1024)
    mode = "would delete" if args.dry_run else "deleted"
    print(f"[delete_pointcloud_caches] {mode} {deleted} files ({size_mb:.2f} MB) under {args.data_root}")


if __name__ == "__main__":
    main()
