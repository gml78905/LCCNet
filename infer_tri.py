import argparse
import math
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from DatasetLGInnotek import DatasetTriModalLGInnotek, _load_point_cloud
from models.tri_calib.model import TriModalCalibNet
from quaternion_distances import quaternion_distance
from utils import merge_inputs, overlay_imgs, quat2mat, quaternion_from_matrix


def _build_model(checkpoint_path, device, dropout=0.0, debug_shapes=False):
    model = TriModalCalibNet(
        camera_pretrained=False,
        activation="leakyrelu",
        head_hidden_dim=256,
        dropout=dropout,
        debug_shapes=debug_shapes,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Be tolerant to either plain or DataParallel-formatted checkpoints.
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _make_dataset(args):
    input_size = None if args.keep_original_size else (288, 512)
    return DatasetTriModalLGInnotek(
        dataset_dir=args.data_root,
        max_r=args.max_r,
        max_t=args.max_t,
        split=args.split,
        use_reflectance=False,
        train_scene=args.train_scenes,
        val_scene=args.val_scenes,
        val_frame_limit=args.val_frame_limit,
        input_size=input_size,
        max_depth=args.max_depth,
    )


def _tensor_to_list(tensor):
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _matrix_batch_to_quaternion(T_batch):
    quats = [quaternion_from_matrix(T_batch[i]) for i in range(T_batch.shape[0])]
    return torch.stack(quats, dim=0)


def _pose_to_matrix(t_vec, q_vec):
    T = torch.eye(4, device=t_vec.device, dtype=t_vec.dtype)
    T[:3, :3] = quat2mat(q_vec)[:3, :3]
    T[:3, 3] = t_vec
    return T.detach().cpu().numpy().astype(np.float32)


def _overlay_to_image(rgb_tensor, proj_tensor):
    blended = overlay_imgs(rgb_tensor.detach().cpu(), proj_tensor.detach().cpu())
    return Image.fromarray((blended * 255.0).clip(0, 255).astype(np.uint8))


def _save_triptych_png(path, rgb_tensor, true_proj, input_proj, pred_proj):
    true_img = _overlay_to_image(rgb_tensor, true_proj)
    input_img = _overlay_to_image(rgb_tensor, input_proj)
    pred_img = _overlay_to_image(rgb_tensor, pred_proj)

    width, height = true_img.size
    header_h = 28
    canvas = Image.new("RGB", (width * 3, height + header_h), color=(255, 255, 255))
    canvas.paste(true_img, (0, header_h))
    canvas.paste(input_img, (width, header_h))
    canvas.paste(pred_img, (width * 2, header_h))

    draw = ImageDraw.Draw(canvas)
    labels = ["true", "init", "pred"]
    for i, label in enumerate(labels):
        x0 = i * width
        draw.rectangle([x0, 0, x0 + width - 1, header_h - 1], fill=(245, 245, 245))
        draw.text((x0 + 10, 6), label, fill=(0, 0, 0))

    canvas.save(path)


def _resolve_save_dir(args):
    if not args.save:
        return None
    if args.save_dir is not None:
        return args.save_dir
    if args.output is not None:
        base = os.path.splitext(args.output)[0]
        return f"{base}_images"
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    return os.path.join(os.getcwd(), "infer_tri_images", ckpt_name)


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.keep_original_size and args.batch_size != 1:
        raise ValueError("--keep-original-size currently requires --batch-size 1.")
    dataset = _make_dataset(args)
    save_dir = _resolve_save_dir(args)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    print(f"[infer_tri] split={args.split} samples={len(dataset)} batch_size={args.batch_size}")
    print(f"[infer_tri] loading checkpoint: {args.checkpoint}")
    if save_dir is not None:
        print(f"[infer_tri] saving projection images to: {save_dir}")
    loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=merge_inputs,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    model = _build_model(args.checkpoint, device, dropout=args.dropout, debug_shapes=args.debug_shapes)

    results = []
    total_input_t = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    total_input_r = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    total_pred_t = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    total_pred_r = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            print(f"[infer_tri] batch {batch_idx + 1}/{len(loader)}")
            rgb = sample["rgb"].to(device, non_blocking=True)
            lidar_proj = sample["lidar_proj"].to(device, non_blocking=True)
            radar_proj = sample["radar_proj"].to(device, non_blocking=True)
            T_cl_input = sample["T_CL_input"].to(device, non_blocking=True)
            T_cr_input = sample["T_CR_input"].to(device, non_blocking=True)
            T_lr_input = torch.linalg.inv(T_cl_input) @ T_cr_input
            gt_batch = {
                "T_CL_t_gt": sample["T_CL_t_gt"].to(device, non_blocking=True),
                "T_CL_q_gt": sample["T_CL_q_gt"].to(device, non_blocking=True),
                "T_CR_t_gt": sample["T_CR_t_gt"].to(device, non_blocking=True),
                "T_CR_q_gt": sample["T_CR_q_gt"].to(device, non_blocking=True),
                "T_LR_t_gt": sample["T_LR_t_gt"].to(device, non_blocking=True),
                "T_LR_q_gt": sample["T_LR_q_gt"].to(device, non_blocking=True),
            }
            input_q = {
                "CL": _matrix_batch_to_quaternion(T_cl_input),
                "CR": _matrix_batch_to_quaternion(T_cr_input),
                "LR": _matrix_batch_to_quaternion(T_lr_input),
            }

            pred = model(rgb, lidar_proj, radar_proj)
            batch_size = rgb.shape[0]

            for item_idx in range(batch_size):
                dataset_idx = batch_idx * args.batch_size + item_idx
                meta = dataset.all_files[dataset_idx]
                input_err_t = {
                    "CL_cm": float(torch.norm(T_cl_input[item_idx, :3, 3] - gt_batch["T_CL_t_gt"][item_idx]).item() * 100.0),
                    "CR_cm": float(torch.norm(T_cr_input[item_idx, :3, 3] - gt_batch["T_CR_t_gt"][item_idx]).item() * 100.0),
                    "LR_cm": float(torch.norm(T_lr_input[item_idx, :3, 3] - gt_batch["T_LR_t_gt"][item_idx]).item() * 100.0),
                }
                input_err_r = {
                    "CL_deg": float(quaternion_distance(input_q["CL"][item_idx:item_idx + 1], gt_batch["T_CL_q_gt"][item_idx:item_idx + 1], device).item() * 180.0 / math.pi),
                    "CR_deg": float(quaternion_distance(input_q["CR"][item_idx:item_idx + 1], gt_batch["T_CR_q_gt"][item_idx:item_idx + 1], device).item() * 180.0 / math.pi),
                    "LR_deg": float(quaternion_distance(input_q["LR"][item_idx:item_idx + 1], gt_batch["T_LR_q_gt"][item_idx:item_idx + 1], device).item() * 180.0 / math.pi),
                }
                pred_err_t = {
                    "CL_cm": float(torch.norm(pred["T_CL_t"][item_idx] - gt_batch["T_CL_t_gt"][item_idx]).item() * 100.0),
                    "CR_cm": float(torch.norm(pred["T_CR_t"][item_idx] - gt_batch["T_CR_t_gt"][item_idx]).item() * 100.0),
                    "LR_cm": float(torch.norm(pred["T_LR_t"][item_idx] - gt_batch["T_LR_t_gt"][item_idx]).item() * 100.0),
                }
                pred_err_r = {
                    "CL_deg": float(quaternion_distance(pred["T_CL_q"][item_idx:item_idx + 1], gt_batch["T_CL_q_gt"][item_idx:item_idx + 1], device).item() * 180.0 / math.pi),
                    "CR_deg": float(quaternion_distance(pred["T_CR_q"][item_idx:item_idx + 1], gt_batch["T_CR_q_gt"][item_idx:item_idx + 1], device).item() * 180.0 / math.pi),
                    "LR_deg": float(quaternion_distance(pred["T_LR_q"][item_idx:item_idx + 1], gt_batch["T_LR_q_gt"][item_idx:item_idx + 1], device).item() * 180.0 / math.pi),
                }
                result = {
                    "index": dataset_idx,
                    "scene": meta.get("scene"),
                    "image_path": meta.get("image_path"),
                    "lidar_path": meta.get("lidar_path"),
                    "radar_path": meta.get("radar_path"),
                    "T_CL_t": _tensor_to_list(pred["T_CL_t"][item_idx]),
                    "T_CL_q": _tensor_to_list(pred["T_CL_q"][item_idx]),
                    "T_CR_t": _tensor_to_list(pred["T_CR_t"][item_idx]),
                    "T_CR_q": _tensor_to_list(pred["T_CR_q"][item_idx]),
                    "T_LR_t": _tensor_to_list(pred["T_LR_t"][item_idx]),
                    "T_LR_q": _tensor_to_list(pred["T_LR_q"][item_idx]),
                    "input_error_translation_cm": input_err_t,
                    "input_error_rotation_deg": input_err_r,
                    "pred_error_translation_cm": pred_err_t,
                    "pred_error_rotation_deg": pred_err_r,
                }

                if save_dir is not None:
                    rgb_img, calib = dataset._load_image(meta["image_path"])
                    rgb_save_tensor = dataset.custom_transform(rgb_img, img_rotation=0.0, flip=False)
                    orig_hw = (rgb_img.height, rgb_img.width)
                    lidar_pc = _load_point_cloud(meta["lidar_path"], dataset.pcd_reader)
                    radar_pc = _load_point_cloud(meta["radar_path"], dataset.pcd_reader)

                    T_cl_gt = _pose_to_matrix(gt_batch["T_CL_t_gt"][item_idx], gt_batch["T_CL_q_gt"][item_idx])
                    T_cr_gt = _pose_to_matrix(gt_batch["T_CR_t_gt"][item_idx], gt_batch["T_CR_q_gt"][item_idx])
                    T_cl_input_np = T_cl_input[item_idx].detach().cpu().numpy().astype(np.float32)
                    T_cr_input_np = T_cr_input[item_idx].detach().cpu().numpy().astype(np.float32)
                    T_cl_pred = _pose_to_matrix(pred["T_CL_t"][item_idx], pred["T_CL_q"][item_idx])
                    T_cr_pred = _pose_to_matrix(pred["T_CR_t"][item_idx], pred["T_CR_q"][item_idx])

                    lidar_gt, _ = dataset._project_to_image(lidar_pc, T_cl_gt, calib, orig_hw)
                    lidar_input, _ = dataset._project_to_image(lidar_pc, T_cl_input_np, calib, orig_hw)
                    lidar_pred, _ = dataset._project_to_image(lidar_pc, T_cl_pred, calib, orig_hw)
                    radar_gt, _ = dataset._project_to_image(radar_pc, T_cr_gt, calib, orig_hw)
                    radar_input, _ = dataset._project_to_image(radar_pc, T_cr_input_np, calib, orig_hw)
                    radar_pred, _ = dataset._project_to_image(radar_pc, T_cr_pred, calib, orig_hw)

                    def _prep_proj(depth_map):
                        return torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0)

                    lidar_true_vis = _prep_proj(lidar_gt)
                    lidar_input_vis = _prep_proj(lidar_input)
                    lidar_pred_vis = _prep_proj(lidar_pred)
                    radar_true_vis = _prep_proj(radar_gt)
                    radar_input_vis = _prep_proj(radar_input)
                    radar_pred_vis = _prep_proj(radar_pred)

                    lidar_save_path = os.path.join(save_dir, f"frame_{dataset_idx:06d}_lidar.png")
                    radar_save_path = os.path.join(save_dir, f"frame_{dataset_idx:06d}_radar.png")
                    _save_triptych_png(
                        lidar_save_path,
                        rgb_save_tensor,
                        lidar_true_vis,
                        lidar_input_vis,
                        lidar_pred_vis,
                    )
                    _save_triptych_png(
                        radar_save_path,
                        rgb_save_tensor,
                        radar_true_vis,
                        radar_input_vis,
                        radar_pred_vis,
                    )
                    result["saved_lidar_image"] = lidar_save_path
                    result["saved_radar_image"] = radar_save_path

                results.append(result)
                total_input_t["CL"] += input_err_t["CL_cm"]
                total_input_t["CR"] += input_err_t["CR_cm"]
                total_input_t["LR"] += input_err_t["LR_cm"]
                total_input_r["CL"] += input_err_r["CL_deg"]
                total_input_r["CR"] += input_err_r["CR_deg"]
                total_input_r["LR"] += input_err_r["LR_deg"]
                total_pred_t["CL"] += pred_err_t["CL_cm"]
                total_pred_t["CR"] += pred_err_t["CR_cm"]
                total_pred_t["LR"] += pred_err_t["LR_cm"]
                total_pred_r["CL"] += pred_err_r["CL_deg"]
                total_pred_r["CR"] += pred_err_r["CR_deg"]
                total_pred_r["LR"] += pred_err_r["LR_deg"]
                print(
                    f"[infer_tri] saved prediction {len(results)} "
                    f"(scene={result['scene']}, index={result['index']})"
                )
                print(
                    f"[infer_tri]   T_CL_t={result['T_CL_t']} T_CL_q={result['T_CL_q']}\n"
                    f"[infer_tri]   T_CR_t={result['T_CR_t']} T_CR_q={result['T_CR_q']}\n"
                    f"[infer_tri]   T_LR_t={result['T_LR_t']} T_LR_q={result['T_LR_q']}"
                )
                print(
                    f"[infer_tri]   input error  : "
                    f"CL={input_err_t['CL_cm']:.3f}cm/{input_err_r['CL_deg']:.3f}deg "
                    f"CR={input_err_t['CR_cm']:.3f}cm/{input_err_r['CR_deg']:.3f}deg "
                    f"LR={input_err_t['LR_cm']:.3f}cm/{input_err_r['LR_deg']:.3f}deg"
                )
                print(
                    f"[infer_tri]   corrected err: "
                    f"CL={pred_err_t['CL_cm']:.3f}cm/{pred_err_r['CL_deg']:.3f}deg "
                    f"CR={pred_err_t['CR_cm']:.3f}cm/{pred_err_r['CR_deg']:.3f}deg "
                    f"LR={pred_err_t['LR_cm']:.3f}cm/{pred_err_r['LR_deg']:.3f}deg"
                )
                if args.max_samples is not None and len(results) >= args.max_samples:
                    break
            if args.max_samples is not None and len(results) >= args.max_samples:
                break

    if len(results) > 0:
        denom = float(len(results))
        print("[infer_tri] summary")
        print(
            f"[infer_tri]   mean input error  : "
            f"CL={total_input_t['CL']/denom:.3f}cm/{total_input_r['CL']/denom:.3f}deg "
            f"CR={total_input_t['CR']/denom:.3f}cm/{total_input_r['CR']/denom:.3f}deg "
            f"LR={total_input_t['LR']/denom:.3f}cm/{total_input_r['LR']/denom:.3f}deg"
        )
        print(
            f"[infer_tri]   mean corrected err: "
            f"CL={total_pred_t['CL']/denom:.3f}cm/{total_pred_r['CL']/denom:.3f}deg "
            f"CR={total_pred_t['CR']/denom:.3f}cm/{total_pred_r['CR']/denom:.3f}deg "
            f"LR={total_pred_t['LR']/denom:.3f}cm/{total_pred_r['LR']/denom:.3f}deg"
        )

    if args.output is not None:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions to {args.output}")
    else:
        print(json.dumps(results, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Tri-modal checkpoint inference for T_CL/T_CR/T_LR.")
    parser.add_argument("--checkpoint", required=True, help="Path to a tri-modal checkpoint .tar file")
    parser.add_argument("--data-root", required=True, help="LG Innotek dataset root")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--train-scenes", nargs="*", default=["afternoon_parking_lot_1", "afternoon_campus_1"])
    parser.add_argument("--val-scenes", nargs="*", default=["afternoon_campus_2"])
    parser.add_argument("--val-frame-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-r", type=float, default=5.0)
    parser.add_argument("--max-t", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--save", action="store_true", help="Save true/input/pred projection overlay images")
    parser.add_argument("--save-dir", default=None, help="Optional directory for saved projection images")
    parser.add_argument("--keep-original-size", action="store_true", help="Run inference with original image/projection size instead of resizing to 288x512")
    parser.add_argument("--debug-shapes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
