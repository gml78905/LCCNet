import argparse
import math
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from DatasetLGInnotek import DatasetTriModalLGInnotek, DatasetTriModalHercules, TriSequenceDataset, _load_point_cloud
from models.tri_calib.model import TriModalCalibNet
from quaternion_distances import quaternion_distance
from utils import merge_inputs, overlay_imgs, project_pointcloud_to_image_torch, quat2mat, quaternion_from_matrix


def _build_model(checkpoint_path, device, dropout=0.0, use_recurrence=False, recurrent_hidden_dim=256):
    model = TriModalCalibNet(
        camera_pretrained=False,
        activation="leakyrelu",
        head_hidden_dim=256,
        dropout=dropout,
        use_recurrence=use_recurrence,
        recurrent_hidden_dim=recurrent_hidden_dim,
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
    dataset_cls = DatasetTriModalHercules if args.dataset == "hercules" else DatasetTriModalLGInnotek
    dataset = dataset_cls(
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
        project_on_gpu=args.project_on_gpu,
    )
    if args.use_sequence:
        dataset = TriSequenceDataset(dataset, seq_len=args.seq_len, stride=args.seq_stride, project_on_gpu=args.project_on_gpu)
    return dataset


def _tensor_to_list(tensor):
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _matrix_batch_to_quaternion(T_batch):
    flat = T_batch.reshape(-1, T_batch.shape[-2], T_batch.shape[-1])
    return quaternion_from_matrix(flat)


def _matrix_to_pose(T_batch):
    flat = T_batch.reshape(-1, T_batch.shape[-2], T_batch.shape[-1])
    t_batch = flat[:, :3, 3]
    q_batch = quaternion_from_matrix(flat)
    return t_batch, q_batch


def _pose_batch_to_matrix(t_batch, q_batch):
    flat_t = t_batch.reshape(-1, t_batch.shape[-1])
    flat_q = q_batch.reshape(-1, q_batch.shape[-1])
    T = quat2mat(flat_q)
    T[:, :3, 3] = flat_t
    return T


def _apply_delta_to_input(pred_t, pred_q, input_T):
    delta_T = _pose_batch_to_matrix(pred_t, pred_q)
    flat_input = input_T.reshape(-1, input_T.shape[-2], input_T.shape[-1])
    corrected_T = torch.bmm(delta_T, flat_input)
    corrected_t, corrected_q = _matrix_to_pose(corrected_T)
    return corrected_t, corrected_q, corrected_T


def _build_tri_projection_batch(sample_batch, device, input_size, max_depth):
    if 'lidar_proj' in sample_batch and 'radar_proj' in sample_batch:
        return (
            sample_batch['lidar_proj'].to(device, non_blocking=True),
            sample_batch['radar_proj'].to(device, non_blocking=True),
        )

    if 'lidar_pc_seq' in sample_batch:
        batch_size_local = len(sample_batch['lidar_pc_seq'])
        seq_len_local = len(sample_batch['lidar_pc_seq'][0])
        calib_seq = sample_batch['calib_seq'].to(device, non_blocking=True)
        image_hw_seq = sample_batch['image_hw_seq']
        T_cl_input_seq = sample_batch['T_CL_input'].to(device, non_blocking=True)
        T_cr_input_seq = sample_batch['T_CR_input'].to(device, non_blocking=True)
        lidar_batch = []
        radar_batch = []
        for b in range(batch_size_local):
            lidar_steps = []
            radar_steps = []
            for t in range(seq_len_local):
                lidar_pc = sample_batch['lidar_pc_seq'][b][t].to(device, non_blocking=True)
                radar_pc = sample_batch['radar_pc_seq'][b][t].to(device, non_blocking=True)
                image_hw = image_hw_seq[b, t].tolist()
                lidar_depth, _ = project_pointcloud_to_image_torch(lidar_pc, T_cl_input_seq[b, t], calib_seq[b, t], image_hw, max_depth)
                radar_depth, radar_aux = project_pointcloud_to_image_torch(radar_pc, T_cr_input_seq[b, t], calib_seq[b, t], image_hw, max_depth)
                lidar_steps.append(lidar_depth.unsqueeze(0))
                radar_steps.append(torch.stack([radar_depth, radar_aux], dim=0))
            lidar_tensor = torch.stack(lidar_steps, dim=0)
            radar_tensor = torch.stack(radar_steps, dim=0)
            if input_size is not None:
                lidar_tensor = F.interpolate(lidar_tensor, size=input_size, mode='bilinear', align_corners=False)
                radar_tensor = F.interpolate(radar_tensor, size=input_size, mode='bilinear', align_corners=False)
            lidar_batch.append(lidar_tensor)
            radar_batch.append(radar_tensor)
        return torch.stack(lidar_batch, dim=0), torch.stack(radar_batch, dim=0)

    batch_size_local = len(sample_batch['lidar_pc'])
    calib_batch = sample_batch['calib'].to(device, non_blocking=True)
    image_hw_batch = sample_batch['image_hw']
    T_cl_input_batch = sample_batch['T_CL_input'].to(device, non_blocking=True)
    T_cr_input_batch = sample_batch['T_CR_input'].to(device, non_blocking=True)
    lidar_batch = []
    radar_batch = []
    for b in range(batch_size_local):
        lidar_pc = sample_batch['lidar_pc'][b].to(device, non_blocking=True)
        radar_pc = sample_batch['radar_pc'][b].to(device, non_blocking=True)
        image_hw = image_hw_batch[b].tolist()
        lidar_depth, _ = project_pointcloud_to_image_torch(lidar_pc, T_cl_input_batch[b], calib_batch[b], image_hw, max_depth)
        radar_depth, radar_aux = project_pointcloud_to_image_torch(radar_pc, T_cr_input_batch[b], calib_batch[b], image_hw, max_depth)
        lidar_batch.append(lidar_depth.unsqueeze(0))
        radar_batch.append(torch.stack([radar_depth, radar_aux], dim=0))
    lidar_tensor = torch.stack(lidar_batch, dim=0)
    radar_tensor = torch.stack(radar_batch, dim=0)
    if input_size is not None:
        lidar_tensor = F.interpolate(lidar_tensor, size=input_size, mode='bilinear', align_corners=False)
        radar_tensor = F.interpolate(radar_tensor, size=input_size, mode='bilinear', align_corners=False)
    return lidar_tensor, radar_tensor


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
    amp_dtype = torch.bfloat16 if args.amp_dtype in ("bf16", "bfloat16") else torch.float16
    amp_enabled = bool(args.use_amp) and device.type == "cuda"
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
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=merge_inputs,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
    model = _build_model(
        args.checkpoint,
        device,
        dropout=args.dropout,
        use_recurrence=args.use_recurrence,
        recurrent_hidden_dim=args.recurrent_hidden_dim,
    )

    results = []
    total_input_t = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    total_input_r = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    total_pred_t = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    total_pred_r = {"CL": 0.0, "CR": 0.0, "LR": 0.0}
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            print(f"[infer_tri] batch {batch_idx + 1}/{len(loader)}")
            rgb = sample["rgb"].to(device, non_blocking=True)
            lidar_proj, radar_proj = _build_tri_projection_batch(
                sample,
                device,
                dataset.input_size if getattr(dataset, "input_size", None) is not None else None,
                args.max_depth,
            )
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

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                pred = model(rgb, lidar_proj, radar_proj)
            corr_t_cl, corr_q_cl, corr_T_cl = _apply_delta_to_input(pred["T_CL_t"], pred["T_CL_q"], T_cl_input)
            corr_t_cr, corr_q_cr, corr_T_cr = _apply_delta_to_input(pred["T_CR_t"], pred["T_CR_q"], T_cr_input)
            corr_t_lr, corr_q_lr, corr_T_lr = _apply_delta_to_input(pred["T_LR_t"], pred["T_LR_q"], T_lr_input)
            batch_size = rgb.shape[0]
            seq_len = rgb.shape[1] if rgb.ndim == 5 else 1
            base_dataset = dataset.base_dataset if hasattr(dataset, "base_dataset") else dataset

            for item_idx in range(batch_size):
                for time_idx in range(seq_len):
                    if seq_len > 1:
                        frame_index = int(sample["frame_index"][item_idx, time_idx].item())
                        meta = base_dataset.all_files[frame_index]
                        delta_t_cl = pred["T_CL_t"][item_idx, time_idx]
                        delta_q_cl = pred["T_CL_q"][item_idx, time_idx]
                        delta_t_cr = pred["T_CR_t"][item_idx, time_idx]
                        delta_q_cr = pred["T_CR_q"][item_idx, time_idx]
                        delta_t_lr = pred["T_LR_t"][item_idx, time_idx]
                        delta_q_lr = pred["T_LR_q"][item_idx, time_idx]
                        gt_t_cl = gt_batch["T_CL_t_gt"][item_idx, time_idx]
                        gt_q_cl = gt_batch["T_CL_q_gt"][item_idx, time_idx]
                        gt_t_cr = gt_batch["T_CR_t_gt"][item_idx, time_idx]
                        gt_q_cr = gt_batch["T_CR_q_gt"][item_idx, time_idx]
                        gt_t_lr = gt_batch["T_LR_t_gt"][item_idx, time_idx]
                        gt_q_lr = gt_batch["T_LR_q_gt"][item_idx, time_idx]
                        T_cl_input_item = T_cl_input[item_idx, time_idx]
                        T_cr_input_item = T_cr_input[item_idx, time_idx]
                        T_lr_input_item = T_lr_input[item_idx, time_idx]
                        flat_index = item_idx * seq_len + time_idx
                    else:
                        frame_index = batch_idx * args.batch_size + item_idx
                        meta = base_dataset.all_files[frame_index]
                        delta_t_cl = pred["T_CL_t"][item_idx]
                        delta_q_cl = pred["T_CL_q"][item_idx]
                        delta_t_cr = pred["T_CR_t"][item_idx]
                        delta_q_cr = pred["T_CR_q"][item_idx]
                        delta_t_lr = pred["T_LR_t"][item_idx]
                        delta_q_lr = pred["T_LR_q"][item_idx]
                        gt_t_cl = gt_batch["T_CL_t_gt"][item_idx]
                        gt_q_cl = gt_batch["T_CL_q_gt"][item_idx]
                        gt_t_cr = gt_batch["T_CR_t_gt"][item_idx]
                        gt_q_cr = gt_batch["T_CR_q_gt"][item_idx]
                        gt_t_lr = gt_batch["T_LR_t_gt"][item_idx]
                        gt_q_lr = gt_batch["T_LR_q_gt"][item_idx]
                        T_cl_input_item = T_cl_input[item_idx]
                        T_cr_input_item = T_cr_input[item_idx]
                        T_lr_input_item = T_lr_input[item_idx]
                        flat_index = item_idx

                    pred_t_cl = corr_t_cl[flat_index]
                    pred_q_cl = corr_q_cl[flat_index]
                    pred_t_cr = corr_t_cr[flat_index]
                    pred_q_cr = corr_q_cr[flat_index]
                    pred_t_lr = corr_t_lr[flat_index]
                    pred_q_lr = corr_q_lr[flat_index]

                    input_err_t = {
                        "CL_cm": float(torch.norm(T_cl_input_item[:3, 3] - gt_t_cl).item() * 100.0),
                        "CR_cm": float(torch.norm(T_cr_input_item[:3, 3] - gt_t_cr).item() * 100.0),
                        "LR_cm": float(torch.norm(T_lr_input_item[:3, 3] - gt_t_lr).item() * 100.0),
                    }
                    input_err_r = {
                        "CL_deg": float(quaternion_distance(input_q["CL"][flat_index:flat_index + 1], gt_q_cl.unsqueeze(0), device).item() * 180.0 / math.pi),
                        "CR_deg": float(quaternion_distance(input_q["CR"][flat_index:flat_index + 1], gt_q_cr.unsqueeze(0), device).item() * 180.0 / math.pi),
                        "LR_deg": float(quaternion_distance(input_q["LR"][flat_index:flat_index + 1], gt_q_lr.unsqueeze(0), device).item() * 180.0 / math.pi),
                    }
                    pred_err_t = {
                        "CL_cm": float(torch.norm(pred_t_cl - gt_t_cl).item() * 100.0),
                        "CR_cm": float(torch.norm(pred_t_cr - gt_t_cr).item() * 100.0),
                        "LR_cm": float(torch.norm(pred_t_lr - gt_t_lr).item() * 100.0),
                    }
                    pred_err_r = {
                        "CL_deg": float(quaternion_distance(pred_q_cl.unsqueeze(0), gt_q_cl.unsqueeze(0), device).item() * 180.0 / math.pi),
                        "CR_deg": float(quaternion_distance(pred_q_cr.unsqueeze(0), gt_q_cr.unsqueeze(0), device).item() * 180.0 / math.pi),
                        "LR_deg": float(quaternion_distance(pred_q_lr.unsqueeze(0), gt_q_lr.unsqueeze(0), device).item() * 180.0 / math.pi),
                    }
                    result = {
                        "index": frame_index,
                        "time_index": time_idx if seq_len > 1 else 0,
                        "scene": meta.get("scene"),
                        "image_path": meta.get("image_path"),
                        "lidar_path": meta.get("lidar_path"),
                        "radar_path": meta.get("radar_path"),
                        "delta_T_CL_t": _tensor_to_list(delta_t_cl),
                        "delta_T_CL_q": _tensor_to_list(delta_q_cl),
                        "delta_T_CR_t": _tensor_to_list(delta_t_cr),
                        "delta_T_CR_q": _tensor_to_list(delta_q_cr),
                        "delta_T_LR_t": _tensor_to_list(delta_t_lr),
                        "delta_T_LR_q": _tensor_to_list(delta_q_lr),
                        "T_CL_t": _tensor_to_list(pred_t_cl),
                        "T_CL_q": _tensor_to_list(pred_q_cl),
                        "T_CR_t": _tensor_to_list(pred_t_cr),
                        "T_CR_q": _tensor_to_list(pred_q_cr),
                        "T_LR_t": _tensor_to_list(pred_t_lr),
                        "T_LR_q": _tensor_to_list(pred_q_lr),
                        "input_error_translation_cm": input_err_t,
                        "input_error_rotation_deg": input_err_r,
                        "pred_error_translation_cm": pred_err_t,
                        "pred_error_rotation_deg": pred_err_r,
                    }

                    if save_dir is not None:
                        loaded_img = base_dataset._load_image(meta["image_path"])
                        if isinstance(loaded_img, tuple):
                            rgb_img, calib = loaded_img
                        else:
                            rgb_img = loaded_img
                            calib = base_dataset.scene_info[meta["scene"]]["K"]
                        rgb_save_tensor = base_dataset.custom_transform(rgb_img, img_rotation=0.0, flip=False)
                        orig_hw = (rgb_img.height, rgb_img.width)
                        lidar_pc = _load_point_cloud(meta["lidar_path"], base_dataset.pcd_reader)
                        radar_pc = _load_point_cloud(meta["radar_path"], base_dataset.pcd_reader)

                        T_cl_gt = _pose_to_matrix(gt_t_cl, gt_q_cl)
                        T_cr_gt = _pose_to_matrix(gt_t_cr, gt_q_cr)
                        T_cl_input_np = T_cl_input_item.detach().cpu().numpy().astype(np.float32)
                        T_cr_input_np = T_cr_input_item.detach().cpu().numpy().astype(np.float32)
                        T_cl_pred = corr_T_cl[flat_index].detach().cpu().numpy().astype(np.float32)
                        T_cr_pred = corr_T_cr[flat_index].detach().cpu().numpy().astype(np.float32)

                        lidar_gt, _ = base_dataset._project_to_image(lidar_pc, T_cl_gt, calib, orig_hw)
                        lidar_input, _ = base_dataset._project_to_image(lidar_pc, T_cl_input_np, calib, orig_hw)
                        lidar_pred, _ = base_dataset._project_to_image(lidar_pc, T_cl_pred, calib, orig_hw)
                        radar_gt, _ = base_dataset._project_to_image(radar_pc, T_cr_gt, calib, orig_hw)
                        radar_input, _ = base_dataset._project_to_image(radar_pc, T_cr_input_np, calib, orig_hw)
                        radar_pred, _ = base_dataset._project_to_image(radar_pc, T_cr_pred, calib, orig_hw)

                        def _prep_proj(depth_map):
                            return torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0)

                        lidar_true_vis = _prep_proj(lidar_gt)
                        lidar_input_vis = _prep_proj(lidar_input)
                        lidar_pred_vis = _prep_proj(lidar_pred)
                        radar_true_vis = _prep_proj(radar_gt)
                        radar_input_vis = _prep_proj(radar_input)
                        radar_pred_vis = _prep_proj(radar_pred)

                        lidar_save_path = os.path.join(save_dir, f"frame_{frame_index:06d}_lidar.png")
                        radar_save_path = os.path.join(save_dir, f"frame_{frame_index:06d}_radar.png")
                        _save_triptych_png(lidar_save_path, rgb_save_tensor, lidar_true_vis, lidar_input_vis, lidar_pred_vis)
                        _save_triptych_png(radar_save_path, rgb_save_tensor, radar_true_vis, radar_input_vis, radar_pred_vis)
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
                    print(f"[infer_tri] saved prediction {len(results)} (scene={result['scene']}, index={result['index']}, t={result['time_index']})")
                    print(
                        f"[infer_tri]   delta T_CL_t={result['delta_T_CL_t']} delta T_CL_q={result['delta_T_CL_q']}\n"
                        f"[infer_tri]   delta T_CR_t={result['delta_T_CR_t']} delta T_CR_q={result['delta_T_CR_q']}\n"
                        f"[infer_tri]   delta T_LR_t={result['delta_T_LR_t']} delta T_LR_q={result['delta_T_LR_q']}"
                    )
                    print(
                        f"[infer_tri]   corrected T_CL_t={result['T_CL_t']} T_CL_q={result['T_CL_q']}\n"
                        f"[infer_tri]   corrected T_CR_t={result['T_CR_t']} T_CR_q={result['T_CR_q']}\n"
                        f"[infer_tri]   corrected T_LR_t={result['T_LR_t']} T_LR_q={result['T_LR_q']}"
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
    parser.add_argument("--data-root", required=True, help="Dataset root")
    parser.add_argument("--dataset", default="lg_innotek", choices=["lg_innotek", "hercules"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--train-scenes", nargs="*", default=["afternoon_parking_lot_1", "afternoon_campus_1"])
    parser.add_argument("--val-scenes", nargs="*", default=["afternoon_campus_2"])
    parser.add_argument("--val-frame-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--project-on-gpu", dest="project_on_gpu", action="store_true")
    parser.add_argument("--no-project-on-gpu", dest="project_on_gpu", action="store_false")
    parser.set_defaults(project_on_gpu=True)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--use-sequence", action="store_true")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument("--use-recurrence", action="store_true")
    parser.add_argument("--recurrent-hidden-dim", type=int, default=256)
    parser.add_argument("--max-r", type=float, default=5.0)
    parser.add_argument("--max-t", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--save", action="store_true", help="Save true/input/pred projection overlay images")
    parser.add_argument("--save-dir", default=None, help="Optional directory for saved projection images")
    parser.add_argument("--keep-original-size", action="store_true", help="Run inference with original image/projection size instead of resizing to 288x512")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
