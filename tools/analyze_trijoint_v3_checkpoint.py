#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DatasetLGInnotek import DatasetTriModalHercules, TriSequenceDataset
from losses_tri import TriModalPairwiseLoss
from models.tri_joint_v3.model import TriModalJointCalibNetV3Lite
from quaternion_distances import quaternion_distance
from utils import merge_inputs, project_pointcloud_to_image_torch, project_pointclouds_to_image_torch_batched


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a trained TriJointV3Lite checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-scene", default='["library_1"]')
    parser.add_argument("--train-scene", default='["SC_1","SC_3","island_1"]')
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--val-frame-limit", type=int, default=32)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--max-r", type=float, default=5.0)
    parser.add_argument("--max-t", type=float, default=0.5)
    return parser.parse_args()


def parse_scene_list(text: str) -> List[str]:
    return json.loads(text)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[TriModalJointCalibNetV3Lite, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TriModalJointCalibNetV3Lite(
        camera_pretrained=False,
        activation="leakyrelu",
        head_hidden_dim=256,
        head_dropout=0.0,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def make_val_loader(args, device: torch.device):
    dataset_val = DatasetTriModalHercules(
        args.data_root,
        max_r=args.max_r,
        max_t=args.max_t,
        split="val",
        use_reflectance=False,
        val_scene=parse_scene_list(args.val_scene),
        train_scene=parse_scene_list(args.train_scene),
        val_frame_limit=args.val_frame_limit,
        input_size=(288, 512),
        max_depth=args.max_depth,
        project_on_gpu=True,
        pointcloud_cache=True,
        pointcloud_cache_write=True,
    )
    dataset_val = TriSequenceDataset(
        dataset_val,
        seq_len=args.seq_len,
        stride=1,
        cache_size=8,
        project_on_gpu=True,
    )
    loader = DataLoader(
        dataset=dataset_val,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        collate_fn=merge_inputs,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    return loader


def prepare_rgb_batch(rgb_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    rgb_batch = rgb_batch.to(device, non_blocking=True).float()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    squeeze_seq = False
    if rgb_batch.ndim == 4:
        rgb_batch = rgb_batch.unsqueeze(1)
        squeeze_seq = True
    rgb_batch = (rgb_batch - mean) / std
    if squeeze_seq:
        rgb_batch = rgb_batch.squeeze(1)
    return rgb_batch


def project_batch_with_optional_vectorization(
    pc_tensor: torch.Tensor,
    pc_mask: torch.Tensor,
    T_tensor: torch.Tensor,
    calib_tensor: torch.Tensor,
    image_hw_tensor: torch.Tensor,
    max_depth: float,
    device: torch.device,
):
    flat_hw = image_hw_tensor.reshape(-1, 2)
    same_hw = bool(torch.all(flat_hw == flat_hw[:1]).item())
    if same_hw:
        return project_pointclouds_to_image_torch_batched(
            pc_tensor, pc_mask, T_tensor, calib_tensor, image_hw_tensor, max_depth
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
            points, flat_T[i], flat_calib[i], flat_hw[i].tolist(), max_depth
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


def build_tri_projection_batch(sample_batch: Dict[str, torch.Tensor], max_depth: float, device: torch.device):
    if "lidar_proj" in sample_batch and "radar_proj" in sample_batch:
        return (
            sample_batch["lidar_proj"].to(device, non_blocking=True),
            sample_batch["radar_proj"].to(device, non_blocking=True),
        )

    input_size = (288, 512)
    lidar_pc_seq = sample_batch["lidar_pc_seq"].to(device, non_blocking=True)
    radar_pc_seq = sample_batch["radar_pc_seq"].to(device, non_blocking=True)
    lidar_mask_seq = sample_batch["lidar_pc_seq_mask"].to(device, non_blocking=True)
    radar_mask_seq = sample_batch["radar_pc_seq_mask"].to(device, non_blocking=True)
    calib_seq = sample_batch["calib_seq"].to(device, non_blocking=True)
    image_hw_seq = sample_batch["image_hw_seq"].to(device, non_blocking=True)
    T_cl_input_seq = sample_batch["T_CL_input"].to(device, non_blocking=True)
    T_cr_input_seq = sample_batch["T_CR_input"].to(device, non_blocking=True)

    lidar_depth, _ = project_batch_with_optional_vectorization(
        lidar_pc_seq, lidar_mask_seq, T_cl_input_seq, calib_seq, image_hw_seq, max_depth, device
    )
    radar_depth, radar_aux = project_batch_with_optional_vectorization(
        radar_pc_seq, radar_mask_seq, T_cr_input_seq, calib_seq, image_hw_seq, max_depth, device
    )
    lidar_tensor = lidar_depth.unsqueeze(2)
    radar_tensor = torch.stack([radar_depth, radar_aux], dim=2)
    bsz, seq_len = lidar_tensor.shape[:2]
    lidar_tensor = F.interpolate(
        lidar_tensor.reshape(bsz * seq_len, 1, lidar_tensor.shape[-2], lidar_tensor.shape[-1]),
        size=input_size,
        mode="bilinear",
        align_corners=False,
    ).reshape(bsz, seq_len, 1, input_size[0], input_size[1])
    radar_tensor = F.interpolate(
        radar_tensor.reshape(bsz * seq_len, 2, radar_tensor.shape[-2], radar_tensor.shape[-1]),
        size=input_size,
        mode="bilinear",
        align_corners=False,
    ).reshape(bsz, seq_len, 2, input_size[0], input_size[1])
    return lidar_tensor, radar_tensor


def flatten_pose_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def flatten_matrix_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 3:
        return tensor
    return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])


def pose_to_matrix(t_batch: torch.Tensor, q_batch: torch.Tensor) -> torch.Tensor:
    loss_helper = TriModalPairwiseLoss()
    return loss_helper._pose_to_matrix(t_batch, q_batch)


def matrix_to_pose(T_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    loss_helper = TriModalPairwiseLoss()
    return loss_helper._matrix_to_pose(T_batch)


def apply_delta(pred_t: torch.Tensor, pred_q: torch.Tensor, input_T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_T = pose_to_matrix(pred_t, pred_q)
    flat_input = flatten_matrix_tensor(input_T)
    corrected_T = torch.bmm(delta_T, flat_input)
    corrected_t, corrected_q = matrix_to_pose(corrected_T)
    return corrected_t, corrected_q, corrected_T


def corrected_errors(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for pair, input_T, gt_t, gt_q in [
        ("CL", batch["T_CL_input"], batch["T_CL_t_gt"], batch["T_CL_q_gt"]),
        ("CR", batch["T_CR_input"], batch["T_CR_t_gt"], batch["T_CR_q_gt"]),
        ("LR", batch["T_LR_input"], batch["T_LR_t_gt"], batch["T_LR_q_gt"]),
    ]:
        pred_t, pred_q, _ = apply_delta(pred[f"T_{pair}_t"], pred[f"T_{pair}_q"], input_T)
        gt_t = flatten_pose_tensor(gt_t)
        gt_q = flatten_pose_tensor(gt_q)
        t_err_cm = torch.norm(pred_t - gt_t, dim=1).mean().item() * 100.0
        r_err_deg = quaternion_distance(pred_q, gt_q, pred_q.device).mean().item() * 180.0 / math.pi
        out[f"{pair}_t_cm"] = t_err_cm
        out[f"{pair}_r_deg"] = r_err_deg
    return out


def input_errors(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for pair, input_T, gt_t, gt_q in [
        ("CL", batch["T_CL_input"], batch["T_CL_t_gt"], batch["T_CL_q_gt"]),
        ("CR", batch["T_CR_input"], batch["T_CR_t_gt"], batch["T_CR_q_gt"]),
        ("LR", batch["T_LR_input"], batch["T_LR_t_gt"], batch["T_LR_q_gt"]),
    ]:
        pred_t, pred_q = matrix_to_pose(flatten_matrix_tensor(input_T))
        gt_t = flatten_pose_tensor(gt_t)
        gt_q = flatten_pose_tensor(gt_q)
        t_err_cm = torch.norm(pred_t - gt_t, dim=1).mean().item() * 100.0
        r_err_deg = quaternion_distance(pred_q, gt_q, pred_q.device).mean().item() * 180.0 / math.pi
        out[f"{pair}_t_cm"] = t_err_cm
        out[f"{pair}_r_deg"] = r_err_deg
    return out


def summarize_aux(aux: Dict[str, torch.Tensor]) -> Dict[str, float]:
    summary = {}
    for key in ["R_cam", "R_lid", "R_rad", "gate_z", "delta_z", "support_strength", "residual_strength"]:
        if key in aux:
            tensor = aux[key]
            summary[f"{key}_mean"] = tensor.mean().item()
            summary[f"{key}_std"] = tensor.std().item()
            summary[f"{key}_min"] = tensor.min().item()
            summary[f"{key}_max"] = tensor.max().item()
    for key in ["h_joint", "z_joint_fused", "z_joint_support_weighted", "z_joint_residual_weighted", "z_joint_ref"]:
        if key in aux:
            summary[f"{key}_norm_mean"] = aux[key].norm(dim=-1).mean().item()
    for key in ["align_cl", "align_cr", "align_lr"]:
        if key in aux:
            summary[f"{key}_best_sim_mean"] = aux[key][..., 0, :, :].mean().item() if aux[key].ndim == 5 else aux[key][:, 0].mean().item()
            summary[f"{key}_disp_norm_mean"] = aux[key][..., 2, :, :].mean().item() if aux[key].ndim == 5 else aux[key][:, 2].mean().item()
    if "invalid_penalty" in aux:
        penalty = aux["invalid_penalty"].mean(dim=0) if aux["invalid_penalty"].ndim == 2 else aux["invalid_penalty"].mean(dim=(0, 1))
        summary["invalid_penalty_cam"] = penalty[0].item()
        summary["invalid_penalty_lid"] = penalty[1].item()
        summary["invalid_penalty_rad"] = penalty[2].item()
    return summary


def flatten_aux_for_summary(aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in aux.items():
        if v.ndim >= 5:
            out[k] = v.reshape(-1, *v.shape[2:])
        elif v.ndim >= 3 and k in ["h_joint", "z_joint_fused", "z_joint_support_weighted", "z_joint_residual_weighted", "z_joint_ref", "delta_z", "gate_z", "dense_summary", "r_summary", "e_align_summary", "invalid_penalty"]:
            out[k] = v.reshape(-1, *v.shape[2:]) if v.ndim > 3 else v.reshape(-1, v.shape[-1])
        else:
            out[k] = v
    return out


def denormalize_rgb(rgb_norm: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = rgb_norm.detach().cpu().numpy()
    rgb = rgb * std + mean
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def save_visualizations(output_dir: Path, rgb, lidar_proj, radar_proj, aux, batch_idx: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_np = denormalize_rgb(rgb[0, 0])
    lidar_np = lidar_proj[0, 0, 0].detach().cpu().numpy()
    radar_depth_np = radar_proj[0, 0, 0].detach().cpu().numpy()
    radar_aux_np = radar_proj[0, 0, 1].detach().cpu().numpy()

    r_cam = aux["R_cam"][0, 0, 0].detach().cpu().numpy()
    r_lid = aux["R_lid"][0, 0, 0].detach().cpu().numpy()
    r_rad = aux["R_rad"][0, 0, 0].detach().cpu().numpy()
    align_cl_disp = aux["align_cl"][0, 0, 2].detach().cpu().numpy()
    align_cr_disp = aux["align_cr"][0, 0, 2].detach().cpu().numpy()
    align_lr_disp = aux["align_lr"][0, 0, 2].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(lidar_np, cmap="viridis")
    axes[0, 1].set_title("Lidar Projection")
    axes[0, 2].imshow(radar_depth_np, cmap="viridis")
    axes[0, 2].set_title("Radar Depth")
    axes[0, 3].imshow(radar_aux_np, cmap="magma")
    axes[0, 3].set_title("Radar Aux")
    axes[1, 0].imshow(r_cam, vmin=0.0, vmax=1.0, cmap="inferno")
    axes[1, 0].set_title("R_cam")
    axes[1, 1].imshow(r_lid, vmin=0.0, vmax=1.0, cmap="inferno")
    axes[1, 1].set_title("R_lid")
    axes[1, 2].imshow(r_rad, vmin=0.0, vmax=1.0, cmap="inferno")
    axes[1, 2].set_title("R_rad")
    axes[1, 3].imshow(aux["support_cam"][0, 0, 0].detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
    axes[1, 3].set_title("support_cam")
    axes[2, 0].imshow(align_cl_disp, cmap="magma")
    axes[2, 0].set_title("align_cl disp_norm")
    axes[2, 1].imshow(align_cr_disp, cmap="magma")
    axes[2, 1].set_title("align_cr disp_norm")
    axes[2, 2].imshow(align_lr_disp, cmap="magma")
    axes[2, 2].set_title("align_lr disp_norm")
    axes[2, 3].imshow(aux["residual_strength"][0, 0, 0].detach().cpu().numpy(), cmap="magma")
    axes[2, 3].set_title("residual_strength")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{batch_idx:02d}_maps.png", dpi=160)
    plt.close(fig)


def write_report(output_dir: Path, checkpoint_meta: dict, aux_summary: Dict[str, float], loss_rows: List[Dict[str, float]], corrected_rows: List[Dict[str, float]], input_rows: List[Dict[str, float]]):
    report_path = output_dir / "report.md"
    corrected_mean = {k: float(np.mean([row[k] for row in corrected_rows])) for k in corrected_rows[0]}
    input_mean = {k: float(np.mean([row[k] for row in input_rows])) for k in input_rows[0]}

    lines = [
        "# TriJointV3Lite Checkpoint Debug Report",
        "",
        "## Checkpoint",
        "",
        f"- Path: `{checkpoint_meta['checkpoint_path']}`",
        f"- Epoch: `{checkpoint_meta.get('epoch')}`",
        f"- Saved val loss: `{checkpoint_meta.get('val_loss')}`",
        "",
        "## Input vs Corrected Error",
        "",
    ]
    for pair in ["CL", "CR", "LR"]:
        lines.append(
            f"- `{pair}` input `{input_mean[f'{pair}_t_cm']:.2f} cm / {input_mean[f'{pair}_r_deg']:.2f} deg`, "
            f"corrected `{corrected_mean[f'{pair}_t_cm']:.2f} cm / {corrected_mean[f'{pair}_r_deg']:.2f} deg`"
        )
    lines.extend([
        "",
        "## Aux Summary",
        "",
    ])
    for key in sorted(aux_summary.keys()):
        lines.append(f"- `{key}`: `{aux_summary[key]:.4f}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if corrected_mean["CR_t_cm"] > input_mean["CR_t_cm"] + 10.0 and corrected_mean["LR_t_cm"] > input_mean["LR_t_cm"] + 10.0:
        lines.append("- The model is over-correcting range-related pairs on average. This is consistent with harmful delta predictions rather than conservative residual correction.")
    if aux_summary.get("R_cam_mean", 0.0) < 0.1:
        lines.append("- Camera reliability is very low on average, which suggests the dense camera path may still be under-utilized.")
    if aux_summary.get("invalid_penalty_rad", 0.0) > 0.05:
        lines.append("- Radar reliability still allocates noticeable mass on low-validity regions. This suggests the reliability path is not yet strongly suppressing weak radar evidence.")
    if aux_summary.get("gate_z_mean", 0.0) > 0.6:
        lines.append("- Refinement gate is fairly open on average, so the refinement block may be applying large latent updates early in training.")
    if aux_summary.get("align_cr_disp_norm_mean", 0.0) > aux_summary.get("align_cl_disp_norm_mean", 0.0):
        lines.append("- Camera-radar local alignment mismatch is larger than camera-lidar, which matches the weaker CR behavior seen in validation.")
    lines.extend([
        "",
        "## Files",
        "",
        "- `sample_00_maps.png`: first validation sample with reliability / alignment maps",
        "- `aux_summary.json`: averaged aux statistics",
        "- `loss_rows.json`: per-batch loss rows",
        "- `corrected_rows.json`: corrected pose errors",
    ])
    report_path.write_text("\n".join(lines))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    val_loader = make_val_loader(args, device)
    loss_fn = TriModalPairwiseLoss(w_t=1.0, w_q=1.0, lambda_loop=0.1)

    aux_summaries = []
    loss_rows = []
    corrected_rows = []
    input_rows = []

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= args.num_batches:
            break
        rgb = prepare_rgb_batch(sample["rgb"], device)
        lidar_proj, radar_proj = build_tri_projection_batch(sample, args.max_depth, device)
        batch = {
            "T_CL_t_gt": sample["T_CL_t_gt"].to(device, non_blocking=True),
            "T_CL_q_gt": sample["T_CL_q_gt"].to(device, non_blocking=True),
            "T_CR_t_gt": sample["T_CR_t_gt"].to(device, non_blocking=True),
            "T_CR_q_gt": sample["T_CR_q_gt"].to(device, non_blocking=True),
            "T_LR_t_gt": sample["T_LR_t_gt"].to(device, non_blocking=True),
            "T_LR_q_gt": sample["T_LR_q_gt"].to(device, non_blocking=True),
            "T_CL_input": sample["T_CL_input"].to(device, non_blocking=True),
            "T_CR_input": sample["T_CR_input"].to(device, non_blocking=True),
        }
        batch["T_LR_input"] = torch.linalg.inv(batch["T_CL_input"]) @ batch["T_CR_input"]

        with torch.no_grad():
            pred, new_state, aux = model(rgb, lidar_proj, radar_proj, state=None, return_aux=True)
            losses = loss_fn(pred, batch)

        loss_rows.append({
            "batch_idx": batch_idx,
            "total_loss": float(losses["total_loss"].item()),
            "loss_cl": float(losses["loss_cl"].item()),
            "loss_cr": float(losses["loss_cr"].item()),
            "loss_lr": float(losses["loss_lr"].item()),
            "loss_loop": float(losses["loss_loop"].item()),
        })
        corrected_rows.append(corrected_errors(pred, batch))
        input_rows.append(input_errors(batch))

        aux_flat = flatten_aux_for_summary(aux)
        aux_summaries.append(summarize_aux(aux_flat))

        if batch_idx == 0:
            save_visualizations(output_dir, rgb, lidar_proj, radar_proj, aux, batch_idx=batch_idx)

    aux_summary = {k: float(np.mean([row[k] for row in aux_summaries])) for k in aux_summaries[0]}
    checkpoint_meta = {
        "checkpoint_path": args.checkpoint,
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
    }
    write_report(output_dir, checkpoint_meta, aux_summary, loss_rows, corrected_rows, input_rows)

    (output_dir / "aux_summary.json").write_text(json.dumps(aux_summary, indent=2))
    (output_dir / "loss_rows.json").write_text(json.dumps(loss_rows, indent=2))
    (output_dir / "corrected_rows.json").write_text(json.dumps(corrected_rows, indent=2))
    (output_dir / "input_rows.json").write_text(json.dumps(input_rows, indent=2))
    print(f"[analyze_trijoint_v3_checkpoint] wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
